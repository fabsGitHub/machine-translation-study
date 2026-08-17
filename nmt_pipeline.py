import argparse
import codecs
import csv
import glob
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None

# reduce CUDA memory fragmentation across many experiments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

# Ensure required NLTK resources are available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# shared path constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "results")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")
SELF_PATH = os.path.abspath(__file__)



# Configuration loading (config.py)

DEFAULT_CONFIG = {
    "system": {
        "seed": 42,
        "float32_matmul_precision": "high",
    },
    "data": {
        "sample_rate": 0.1, 
        "test_split": 0.2,  
        "seed": 42,
        "max_word_len": 50,
        "max_char_len": 300,
        "max_vocab_size": 30000,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
    "training": {
        "epochs": 1,
        "batch_size_word": 128,
        "batch_size_char": 256,
        "grad_accum_steps": 1,
        "precision": "bfloat16",
        "use_8bit_adam": True,
    },
    "profiles": {
        "word": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 256,
            "hidden_dim": 256,
            "batch_size": 128,
        },
        "char": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 64,
            "hidden_dim": 512,
            "batch_size": 256,
        },
    },
}


def load_config(config_path="config/config.yaml"):
    """Loads config.yaml, falling back to defaults if missing."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    # merge instead of replace, so partial configs keep defaults
                    merged = DEFAULT_CONFIG.copy()
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                            merged[k] = {**merged[k], **v}
                        else:
                            merged[k] = v
                    return merged
        except Exception:
            pass
    return DEFAULT_CONFIG

# Shared utilities: logging, seeding, checkpoint-cache checks (utils.py)

class DualStreamTee:
    """Mirrors stdout/stderr to both the terminal and a log file."""
    def __init__(self, original_stream, log_file):
        """Stores the underlying stream and log file."""
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        """Writes a message to both the terminal and the log file."""
        self.original_stream.write(message)
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        """Flushes both streams."""
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()


def setup_logging(log_filename="execution.log", log_dir="data/results", rank=0):
    """Sets up logging to both terminal and file, and mirrors print() output."""
    if rank != 0:
        return None  # only rank 0 logs to file

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # mirror print() calls to the log file too
    log_file_obj = open(log_path, mode="a", encoding="utf-8")
    sys.stdout = DualStreamTee(sys.__stdout__, log_file_obj)
    sys.stderr = DualStreamTee(sys.__stderr__, log_file_obj)

    print(f"📝 Logging initialized -> Dual-streaming outputs to terminal and: {log_path}")
    return logger


def set_seed(seed=42, deterministic=False):
    """Sets random seeds and configures GPU precision/determinism settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # autotune convolution kernels for fixed input shapes
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def check_artifact_cache(output_dir, experiment_tags):
    """Checks whether a cached model + config exist for any of the given experiment tags."""
    for tag in experiment_tags:
        cfg = os.path.join(output_dir, f"best_config_{tag}.json")
        pt = os.path.join(output_dir, f"best_model_{tag}.pt")
        if os.path.exists(cfg) and os.path.exists(pt):
            return cfg, pt
    return None, None


def is_cache_valid(model_path, config_path):
    """Checks whether a cached model finished training successfully."""
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        return False
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            
        is_completed = bool(cfg.get("completed", False))
        
        return is_completed
        
    except Exception:
        return False



# Vocabulary, tokenization, Dataset/Sampler/DataLoader (dataset.py)


def _worker_init_fn(_worker_id):
    """Caps each DataLoader worker to 1 thread so workers don't fight over CPU cores."""
    torch.set_num_threads(1)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def pad_vocab_size(size, multiple=16):
    """Rounds a vocab size up to the nearest multiple."""
    return ((size + multiple - 1) // multiple) * multiple


def _build_vocab_worker(chunk, token_type):
    """Counts token frequencies in a chunk of sentences (word or char level)."""
    counts = Counter()
    if token_type == "char":
        for sentence in chunk:
            counts.update(str(sentence).strip())
    else:
        for sentence in chunk:
            counts.update(str(sentence).strip().split())
    return counts


def _numericalize_chunk_worker(
    chunk_src,
    chunk_trg,
    token_type,
    src_stoi,
    trg_stoi,
    src_max_idx,
    trg_max_idx,
):
    """Converts a chunk of text pairs into padded token-index arrays."""
    results = []
    src_get = src_stoi.get
    trg_get = trg_stoi.get
    is_char = token_type == "char"

    for src_text, trg_text in zip(chunk_src, chunk_trg):
        s_str = str(src_text).strip()
        t_str = str(trg_text).strip()

        src_tokens = list(s_str) if is_char else s_str.split()
        trg_tokens = list(t_str) if is_char else t_str.split()

        src_idx = [SOS_IDX]
        for t in src_tokens:
            idx = src_get(t, UNK_IDX)
            if not (0 <= idx <= src_max_idx):
                idx = UNK_IDX
            src_idx.append(idx)
        src_idx.append(EOS_IDX)

        trg_idx = [SOS_IDX]
        for t in trg_tokens:
            idx = trg_get(t, UNK_IDX)
            if not (0 <= idx <= trg_max_idx):
                idx = UNK_IDX
            trg_idx.append(idx)
        trg_idx.append(EOS_IDX)

        results.append((
            np.array(src_idx, dtype=np.int64),
            np.array(trg_idx, dtype=np.int64),
        ))
    return results


class Vocabulary:
    """Maps tokens (words or characters) to integer indices and back."""

    def __init__(self, token_type="word", pad_multiple=16, max_size=None):
        """Sets up the vocab with the 4 special tokens (PAD/UNK/SOS/EOS)."""
        self.token_type = token_type
        self.pad_multiple = pad_multiple
        self.max_size = max_size
        self.itos = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
        }
        self.stoi = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }

    def __len__(self):
        """Returns the vocab size."""
        return len(self.itos)

    @property
    def padded_size(self):
        """Returns the vocab size rounded up for GPU-friendly tensor shapes."""
        return pad_vocab_size(len(self.itos), multiple=self.pad_multiple)

    def tokenize(self, text):
        """Splits text into characters or words, depending on token_type."""
        text = str(text).strip()
        if self.token_type == "char":
            return list(text)
        return text.split()

    def build_vocab(self, sentence_list):
        """Builds the vocab from a list of sentences, keeping the most frequent tokens."""
        num_sentences = len(sentence_list)
        total_counts = Counter()

        if num_sentences >= 10000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_sentences + num_workers - 1) // num_workers
            chunks = [
                sentence_list[i : i + chunk_size]
                for i in range(0, num_sentences, chunk_size)
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _build_vocab_worker, chunk, self.token_type
                    )
                    for chunk in chunks
                ]
                for future in futures:
                    total_counts.update(future.result())
        else:
            for sentence in sentence_list:
                total_counts.update(self.tokenize(sentence))

        # keep only the most frequent tokens, rest fall back to <UNK>
        ranked_tokens = [tok for tok, _ in total_counts.most_common(self.max_size)]
        for token in ranked_tokens:
            if token not in self.stoi:
                idx = len(self.itos)
                self.stoi[token] = idx
                self.itos[idx] = token

    def numericalize(self, text):
        """Converts text into a list of vocab indices."""
        tokenized = self.tokenize(text)
        max_valid_idx = len(self.itos) - 1
        indices = []

        for token in tokenized:
            idx = self.stoi.get(token, UNK_IDX)
            if not (0 <= idx <= max_valid_idx):
                idx = UNK_IDX
            indices.append(idx)

        return indices


class PretokenizedNMTDataset(Dataset):
    """Dataset of tokenized sentence pairs, cached as flat index arrays on disk."""

    def __init__(
        self,
        csv_path,
        src_lang="de",
        trg_lang="en",
        token_type="word",
        src_vocab=None,
        trg_vocab=None,
        mock_mode=False,
    ):
        """Loads a cached tensor matrix if present, otherwise tokenizes the CSV and builds one."""
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.token_type = token_type

        cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.basename(csv_path).replace(".csv", "")
        cache_path = os.path.join(
            cache_dir, f"matrix_{base_name}_{token_type}.pt"
        )

        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            self.src_data = cached["src_data"]
            self.trg_data = cached["trg_data"]
            self.src_offsets = cached["src_offsets"]
            self.trg_offsets = cached["trg_offsets"]
            self.src_lengths = cached["src_lengths"]
            self.trg_lengths = cached["trg_lengths"]
            self.src_vocab = (
                src_vocab if src_vocab is not None else cached["src_vocab"]
            )
            self.trg_vocab = (
                trg_vocab if trg_vocab is not None else cached["trg_vocab"]
            )
            return

        df = pd.read_csv(csv_path)
        src_texts = df[src_lang].astype(str).tolist()
        trg_texts = df[trg_lang].astype(str).tolist()
        del df

        max_vocab_size = load_config().get("data", {}).get("max_vocab_size", 30000)

        if src_vocab is None:
            self.src_vocab = Vocabulary(token_type, max_size=max_vocab_size)
            self.src_vocab.build_vocab(src_texts)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocabulary(token_type, max_size=max_vocab_size)
            self.trg_vocab.build_vocab(trg_texts)
        else:
            self.trg_vocab = trg_vocab

        num_samples = len(src_texts)
        raw_data = []

        if num_samples >= 5000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_samples + num_workers - 1) // num_workers

            src_chunks = [
                src_texts[i : i + chunk_size]
                for i in range(0, num_samples, chunk_size)
            ]
            trg_chunks = [
                trg_texts[i : i + chunk_size]
                for i in range(0, num_samples, chunk_size)
            ]

            src_max_idx = len(self.src_vocab.itos) - 1
            trg_max_idx = len(self.trg_vocab.itos) - 1

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _numericalize_chunk_worker,
                        s_chunk,
                        t_chunk,
                        token_type,
                        self.src_vocab.stoi,
                        self.trg_vocab.stoi,
                        src_max_idx,
                        trg_max_idx,
                    )
                    for s_chunk, t_chunk in zip(src_chunks, trg_chunks)
                ]
                for future in futures:
                    raw_data.extend(future.result())
        else:
            for src, trg in zip(src_texts, trg_texts):
                src_num = np.array(
                    [SOS_IDX] + self.src_vocab.numericalize(src) + [EOS_IDX],
                    dtype=np.int64,
                )
                trg_num = np.array(
                    [SOS_IDX] + self.trg_vocab.numericalize(trg) + [EOS_IDX],
                    dtype=np.int64,
                )
                raw_data.append((src_num, trg_num))

        src_arrays = [pair[0] for pair in raw_data]
        trg_arrays = [pair[1] for pair in raw_data]

        self.src_lengths = np.array(
            [len(s) for s in src_arrays], dtype=np.int32
        )
        self.trg_lengths = np.array(
            [len(t) for t in trg_arrays], dtype=np.int32
        )

        self.src_offsets = np.zeros(num_samples + 1, dtype=np.int64)
        self.trg_offsets = np.zeros(num_samples + 1, dtype=np.int64)

        np.cumsum(self.src_lengths, out=self.src_offsets[1:])
        np.cumsum(self.trg_lengths, out=self.trg_offsets[1:])

        self.src_data = (
            np.concatenate(src_arrays)
            if src_arrays
            else np.array([], dtype=np.int64)
        )
        self.trg_data = (
            np.concatenate(trg_arrays)
            if trg_arrays
            else np.array([], dtype=np.int64)
        )

        del raw_data, src_arrays, trg_arrays

        torch.save({
            "src_data": self.src_data,
            "trg_data": self.trg_data,
            "src_offsets": self.src_offsets,
            "trg_offsets": self.trg_offsets,
            "src_lengths": self.src_lengths,
            "trg_lengths": self.trg_lengths,
            "src_vocab": self.src_vocab,
            "trg_vocab": self.trg_vocab,
        }, cache_path)
        print(f"⚡ Binary matrix cache saved -> {cache_path}")

    def __len__(self):
        """Returns the number of sentence pairs."""
        return len(self.src_offsets) - 1

    def __getitem__(self, idx):
        """Returns the src/trg index tensors for one sentence pair."""
        s_start, s_end = self.src_offsets[idx], self.src_offsets[idx + 1]
        t_start, t_end = self.trg_offsets[idx], self.trg_offsets[idx + 1]

        src_arr = self.src_data[s_start:s_end]
        trg_arr = self.trg_data[t_start:t_end]

        return torch.from_numpy(src_arr), torch.from_numpy(trg_arr)


class BucketBatchSampler(Sampler):
    """Batches similar-length sequences together to minimize padding waste."""

    def __init__(self, dataset, batch_size, shuffle=True, mega_batch_mult=100):
        """Precomputes sequence lengths used for bucketing."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_size = batch_size * mega_batch_mult

        if hasattr(self.dataset, "src_lengths"):
            self.lengths = self.dataset.src_lengths
        elif hasattr(self.dataset, "data"):
            self.lengths = np.array(
                [len(item[0]) for item in self.dataset.data], dtype=np.int32
            )
        else:
            self.lengths = np.array(
                [len(self.dataset[i][0]) for i in range(len(self.dataset))],
                dtype=np.int32,
            )

    def __iter__(self):
        """Yields batches of indices, sorted by length within shuffled mega-batches."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []
        lengths = self.lengths
        for i in range(0, len(indices), self.mega_batch_size):
            mega_batch = indices[i : i + self.mega_batch_size]
            sorted_order = mega_batch[np.argsort(lengths[mega_batch])]
            for j in range(0, len(sorted_order), self.batch_size):
                batch = sorted_order[j : j + self.batch_size]
                batches.append(batch)

        # drop a trailing partial batch, but only if another batch remains
        if self.shuffle and len(batches) > 1 and len(batches[-1]) < self.batch_size:
            batches.pop()

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        """Returns the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    """Pads a batch of variable-length sequences to the same length."""
    src_list, trg_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_list, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded


def get_dataloader(
    csv_path,
    batch_size=256,
    shuffle=True,
    src_vocab=None,
    trg_vocab=None,
    src_lang="de",
    trg_lang="en",
    token_type="word",
    num_workers=None,
):
    """Builds a DataLoader (bucketed if shuffling, else sequential) for a preprocessed CSV split."""
    if num_workers is None:
        configured = load_config().get("data", {}).get("num_workers")
        if configured is not None:
            num_workers = int(configured)
        else:
            # leave a core free for the main process/OS
            num_workers = max(1, min((os.cpu_count() or 4) - 1, 12))

    dataset = PretokenizedNMTDataset(
        csv_path=csv_path,
        src_lang=src_lang,
        trg_lang=trg_lang,
        token_type=token_type,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
    )

    if shuffle:
        sampler = BucketBatchSampler(
            dataset, batch_size=batch_size, shuffle=True
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    return loader, dataset.src_vocab, dataset.trg_vocab

# Pretrained embedding loading: GloVe / word2vec-style (embeddings.py)


def _get_cache_dir():
    """Returns (and creates) the folder used to cache parsed embedding files."""
    cache_dir = os.path.join("data", ".embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_and_extract_glove(glove_dir="data", emb_dim=300):
    """Downloads and extracts GloVe vectors if they do not already exist."""
    os.makedirs(glove_dir, exist_ok=True)
    txt_path = os.path.join(glove_dir, f"glove.6B.{emb_dim}d.txt")

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    url = "https://nlp.stanford.edu/data/glove.6B.zip"

    if not os.path.exists(zip_path):
        print(f"📥 Downloading GloVe embeddings from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"⚠️ Failed to download GloVe: {e}")
            return None

    print(f"📦 Extracting {zip_path} to {glove_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(glove_dir)
        print("✅ GloVe embeddings extracted successfully.")
    except Exception as e:
        print(f"⚠️ Failed to extract GloVe: {e}")
        return None

    return txt_path


def load_word2vec_keyed_vectors(filepath, binary=False):
    """Loads KeyedVectors using fast binary PyTorch disk caching to eliminate parse overhead."""
    cache_dir = _get_cache_dir()
    base_name = os.path.basename(filepath).replace(".", "_")
    pt_cache_path = os.path.join(cache_dir, f"cache_{base_name}.pt")

    if os.path.exists(pt_cache_path):
        try:
            return torch.load(pt_cache_path, weights_only=False)
        except Exception:
            pass

    # Rank 0 handles initial parsing if distributed
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
        if os.path.exists(pt_cache_path):
            return torch.load(pt_cache_path, weights_only=False)

    from gensim.models import KeyedVectors

    print(
        f"📦 Loading pre-trained vectors from {filepath} (Building fast binary cache)..."
    )
    wv = KeyedVectors.load_word2vec_format(filepath, binary=binary)

    vector_dict = {word: wv[word] for word in wv.key_to_index}
    torch.save(vector_dict, pt_cache_path)
    print(f"⚡ Saved fast binary embedding cache -> {pt_cache_path}")

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    return vector_dict


def _load_headerless_vector_dict(filepath, emb_dim=300):
    """Parses a headerless GloVe .txt vector file, with the same disk cache as load_word2vec_keyed_vectors."""
    cache_dir = _get_cache_dir()
    base_name = os.path.basename(filepath).replace(".", "_")
    pt_cache_path = os.path.join(cache_dir, f"cache_{base_name}.pt")

    if os.path.exists(pt_cache_path):
        try:
            return torch.load(pt_cache_path, weights_only=False)
        except Exception:
            pass

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
        if os.path.exists(pt_cache_path):
            return torch.load(pt_cache_path, weights_only=False)

    print(f"📦 Loading pre-trained vectors from {filepath} (Building fast binary cache)...")
    vector_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != emb_dim + 1:
                continue
            vector_dict[parts[0]] = np.array(parts[1:], dtype=np.float32)

    torch.save(vector_dict, pt_cache_path)
    print(f"⚡ Saved fast binary embedding cache -> {pt_cache_path}")

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    return vector_dict


def populate_embedding_matrix(vocab, vector_dict, emb_dim=300, token_type="word"):
    """Maps pre-trained vector dictionary to a vocabulary tensor matrix."""
    vocab_size = len(vocab)
    weights = torch.randn(vocab_size, emb_dim) * 0.01

    if token_type == "char":
        print("⚠️ [Word2Vec/GloVe] Token level is 'char'. Pre-trained word vectors are word-level. Using standard initialized embeddings.")
        return weights

    stoi = vocab.stoi if hasattr(vocab, "stoi") else getattr(vocab, "word2idx", {})
    found = 0
    special_tokens = {"<PAD>", "<UNK>", "<SOS>", "<EOS>"}

    for token, idx in stoi.items():
        if token in special_tokens:
            if token == "<PAD>":
                weights[idx] = torch.zeros(emb_dim)
            continue

        clean_token = token.strip(".,!?\"'()[]{}")
        candidates = [
            token,
            clean_token,
            token.lower(),
            clean_token.lower(),
            token.capitalize(),
            clean_token.capitalize(),
        ]

        matched_vec = None
        for cand in candidates:
            if cand in vector_dict:
                matched_vec = vector_dict[cand]
                break

        if matched_vec is not None:
            # truncate or pad the vector to the requested emb_dim
            vec_len = len(matched_vec)
            if vec_len > emb_dim:
                matched_vec = matched_vec[:emb_dim]
            elif vec_len < emb_dim:
                matched_vec = np.pad(matched_vec, (0, emb_dim - vec_len), mode="constant")

            weights[idx] = torch.from_numpy(matched_vec.copy())
            found += 1

    total_eval = max(1, len(stoi) - len(special_tokens))
    coverage = (found / total_eval) * 100.0
    print(f"✅ Loaded {found}/{total_eval} tokens ({coverage:.1f}%) from pre-trained vectors.")
    return weights

_WORD2VEC_FILES = {
    "en": ("GoogleNews-vectors-negative300.bin", True),
    "de": ("german.word2vec.bin", True),
    "sv": ("swedish.word2vec.bin", True),
}
_GLOVE_FILES = {
    "en": ("glove.6B.300d.txt", "glove_txt"),
    "de": ("german.word2vec.bin", True),
    "sv": ("swedish.word2vec.bin", True),
}


def generate_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    silent=False,
    pair_prefix=None,
    token_type="word",
    data_dir="data",
):
    """Loads pretrained Word2Vec embeddings for a vocabulary (language-specific file per lang)."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping Word2Vec loading.")
        return None

    filename, binary = _WORD2VEC_FILES.get(lang, _WORD2VEC_FILES["en"])
    vec_file = os.path.join(data_dir, filename)

    if not os.path.exists(vec_file):
        if not silent:
            print(f"⚠️ Vector file {vec_file} not found. Skipping.")
        return None

    try:
        vector_dict = load_word2vec_keyed_vectors(vec_file, binary=binary)
        return populate_embedding_matrix(
            vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load Word2Vec for {lang}: {e}")
        return None


def precompute_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    silent=False,
    pair_prefix=None,
    token_type="word",
):
    """Alias for generate_word2vec_embeddings (name expected by preprocess.py)."""
    return generate_word2vec_embeddings(
        vocab=vocab,
        train_csv=train_csv,
        lang=lang,
        emb_dim=emb_dim,
        silent=silent,
        pair_prefix=pair_prefix,
        token_type=token_type,
    )


def _load_pretrained_vector_dict(lang, source, emb_dim, data_dir, silent):
    """Loads the correct pretrained vector file for a given language and source ('glove' or 'word2vec')."""
    files = _GLOVE_FILES if source == "glove" else _WORD2VEC_FILES
    filename, mode = files.get(lang, files["en"])
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        if not silent:
            print(f"⚠️ Pretrained vector file {filepath} unavailable for lang={lang}.")
        return None

    if mode == "glove_txt":
        return _load_headerless_vector_dict(filepath, emb_dim=emb_dim)
    return load_word2vec_keyed_vectors(filepath, binary=bool(mode))


def load_glove_embeddings(
    vocab,
    glove_file_path=None,
    emb_dim=300,
    silent=False,
    token_type="word",
    glove_dir="data",
    lang="en",
):
    """Loads pretrained GloVe embeddings for a single vocabulary."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None

    if glove_file_path and os.path.exists(glove_file_path) and lang == "en":
        vector_dict = _load_headerless_vector_dict(glove_file_path, emb_dim=emb_dim)
    else:
        vector_dict = _load_pretrained_vector_dict(lang, "glove", emb_dim, glove_dir, silent)

    if vector_dict is None:
        if not silent:
            print("⚠️ GloVe embeddings file unavailable.")
        return None

    try:
        return populate_embedding_matrix(
            vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load GloVe embeddings: {e}")
        return None


def load_glove_embeddings_pair(
    src_vocab,
    trg_vocab,
    src_lang="de",
    trg_lang="en",
    emb_dim=300,
    glove_dir="data",
    silent=False,
    token_type="word",
):
    """Loads pretrained GloVe embeddings for a source/target vocab pair, one file per language."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None, None

    try:
        src_dict = _load_pretrained_vector_dict(src_lang, "glove", emb_dim, glove_dir, silent)
        trg_dict = (
            src_dict
            if trg_lang == src_lang
            else _load_pretrained_vector_dict(trg_lang, "glove", emb_dim, glove_dir, silent)
        )

        src_emb = (
            populate_embedding_matrix(src_vocab, src_dict, emb_dim=emb_dim, token_type=token_type)
            if src_dict is not None
            else None
        )
        trg_emb = (
            populate_embedding_matrix(trg_vocab, trg_dict, emb_dim=emb_dim, token_type=token_type)
            if trg_dict is not None
            else None
        )
        return src_emb, trg_emb
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load GloVe embeddings: {e}")
        return None, None

# Encoder / Decoder (Luong & Bahdanau attention) / Seq2Seq (models.py)



class Encoder(nn.Module):
    """RNN encoder: embeds the source sequence and runs it through an RNN/GRU/LSTM."""
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        n_layers=2,
        dropout=0.3,
        rnn_type="LSTM",
        bidirectional=True,
        pretrained_emb=None,
        freeze_emb=False,
        custom_emb_dim=None,
    ):
        """Builds the embedding layer (optionally pretrained) and the RNN."""
        super().__init__()
        self.rnn_type = rnn_type
        emb_dim_in = custom_emb_dim if custom_emb_dim else emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim_in)

        if pretrained_emb is not None:
            pretrained_tensor = (
                pretrained_emb
                if isinstance(pretrained_emb, torch.Tensor)
                else torch.as_tensor(pretrained_emb, dtype=torch.float32)
            )
            self.embedding.weight.data[: pretrained_tensor.size(0)].copy_(pretrained_tensor)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        self.project = (
            nn.Linear(emb_dim_in, emb_dim)
            if custom_emb_dim and custom_emb_dim != emb_dim
            else None
        )
        self.dropout = nn.Dropout(dropout)

        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, src):
        """Runs the source sequence through the embedding and RNN."""
        embedded = self.dropout(self.embedding(src))
        if self.project is not None:
            embedded = self.project(embedded)

        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class LuongAttention(nn.Module):
    """Multiplicative (Luong-style) attention over encoder outputs."""
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, enc_hidden_dim)

    def forward(self, hidden, encoder_outputs):
        """Returns attention weights over encoder outputs for the current decoder state."""
        score = torch.bmm(
            encoder_outputs, self.attn(hidden).unsqueeze(2)
        ).squeeze(2)
        return F.softmax(score, dim=1)


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) attention over encoder outputs."""
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(enc_hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, proj_enc_outputs=None):
        """Returns attention weights over encoder outputs for the current decoder state."""
        if proj_enc_outputs is None:
            proj_enc_outputs = self.U_a(encoder_outputs)

        energy = torch.tanh(self.W_a(hidden).unsqueeze(1) + proj_enc_outputs)
        score = self.v_a(energy).squeeze(2)
        return F.softmax(score, dim=1)


class Decoder(nn.Module):
    """RNN decoder with optional Luong/Bahdanau attention, one token at a time."""
    def __init__(
        self,
        vocab_size,
        emb_dim,
        enc_hidden_dim,
        hidden_dim,
        n_layers=2,
        dropout=0.3,
        rnn_type="LSTM",
        attention_type="none",
        pretrained_emb=None,
        freeze_emb=False,
        custom_emb_dim=None,
    ):
        """Builds the embedding, optional attention module, RNN, and output projection."""
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        emb_dim_in = custom_emb_dim if custom_emb_dim else emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim_in)

        if pretrained_emb is not None:
            pretrained_tensor = (
                pretrained_emb
                if isinstance(pretrained_emb, torch.Tensor)
                else torch.as_tensor(pretrained_emb, dtype=torch.float32)
            )
            self.embedding.weight.data[: pretrained_tensor.size(0)].copy_(pretrained_tensor)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        self.project = (
            nn.Linear(emb_dim_in, emb_dim)
            if custom_emb_dim and custom_emb_dim != emb_dim
            else None
        )
        self.dropout = nn.Dropout(dropout)

        if attention_type == "luong":
            self.attention = LuongAttention(hidden_dim, enc_hidden_dim)
        elif attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        else:
            self.attention = None

        rnn_in_dim = emb_dim + (
            enc_hidden_dim if attention_type != "none" else 0
        )
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            rnn_in_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        fc_in_dim = hidden_dim + (
            enc_hidden_dim if attention_type != "none" else 0
        )
        self.fc_out = nn.Linear(fc_in_dim, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs, proj_enc_outputs=None):
        """Decodes a single output token given the previous token and hidden state."""
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        if self.project is not None:
            embedded = self.project(embedded)

        if self.attention_type != "none":
            h_top = (
                hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            )
            if self.attention_type == "bahdanau":
                attn_weights = self.attention(h_top, encoder_outputs, proj_enc_outputs=proj_enc_outputs)
            else:
                attn_weights = self.attention(h_top, encoder_outputs)

            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            )
            rnn_in = torch.cat((embedded, context), dim=2)
        else:
            context = None
            attn_weights = None
            rnn_in = embedded

        output, hidden = self.rnn(rnn_in, hidden)

        if self.attention_type != "none":
            fc_in = torch.cat((output, context), dim=2)
        else:
            fc_in = output

        prediction = self.fc_out(fc_in.squeeze(1))
        return prediction, hidden, attn_weights

    def forward(
        self, trg, hidden, encoder_outputs, teacher_forcing_ratio=0.4
    ):
        """Decodes the full target sequence, using teacher forcing during training."""
        batch_size, trg_len = trg.shape

        proj_enc = (
            self.attention.U_a(encoder_outputs)
            if self.attention_type == "bahdanau"
            else None
        )

        use_teacher_forcing = self.training and (
            (torch.rand(1, device=trg.device).item() < teacher_forcing_ratio)
            if teacher_forcing_ratio > 0.0
            else False
        )

        if use_teacher_forcing and self.attention_type == "none":
            trg_in = trg[:, :-1]
            embedded = self.dropout(self.embedding(trg_in))
            if self.project is not None:
                embedded = self.project(embedded)

            rnn_out, _ = self.rnn(embedded, hidden)
            predictions = self.fc_out(rnn_out)

            zero_step = torch.zeros(
                (batch_size, 1, self.vocab_size), device=trg.device, dtype=predictions.dtype
            )
            return torch.cat([zero_step, predictions], dim=1)

        outputs = torch.zeros(
            batch_size, trg_len, self.vocab_size, device=trg.device, dtype=encoder_outputs.dtype
        )
        input_token = trg[:, 0]

        if self.training and teacher_forcing_ratio > 0.0:
            use_tf_steps = torch.rand(trg_len - 1, device=trg.device) < teacher_forcing_ratio
        else:
            use_tf_steps = torch.zeros(trg_len - 1, dtype=torch.bool, device=trg.device)

        for t in range(1, trg_len):
            pred, hidden, _ = self.forward_step(
                input_token, hidden, encoder_outputs, proj_enc_outputs=proj_enc
            )
            outputs[:, t] = pred
            next_tf = trg[:, t]
            input_token = torch.where(use_tf_steps[t - 1], next_tf, pred.argmax(dim=1))

        return outputs

class Seq2Seq(nn.Module):
    """Wraps an Encoder and Decoder into one translation model."""
    def __init__(self, encoder, decoder, device=None):
        """Stores the encoder/decoder and sets up a bridge layer if their hidden sizes differ."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if encoder.rnn.bidirectional:
            enc_hidden_dim = encoder.rnn.hidden_size * 2
            dec_hidden_dim = decoder.hidden_dim
            if enc_hidden_dim != dec_hidden_dim:
                self.bridge_h = nn.Linear(enc_hidden_dim, dec_hidden_dim)
                self.bridge_c = (
                    nn.Linear(enc_hidden_dim, dec_hidden_dim)
                    if encoder.rnn_type == "LSTM"
                    else None
                )
            else:
                self.bridge_h = None
                self.bridge_c = None
        else:
            self.bridge_h = None
            self.bridge_c = None

    def _bridge_hidden(self, hidden):
        """Projects the encoder's (bidirectional) final hidden state to the decoder's hidden size."""
        if not self.encoder.rnn.bidirectional or self.bridge_h is None:
            return hidden

        if self.encoder.rnn_type == "LSTM":
            h, c = hidden
            n_layers = self.encoder.rnn.num_layers
            h_cat = torch.cat([h[0:n_layers], h[n_layers:]], dim=2)
            c_cat = torch.cat([c[0:n_layers], c[n_layers:]], dim=2)
            h_bridged = torch.tanh(self.bridge_h(h_cat))
            c_bridged = torch.tanh(self.bridge_c(c_cat))
            return (h_bridged, c_bridged)
        else:
            n_layers = self.encoder.rnn.num_layers
            h_cat = torch.cat([hidden[0:n_layers], hidden[n_layers:]], dim=2)
            return torch.tanh(self.bridge_h(h_cat))

    def forward(self, src, trg, teacher_forcing_ratio=0.4):
        """Encodes the source, then decodes the target sequence."""
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._bridge_hidden(hidden)
        outputs = self.decoder(
            trg, hidden, encoder_outputs, teacher_forcing_ratio=teacher_forcing_ratio
        )
        return outputs

# Corpus download, cleaning, sampling, train/val/test splitting (preprocess.py)



MOCK_DATA_DE_EN = {
    "de": [
        "hallo welt",
        "wie geht es dir heute",
        "maschinelles lernen macht unglaublichen spass",
        "der schnelle braune fuchs springt ueber den faulen hund",
        "tiefe neuronale netze erfordern strukturelle optimierung",
        "hallo welt und maschinelles lernen",
        "wie geht es den tiefen neuronalen netzen",
        "der schnelle fuchs macht unglaublichen spass",
    ],
    "en": [
        "hello world",
        "how are you today",
        "machine learning is incredibly fun",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks require structural optimization",
        "hello world and machine learning",
        "how are deep neural networks doing today",
        "the quick fox is incredibly fun",
    ],
}

MOCK_DATA_SV_EN = {
    "en": [
        "hello world",
        "how are you today",
        "machine learning is incredibly fun",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks require structural optimization",
        "hello world and machine learning",
        "how are deep neural networks doing today",
        "the quick fox is incredibly fun",
    ],
    "sv": [
        "hej världen",
        "hur mår du idag",
        "maskininlärning är otroligt roligt",
        "den snabba bruna räven hoppar över den lata hunden",
        "djupa neurala nätverk kräver strukturell optimering",
        "hej världen och maskininlärning",
        "hur mår de djupa neurala nätverken idag",
        "den snabba räven är otroligt rolig",
    ],
}


def get_split_path(processed_dir, split_type, src_lang, trg_lang):
    """Generates standardized mutual filename: <split>_<src>_<trg>.csv"""
    return os.path.join(processed_dir, f"{split_type}_{src_lang}_{trg_lang}.csv")


def splits_already_exist(processed_dir):
    """Checks if all required language pair split CSV files already exist."""
    pairs = [("de", "en"), ("en", "de"), ("en", "sv")]
    splits = ["train", "val", "test"]
    for src, trg in pairs:
        for split in splits:
            if not os.path.exists(get_split_path(processed_dir, split, src, trg)):
                return False
    return True


def locate_raw_files(raw_dir, lang_pair="de-en"):
    """Robust multi-tier file locator across data/raw, Kaggle inputs, and custom structures."""
    l1, l2 = lang_pair.split("-")

    # 1. Direct standard expected paths in data/raw/
    f1_std = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l1}")
    f2_std = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l2}")
    if os.path.exists(f1_std) and os.path.exists(f2_std):
        return f1_std, f2_std

    # 2. Search locally inside data/raw/
    if os.path.exists(raw_dir):
        f1_cand, f2_cand = None, None
        for f in os.listdir(raw_dir):
            f_lower = f.lower()
            if lang_pair in f_lower or "europarl" in f_lower:
                if f_lower.endswith(f".{l1}"):
                    f1_cand = os.path.join(raw_dir, f)
                elif f_lower.endswith(f".{l2}"):
                    f2_cand = os.path.join(raw_dir, f)
        if f1_cand and f2_cand:
            return f1_cand, f2_cand

    # 3. Search /kaggle/input directory tree
    if os.path.exists("/kaggle/input"):
        f1_cand, f2_cand = None, None
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                f_lower = f.lower()
                if lang_pair in f_lower or "europarl" in f_lower:
                    if f_lower.endswith(f".{l1}") and (
                        f1_cand is None or lang_pair in f_lower
                    ):
                        f1_cand = os.path.join(root, f)
                    elif f_lower.endswith(f".{l2}") and (
                        f2_cand is None or lang_pair in f_lower
                    ):
                        f2_cand = os.path.join(root, f)
        if f1_cand and f2_cand:
            return f1_cand, f2_cand

    return None, None


def download_and_extract_europarl(raw_dir, lang_pair="de-en"):
    """Generic handler for downloading and extracting Europarl datasets."""
    f1, f2 = locate_raw_files(raw_dir, lang_pair)
    if f1 and f2 and os.path.exists(f1) and os.path.exists(f2):
        print(f"✓ Europarl {lang_pair.upper()} text files already present locally.")
        return f1, f2

    url = f"https://www.statmt.org/europarl/v7/{lang_pair}.tgz"
    tar_path = os.path.join(raw_dir, f"{lang_pair}.tgz")

    if not os.path.exists(tar_path):
        print(f"Downloading Europarl {lang_pair.upper()} dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    print(f"Extracting {lang_pair} tarball...")
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=raw_dir, filter="data")
        else:
            tar.extractall(path=raw_dir)
    print("✓ Extraction complete.")

    l1, l2 = lang_pair.split("-")
    f1 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l1}")
    f2 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l2}")
    return f1, f2


def _fast_read_lines(filepath):
    """Fast vectorized text file reading using PyArrow/C engine where available."""
    try:
        df = pd.read_csv(
            filepath,
            sep="\n",
            header=None,
            engine="pyarrow",
            quoting=3,
            on_bad_lines="skip",
            dtype_backend="pyarrow",
        )
        return df[0]
    except Exception:
        try:
            df = pd.read_csv(
                filepath,
                sep="\n",
                header=None,
                engine="c",
                quoting=3,
                on_bad_lines="skip",
            )
            return df[0]
        except Exception:
            with open(filepath, "r", encoding="utf-8") as f:
                return pd.Series(f.read().splitlines())


def preprocess_data(
    df,
    src_col="de",
    trg_col="en",
    token_type="word",
    max_word_len=64,
    max_char_len=256,
):
    """Cleans, lowercases, deduplicates, and length-filters a raw sentence-pair DataFrame."""
    df = df.copy()

    try:
        df[src_col] = df[src_col].astype("string[pyarrow]")
        df[trg_col] = df[trg_col].astype("string[pyarrow]")
    except Exception:
        df[src_col] = df[src_col].astype(str)
        df[trg_col] = df[trg_col].astype(str)

    df[src_col] = df[src_col].str.strip()
    df[trg_col] = df[trg_col].str.strip()

    # Filter empty rows and XML tag markers
    df = df[(df[src_col] != "") & (df[trg_col] != "")]
    df = df[~df[src_col].str.startswith("<") & ~df[trg_col].str.startswith("<")]

    df[src_col] = df[src_col].str.lower()
    df[trg_col] = df[trg_col].str.lower()

    # Punctuation isolation regex
    punct_regex = r"([.,!?\"':;)(])"
    df[src_col] = (
        df[src_col]
        .str.replace(punct_regex, r" \1 ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[trg_col] = (
        df[trg_col]
        .str.replace(punct_regex, r" \1 ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.drop_duplicates()

    def get_word_len(series):
        """Counts words per row via space count + 1."""
        return series.str.count(" ") + 1

    if token_type == "char":
        df["src_len"] = df[src_col].str.len()
        df["trg_len"] = df[trg_col].str.len()
        df = df[(df["src_len"] <= max_char_len) & (df["trg_len"] <= max_char_len)]
        df = df.drop(columns=["src_len", "trg_len"])
    elif token_type == "both":
        src_w_len = get_word_len(df[src_col])
        trg_w_len = get_word_len(df[trg_col])
        src_c_len = df[src_col].str.len()
        trg_c_len = df[trg_col].str.len()
        df = df[
            (src_w_len <= max_word_len)
            & (trg_w_len <= max_word_len)
            & (src_c_len <= max_char_len)
            & (trg_c_len <= max_char_len)
        ]
    else:  # "word"
        src_w_len = get_word_len(df[src_col])
        trg_w_len = get_word_len(df[trg_col])
        df = df[(src_w_len <= max_word_len) & (trg_w_len <= max_word_len)]

    return df.reset_index(drop=True)


def process_and_save_pair(
    df, src_lang, trg_lang, processed_dir, test_split, seed, mock=False
):
    """Splits data and saves to standard mutual paths: <split>_<src>_<trg>.csv once."""
    if mock:
        train_df = df
        val_df = df.iloc[3:4]
        test_df = df.iloc[4:]
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=seed
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.10, random_state=seed
        )
        print(
            f"📦 [{src_lang.upper()}-{trg_lang.upper()} SPLITS] Train:"
            f" {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )

    train_path = get_split_path(processed_dir, "train", src_lang, trg_lang)
    val_path = get_split_path(processed_dir, "val", src_lang, trg_lang)
    test_path = get_split_path(processed_dir, "test", src_lang, trg_lang)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


def _cache_single_pair(src, trg, processed_dir, token_type):
    """Pre-tokenizes one language pair's CSV splits and caches binary tensors + embedding matrices."""
    pair_tag = f"{src}_{trg}"
    train_csv = get_split_path(processed_dir, "train", src, trg)
    val_csv = get_split_path(processed_dir, "val", src, trg)
    test_csv = get_split_path(processed_dir, "test", src, trg)

    if os.path.exists(train_csv):
        print(
            "\n📦 Pre-tokenizing & caching binary tensors for pair:"
            f" {src.upper()} -> {trg.upper()}"
        )
        train_ds = PretokenizedNMTDataset(
            train_csv, src_lang=src, trg_lang=trg, token_type=token_type
        )
        if os.path.exists(val_csv):
            PretokenizedNMTDataset(
                val_csv,
                src_lang=src,
                trg_lang=trg,
                token_type=token_type,
                src_vocab=train_ds.src_vocab,
                trg_vocab=train_ds.trg_vocab,
            )
        if os.path.exists(test_csv):
            PretokenizedNMTDataset(
                test_csv,
                src_lang=src,
                trg_lang=trg,
                token_type=token_type,
                src_vocab=train_ds.src_vocab,
                trg_vocab=train_ds.trg_vocab,
            )

        precompute_enabled = load_config().get("data", {}).get("precompute_word2vec_cache", False)
        if token_type in ["word", "both"] and precompute_enabled:
            matrix_cache_dir = os.path.join(processed_dir, ".matrix_cache")
            os.makedirs(matrix_cache_dir, exist_ok=True)

            for dim in [128, 256]:
                for lang, vocab in [(src, train_ds.src_vocab), (trg, train_ds.trg_vocab)]:
                    cache_filename = f"emb_matrix_{pair_tag}_{lang}_{dim}d_{token_type}.pt"
                    cache_path = os.path.join(matrix_cache_dir, cache_filename)

                    if os.path.exists(cache_path):
                        print(f"⚡ Embedding matrix cache loaded -> {cache_path}")
                    else:
                        weights = precompute_word2vec_embeddings(
                            vocab=vocab,
                            train_csv=train_csv,
                            lang=lang,
                            emb_dim=dim,
                            pair_prefix=pair_tag,
                            token_type=token_type,
                        )
                        if weights is not None:
                            torch.save(weights, cache_path)
                            print(f"⚡ Saved embedding matrix cache -> {cache_path}")


def execute_offline_caching(processed_dir, token_type="word"):
    """Pre-tokenizes and caches binary tensors + embeddings for all 3 language pairs."""
    print("\n" + "─" * 75)
    print("⚡ [OFFLINE BINARY CACHING & TENSOR PRE-SERIALIZATION]")
    print("─" * 75)

    pairs = [("de", "en"), ("en", "de"), ("en", "sv")]

    for src, trg in pairs:
        _cache_single_pair(src, trg, processed_dir, token_type)


def preprocess_main():
    """CLI entry point: downloads, cleans, splits, and caches all 3 language pairs."""
    parser = argparse.ArgumentParser(
        description="NMT Pipeline Preprocessing Stage"
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--token_type",
        type=str,
        default="word",
        choices=["word", "char", "both"],
    )
    args = parser.parse_args()

    setup_logging(log_filename=f"preprocess_{args.token_type}.log", log_dir="data/results")

    config = load_config()
    sample_rate = config.get("data", {}).get("sample_rate", 1.0)
    test_split = config.get("data", {}).get("test_split", 0.1)
    seed = config.get("data", {}).get("seed", 42)
    max_word_len = config.get("data", {}).get("max_word_len", 64)
    max_char_len = config.get("data", {}).get("max_char_len", 256)

    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # skip re-downloading/cleaning if splits already exist
    if not args.mock and splits_already_exist(processed_dir):
        print("✓ Processed dataset CSV splits already exist locally. Skipping raw text cleaning.")
        execute_offline_caching(processed_dir, token_type=args.token_type)
        print("\n✓ Dataset preprocessing and binary caching completed successfully.")
        return

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if "glove" in f.lower() and "300d" in f.lower() and f.endswith(".txt"):
                    glove_dest = os.path.join(ROOT_DIR, "data", "glove.6B.300d.txt")
                    if not os.path.exists(glove_dest):
                        try:
                            os.symlink(os.path.join(root, f), glove_dest)
                        except Exception:
                            shutil.copy(os.path.join(root, f), glove_dest)
                    break

    # 1. GERMAN - ENGLISH PATHWAY
    if args.mock:
        cleaned_de_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_DE_EN),
            src_col="de",
            trg_col="en",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )
    else:
        de_file, en_file = locate_raw_files(raw_dir, "de-en")
        if not de_file or not en_file:
            de_file, en_file = download_and_extract_europarl(raw_dir, "de-en")

        download_and_extract_glove(os.path.dirname(raw_dir))

        de_sentences = _fast_read_lines(de_file)
        en_sentences = _fast_read_lines(en_file)
        raw_df = pd.DataFrame({"de": de_sentences, "en": en_sentences})

        sampled_df = raw_df.sample(
            frac=min(1.0, sample_rate), random_state=seed
        ).reset_index(drop=True)
        cleaned_de_df = preprocess_data(
            sampled_df,
            src_col="de",
            trg_col="en",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )

    process_and_save_pair(
        cleaned_de_df,
        src_lang="de",
        trg_lang="en",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )
    process_and_save_pair(
        cleaned_de_df,
        src_lang="en",
        trg_lang="de",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )

    # 2. ENGLISH - SWEDISH PATHWAY
    if args.mock:
        cleaned_sv_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_SV_EN),
            src_col="en",
            trg_col="sv",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )
    else:
        sv_file, en_sv_file = locate_raw_files(raw_dir, "sv-en")
        if not sv_file or not en_sv_file:
            sv_file, en_sv_file = download_and_extract_europarl(raw_dir, "sv-en")

        sv_sentences = _fast_read_lines(sv_file)
        en_sv_sentences = _fast_read_lines(en_sv_file)
        raw_sv_df = pd.DataFrame({"en": en_sv_sentences, "sv": sv_sentences})

        sampled_sv_df = raw_sv_df.sample(
            frac=min(1.0, sample_rate), random_state=seed
        ).reset_index(drop=True)
        cleaned_sv_df = preprocess_data(
            sampled_sv_df,
            src_col="en",
            trg_col="sv",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )

    process_and_save_pair(
        cleaned_sv_df,
        src_lang="en",
        trg_lang="sv",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )

    # 3. OFFLINE BINARY CACHING
    execute_offline_caching(processed_dir, token_type=args.token_type)

    print("\n✓ Dataset preprocessing and binary caching completed successfully.")



# Inference, BLEU/METEOR scoring, reporting (evaluate.py)

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None


# Ensure required NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


# Helper Utilities & Inference

def idx_to_tokens(indices, vocab):
    """Converts token indices back to readable string tokens."""
    if hasattr(vocab, 'get_itos'):
        itos = vocab.get_itos()
        tokens = [itos[i] for i in indices]
    elif hasattr(vocab, 'itos'):
        tokens = [vocab.itos[i] for i in indices]
    elif isinstance(vocab, dict):
        inv_vocab = {v: k for k, v in vocab.items()}
        tokens = [inv_vocab.get(i, "<unk>") for i in indices]
    else:
        tokens = [str(i) for i in indices]
    return tokens


def build_model_from_checkpoint(checkpoint, device):
    """Reconstructs the Seq2Seq model architecture from checkpoint metadata."""
    cfg = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    trg_vocab = checkpoint['trg_vocab']

    num_directions = 2 if cfg.get('bidirectional', True) else 1
    emb_dim = cfg.get('emb_dim', 256)
    hidden_dim = cfg.get('hidden_dim', 512)
    dropout = cfg.get('dropout', 0.3)
    rnn_type = cfg.get('rnn_type', 'LSTM')
    attention_type = cfg.get('attention_type', 'none')
    embedding_source = cfg.get('embedding_source', 'scratch')
    freeze_emb = cfg.get('freeze_emb', False)

    emb_override = 300 if embedding_source == 'glove' else None

    encoder = Encoder(
        len(src_vocab), emb_dim, hidden_dim, 2, dropout,
        rnn_type, cfg.get('bidirectional', True), None, freeze_emb, emb_override
    )
    decoder = Decoder(
        len(trg_vocab), emb_dim, hidden_dim * num_directions, hidden_dim, 2,
        dropout, rnn_type, attention_type, None, freeze_emb, emb_override
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    clean_state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in checkpoint['model_state_dict'].items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model, src_vocab, trg_vocab, cfg


def translate_sentence(model, src_tokens, src_vocab, trg_vocab, device, max_len=50):
    """Translates a source sequence and captures target output and attention matrix."""
    model.eval()

    if hasattr(src_vocab, 'stoi'):
        src_indices = [SOS_IDX] + [src_vocab.stoi.get(tok, UNK_IDX) for tok in src_tokens] + [EOS_IDX]
    elif hasattr(src_vocab, '__getitem__'):
        src_indices = [SOS_IDX] + [src_vocab[tok] if tok in src_vocab else src_vocab.get('<unk>', 0) for tok in src_tokens] + [EOS_IDX]
    else:
        src_indices = [SOS_IDX] + [src_vocab.get(tok, 0) for tok in src_tokens] + [EOS_IDX]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        # bridge the encoder's hidden state (Seq2Seq.forward normally does this)
        hidden = model._bridge_hidden(hidden)

        trg_indexes = [SOS_IDX]
        attentions = []

        # precompute Bahdanau's encoder projection once instead of per step
        proj_enc = (
            model.decoder.attention.U_a(encoder_outputs)
            if getattr(model.decoder, 'attention_type', None) == 'bahdanau'
            else None
        )

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            # greedy decoding uses forward_step() one token at a time
            output, hidden, attn = model.decoder.forward_step(
                trg_tensor, hidden, encoder_outputs, proj_enc_outputs=proj_enc
            )

            if attn is not None:
                attentions.append(attn.squeeze(0).cpu().detach().numpy())

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break

    translated_tokens = idx_to_tokens(trg_indexes[1:], trg_vocab)
    if translated_tokens and translated_tokens[-1] == "<eos>":
        translated_tokens = translated_tokens[:-1]

    attn_matrix = np.array(attentions) if len(attentions) > 0 else None
    return translated_tokens, attn_matrix


# Primary Required Functions

def visualize_attention(model_path, src_sentence=None, save_path=None, device=None):
    """Generates and saves an attention heatmap for a sample input sentence."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint missing: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model, src_vocab, trg_vocab, cfg = build_model_from_checkpoint(checkpoint, device)

    if cfg.get("attention_type", "none") == "none":
        print(f"⚠️ Model at {model_path} does not use attention (attention_type='none'). Skipping visualization.")
        return

    if src_sentence is None:
        src_sentence = "ein kleiner hund läuft über den rasen ."

    token_type = cfg.get("token_type", "word")
    src_tokens = list(src_sentence) if token_type == "char" else src_sentence.strip().split()

    translated_tokens, attn_matrix = translate_sentence(model, src_tokens, src_vocab, trg_vocab, device)

    if attn_matrix is None or attn_matrix.size == 0:
        print("⚠️ No attention weights captured during inference.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix[:len(translated_tokens), :len(src_tokens) + 2],
        xticklabels=["<sos>"] + src_tokens + ["<eos>"],
        yticklabels=translated_tokens,
        cmap="viridis",
        annot=False
    )
    plt.xlabel("Source Sequence")
    plt.ylabel("Target Sequence")
    plt.title(f"Attention Map ({cfg.get('experiment', 'NMT')} - {cfg.get('attention_type', 'Luong').upper()})")
    plt.tight_layout()

    if save_path is None:
        exp_name = cfg.get('experiment', 'attention_map')
        save_path = os.path.join(ROOT_DIR, "data", "results", f"{exp_name}_attention.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Attention heatmap visualization saved to: {save_path}")


def generate_all_reports(token_type="word", output_dir=None):
    """Aggregates all experiment JSON outputs into unified summary tables (CSV and JSON)."""
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "data", "results")

    json_files = glob.glob(os.path.join(output_dir, "best_config_*.json"))
    if not json_files:
        print(f"⚠️ No result json logs found in {output_dir}")
        return None

    records = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if token_type and data.get("token_type", "word") != token_type:
                continue

            # skip tuning-stage bookkeeping files, which have no "experiment" key
            if not data.get("experiment"):
                continue

            records.append({
                "Experiment": data.get("experiment", "N/A"),
                "RNN Type": data.get("rnn_type", "LSTM"),
                "Attention": data.get("attention_type", "none"),
                "Token Type": data.get("token_type", "word"),
                "Embedding": data.get("embedding_source", "scratch"),
                "BLEU": data.get("bleu", data.get("bleu_score", None)),
                "METEOR": data.get("meteor", data.get("mean_meteor", None)),
                "Best Val Loss": data.get("best_val_loss", None),
                "Epochs Trained": data.get("epochs_trained", None),
                "Train Time": data.get("train_time", "N/A"),
                "Inference Time": data.get("inference_time", "N/A")
            })
        except Exception as e:
            print(f"⚠️ Error loading {filepath}: {e}")

    if not records:
        print(f"⚠️ No records matched token_type='{token_type}'.")
        return None

    df = pd.DataFrame(records)
    if "BLEU" in df.columns and df["BLEU"].notnull().any():
        df = df.sort_values(by="BLEU", ascending=False)

    summary_csv = os.path.join(output_dir, f"evaluation_report_{token_type}.csv")
    summary_json = os.path.join(output_dir, f"evaluation_report_{token_type}.json")

    df.to_csv(summary_csv, index=False)
    df.to_json(summary_json, orient="records", indent=4)

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY EVALUATION REPORT ({token_type.upper()} LEVEL)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"📁 Summary report written to: {summary_csv}\n")

    return df


# Evaluation Pipeline

def _bucket_for_length(n):
    """Buckets a sentence length into a Short/Medium/Long/Very Long label."""
    if n <= 10:
        return "Short (1-10 tokens)"
    elif n <= 20:
        return "Medium (11-20 tokens)"
    elif n <= 30:
        return "Long (21-30 tokens)"
    return "Very Long (31+ tokens)"


def _config_json_path_for_checkpoint(checkpoint_path):
    """Returns the config JSON path matching a given model checkpoint path."""
    base = os.path.basename(checkpoint_path).replace("best_model_", "best_config_").replace(".pt", ".json")
    return os.path.join(os.path.dirname(checkpoint_path), base)


def evaluate_checkpoint(checkpoint_path, max_samples=1000, device=None):
    """Computes BLEU/METEOR on the held-out test set and buckets results by sentence length."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, src_vocab, trg_vocab, cfg = build_model_from_checkpoint(checkpoint, device)

    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    src_lang = cfg.get("src_lang", "de")
    trg_lang = cfg.get("trg_lang", "en")
    token_type = cfg.get("token_type", "word")

    test_csv = os.path.join(processed_dir, f"test_{src_lang}_{trg_lang}.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(processed_dir, "test.csv")

    global_cfg = load_config()
    eval_batch_size = global_cfg.get("training", {}).get("eval_batch_size", 32)

    test_loader, _, _ = get_dataloader(
        test_csv, batch_size=eval_batch_size, shuffle=False,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        src_lang=src_lang, trg_lang=trg_lang, token_type=token_type
    )

    targets = []
    hypotheses = []
    meteor_scores = []
    src_lengths = []

    count = 0
    smoother = SmoothingFunction().method1

    for src_batch, trg_batch in test_loader:
        if count >= max_samples:
            break

        for i in range(src_batch.size(0)):
            if count >= max_samples:
                break

            src_idxs = [idx.item() for idx in src_batch[i] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
            trg_idxs = [idx.item() for idx in trg_batch[i] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]

            src_tokens = idx_to_tokens(src_idxs, src_vocab)
            trg_tokens = idx_to_tokens(trg_idxs, trg_vocab)

            pred_tokens, _ = translate_sentence(model, src_tokens, src_vocab, trg_vocab, device)

            hypotheses.append(pred_tokens)
            targets.append([trg_tokens])
            src_lengths.append(len(src_idxs))

            if meteor_score is not None:
                try:
                    ref_str = " ".join(trg_tokens)
                    hyp_str = " ".join(pred_tokens)
                    meteor_scores.append(meteor_score([ref_str.split()], hyp_str.split()))
                except Exception:
                    meteor_scores.append(0.0)
            else:
                meteor_scores.append(0.0)

            count += 1

    bleu = corpus_bleu(targets, hypotheses, smoothing_function=smoother) * 100.0
    mean_meteor = (sum(meteor_scores) / len(meteor_scores) * 100.0) if meteor_scores else 0.0

    print(f"BLEU: {bleu:.4f}")
    print(f"METEOR: {mean_meteor:.4f}")

    # Bucket by source sentence length to answer "does length impact performance".
    buckets = {}
    for idx, length in enumerate(src_lengths):
        key = _bucket_for_length(length)
        buckets.setdefault(key, {"refs": [], "hyps": [], "meteors": []})
        buckets[key]["refs"].append(targets[idx])
        buckets[key]["hyps"].append(hypotheses[idx])
        buckets[key]["meteors"].append(meteor_scores[idx])

    bucket_order = ["Short (1-10 tokens)", "Medium (11-20 tokens)", "Long (21-30 tokens)", "Very Long (31+ tokens)"]
    bucket_analysis = {}
    for key in bucket_order:
        data = buckets.get(key)
        if not data or not data["hyps"]:
            bucket_analysis[key] = {"sample_count": 0, "bleu": 0.0, "meteor": 0.0}
            continue
        b_bleu = corpus_bleu(data["refs"], data["hyps"], smoothing_function=smoother) * 100.0
        b_meteor = (sum(data["meteors"]) / len(data["meteors"])) * 100.0
        bucket_analysis[key] = {
            "sample_count": len(data["hyps"]),
            "bleu": round(b_bleu, 2),
            "meteor": round(b_meteor, 2),
        }
        n = bucket_analysis[key]["sample_count"]
        bb = bucket_analysis[key]["bleu"]
        bm = bucket_analysis[key]["meteor"]
        print(f"  [{key}] n={n} BLEU={bb:.2f} METEOR={bm:.2f}")

    try:
        config_json_path = _config_json_path_for_checkpoint(checkpoint_path)
        c_data = {}
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
        c_data["bucket_analysis"] = bucket_analysis
        c_data["eval_split"] = "test"
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(c_data, f, indent=4)
    except Exception as e:
        print(f"Warning: could not persist bucket_analysis to config JSON: {e}")

    return bleu, mean_meteor, bucket_analysis


# CLI Entry Point

def evaluate_main():
    """CLI entry point: evaluate a checkpoint, visualize attention, or build a report."""
    parser = argparse.ArgumentParser(description="Evaluation and Reporting Interface")
    parser.add_argument("mode", choices=["evaluate", "report", "visualize"], nargs="?", default="report")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max test samples for BLEU evaluation")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--sentence", type=str, default=None, help="Sample sentence for attention visualization")
    args = parser.parse_args()

    if args.mode == "evaluate":
        if not args.checkpoint:
            print("❌ --checkpoint is required for 'evaluate' mode.")
            sys.exit(1)
        evaluate_checkpoint(args.checkpoint, max_samples=args.max_samples)

    elif args.mode == "visualize":
        if not args.checkpoint:
            print("❌ --checkpoint is required for 'visualize' mode.")
            sys.exit(1)
        visualize_attention(args.checkpoint, src_sentence=args.sentence)

    elif args.mode == "report":
        generate_all_reports(token_type=args.token_type)



# DE->EN->SV zero-shot pivot chain (pivot.py)

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None


class PivotTranslator:
    """Chains a DE->EN model and an EN->SV model to translate DE->SV without direct training data."""
    def __init__(self, de_en_path, en_sv_path, device, token_type="word"):
        """Loads both leg models from their checkpoints."""
        self.device = device
        self.token_type = token_type
        print("Loading DE -> EN Model Layout...")
        de_en_checkpoint = torch.load(de_en_path, map_location=device, weights_only=False)
        self.de_en_model, self.de_en_src_vocab, self.de_en_trg_vocab = self._reconstruct_model(de_en_checkpoint)
        del de_en_checkpoint

        print("Loading EN -> SV Model Layout...")
        en_sv_checkpoint = torch.load(en_sv_path, map_location=device, weights_only=False)
        self.en_sv_model, self.en_sv_src_vocab, self.en_sv_trg_vocab = self._reconstruct_model(en_sv_checkpoint)
        del en_sv_checkpoint

    def _reconstruct_model(self, checkpoint):
        """Rebuilds a Seq2Seq model from a checkpoint's config and weights."""
        config = checkpoint['config']
        src_vocab = checkpoint['src_vocab']
        trg_vocab = checkpoint['trg_vocab']

        is_bidi = config.get('bidirectional', True)
        enc_hidden_dim = config['hidden_dim'] * (2 if is_bidi else 1)

        enc = Encoder(
            len(src_vocab),
            config['emb_dim'],
            config['hidden_dim'],
            config.get('n_layers', 2),
            config['dropout'],
            rnn_type=config['rnn_type'],
            bidirectional=is_bidi,
        )
        dec = Decoder(
            len(trg_vocab),
            config['emb_dim'],
            enc_hidden_dim,
            config['hidden_dim'],
            config.get('n_layers', 2),
            config['dropout'],
            rnn_type=config['rnn_type'],
            attention_type=config.get('attention_type', 'none'),
        )

        model = Seq2Seq(enc, dec, self.device).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, src_vocab, trg_vocab

    def _tokenize(self, text):
        """Splits text into characters or words, depending on token_type."""
        text = str(text).strip()
        return list(text) if self.token_type == "char" else text.split()

    def _join(self, tokens):
        """Joins tokens back into a string."""
        return "".join(tokens) if self.token_type == "char" else " ".join(tokens)

    def translate(self, de_sentence):
        """Translates DE -> SV via the intermediate EN hop."""
        src_tokens = self._tokenize(de_sentence)

        with torch.no_grad():
            en_tokens, _ = translate_sentence(
                self.de_en_model, src_tokens, self.de_en_src_vocab, self.de_en_trg_vocab, self.device
            )
            en_sentence = self._join(en_tokens)

            # re-tokenize against the EN-SV model's own vocab, not the DE-EN model's
            en_tokens_for_sv = self._tokenize(en_sentence)
            sv_tokens, _ = translate_sentence(
                self.en_sv_model, en_tokens_for_sv, self.en_sv_src_vocab, self.en_sv_trg_vocab, self.device
            )
            sv_sentence = self._join(sv_tokens)

        return sv_sentence, en_sentence, en_tokens, sv_tokens

    def translate_with_attention(self, de_sentence):
        """Same as translate(), but also returns both legs' attention matrices for visualization."""
        src_tokens = self._tokenize(de_sentence)

        with torch.no_grad():
            en_tokens, attn_de_en = translate_sentence(
                self.de_en_model, src_tokens, self.de_en_src_vocab, self.de_en_trg_vocab, self.device
            )
            en_sentence = self._join(en_tokens)

            en_tokens_for_sv = self._tokenize(en_sentence)
            sv_tokens, attn_en_sv = translate_sentence(
                self.en_sv_model, en_tokens_for_sv, self.en_sv_src_vocab, self.en_sv_trg_vocab, self.device
            )
            sv_sentence = self._join(sv_tokens)

        return {
            "de_sentence": de_sentence,
            "en_sentence": en_sentence,
            "sv_sentence": sv_sentence,
            "leg1": {
                "src_tokens": ["<sos>"] + src_tokens + ["<eos>"],
                "trg_tokens": en_tokens,
                "attn": attn_de_en,
                "title": "DE -> EN",
            },
            "leg2": {
                "src_tokens": ["<sos>"] + en_tokens_for_sv + ["<eos>"],
                "trg_tokens": sv_tokens,
                "attn": attn_en_sv,
                "title": "EN -> SV",
            },
        }


def run_quantitative_evaluation(translator, token_type, experiment, max_samples=None):
    """Runs the DE->EN->SV pivot chain over the eval set and reports BLEU/METEOR for both hops."""
    import pandas as pd

    eval_csv = os.path.join(ROOT_DIR, "data", "processed", "pivot_de_en_sv_eval.csv")
    if not os.path.exists(eval_csv):
        print(f"\n[ERROR] Pivot evaluation set not found at {eval_csv}.")
        print("Run 'python src/build_pivot_eval_set.py' first to build it from the raw corpora.")
        return

    df = pd.read_csv(eval_csv)
    if max_samples is not None and max_samples < len(df):
        df = df.iloc[:max_samples]

    print(f"\n📊 Running quantitative pivot evaluation on {len(df):,} DE->SV pairs (experiment: {experiment})...")

    smoother = SmoothingFunction().method1
    sv_refs, sv_hyps = [], []
    en_refs, en_hyps = [], []

    for i, row in enumerate(df.itertuples(index=False)):
        de_sentence, en_reference, sv_reference = row.de, row.en, row.sv
        sv_output, en_output, _, _ = translator.translate(de_sentence)

        sv_hyps.append(sv_output.split() if token_type != "char" else list(sv_output))
        sv_refs.append([sv_reference.split() if token_type != "char" else list(sv_reference)])
        en_hyps.append(en_output.split() if token_type != "char" else list(en_output))
        en_refs.append([en_reference.split() if token_type != "char" else list(en_reference)])

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(df)} translated")

    sv_bleu = corpus_bleu(sv_refs, sv_hyps, smoothing_function=smoother) * 100.0
    en_bleu = corpus_bleu(en_refs, en_hyps, smoothing_function=smoother) * 100.0

    sv_meteor = 0.0
    en_meteor = 0.0
    if meteor_score is not None:
        try:
            sv_meteors = [meteor_score([r[0]], h) for r, h in zip(sv_refs, sv_hyps)]
            en_meteors = [meteor_score([r[0]], h) for r, h in zip(en_refs, en_hyps)]
            sv_meteor = (sum(sv_meteors) / len(sv_meteors)) * 100.0 if sv_meteors else 0.0
            en_meteor = (sum(en_meteors) / len(en_meteors)) * 100.0 if en_meteors else 0.0
        except Exception as e:
            print(f"Warning: METEOR computation failed: {e}")

    print(f"\n✅ Pivot evaluation complete ({len(df):,} sentences):")
    print(f"  DE -> EN (intermediate stage): BLEU={en_bleu:.2f} METEOR={en_meteor:.2f}")
    print(f"  DE -> EN -> SV (final output): BLEU={sv_bleu:.2f} METEOR={sv_meteor:.2f}")

    results = {
        "experiment": experiment,
        "token_type": token_type,
        "attention_type": "pivot",
        "rnn_type": "PIVOT",
        "embedding_source": "n/a",
        "n_samples": len(df),
        "intermediate_en_bleu": round(en_bleu, 2),
        "intermediate_en_meteor": round(en_meteor, 2),
        "bleu": round(sv_bleu, 2),
        "meteor": round(sv_meteor, 2),
        "eval_split": "pivot_aligned_eval_set",
    }

    output_dir = os.path.join(ROOT_DIR, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"best_config_{experiment}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved -> {out_path}")


def pivot_main():
    """CLI entry point: translate a sentence or run quantitative pivot evaluation."""
    parser = argparse.ArgumentParser(description="Zero-Shot German to Swedish via English Pivot")
    parser.add_argument("--de_en_model", type=str, required=True)
    parser.add_argument("--en_sv_model", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Text sentence to translate")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap the number of pivot evaluation pairs")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--experiment", type=str, default="PIVOT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = PivotTranslator(args.de_en_model, args.en_sv_model, device, token_type=args.token_type)

    if args.text:
        sv_output, intermediate_en, _, _ = translator.translate(args.text)
        print(f"\nOrigin (DE):      {args.text}")
        print(f"Pivot (EN):       {intermediate_en}")
        print(f"Output (SV):      {sv_output}")

    if args.evaluate:
        run_quantitative_evaluation(translator, args.token_type, args.experiment, max_samples=args.max_samples)

# Builds the aligned DE/EN/SV pivot evaluation set (build_pivot_eval_set.py)

PIVOT_EVAL_SET_SIZE = 3000
PIVOT_EVAL_SEED = 42
PIVOT_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PIVOT_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


def read_lines(path):
    """Reads a text file into a list of stripped lines."""
    with codecs.open(path, "r", encoding="utf-8", errors="replace") as f:
        return [l.strip() for l in f]


def build_pivot_eval_main():
    """CLI entry point: builds the aligned DE/EN/SV pivot evaluation CSV from raw corpora."""
    print("=" * 75)
    print("Building genuine DE -> SV pivot evaluation set (via shared English side)")
    print("=" * 75)

    de_path = os.path.join(PIVOT_RAW_DIR, "europarl-v7.de-en.de")
    en1_path = os.path.join(PIVOT_RAW_DIR, "europarl-v7.de-en.en")
    en2_path = os.path.join(PIVOT_RAW_DIR, "europarl-v7.sv-en.en")
    sv_path = os.path.join(PIVOT_RAW_DIR, "europarl-v7.sv-en.sv")

    for p in (de_path, en1_path, en2_path, sv_path):
        if not os.path.exists(p):
            print(f"ERROR: required raw file missing: {p}")
            sys.exit(1)

    print("Reading raw corpora...")
    de_lines = read_lines(de_path)
    en1_lines = read_lines(en1_path)
    en2_lines = read_lines(en2_path)
    sv_lines = read_lines(sv_path)
    print(f"  DE-EN: {len(de_lines):,} lines | EN-SV: {len(en2_lines):,} lines")

    # clean both pairs the same way training data is cleaned
    de_en_df = pd.DataFrame({"de": de_lines, "en": en1_lines})
    de_en_clean = preprocess_data(de_en_df, src_col="de", trg_col="en", token_type="word")

    en_sv_df = pd.DataFrame({"en": en2_lines, "sv": sv_lines})
    en_sv_clean = preprocess_data(en_sv_df, src_col="en", trg_col="sv", token_type="word")

    print(f"  After cleaning: DE-EN {len(de_en_clean):,} rows | EN-SV {len(en_sv_clean):,} rows")

    # dedupe by English text to build the join key
    de_en_map = dict(zip(de_en_clean["en"], de_en_clean["de"]))
    en_sv_map = dict(zip(en_sv_clean["en"], en_sv_clean["sv"]))

    shared_en = list(set(de_en_map.keys()) & set(en_sv_map.keys()))
    print(f"  Shared English sentences (alignment anchors): {len(shared_en):,}")

    random.seed(PIVOT_EVAL_SEED)
    random.shuffle(shared_en)
    selected_en = shared_en[:PIVOT_EVAL_SET_SIZE]

    rows = [{"de": de_en_map[en], "en": en, "sv": en_sv_map[en]} for en in selected_en]
    out_df = pd.DataFrame(rows)

    out_path = os.path.join(PIVOT_PROCESSED_DIR, "pivot_de_en_sv_eval.csv")
    os.makedirs(PIVOT_PROCESSED_DIR, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(out_df):,} genuine DE->SV pivot evaluation triples -> {out_path}")
    print("\nSample rows:")
    print(out_df.head(3).to_string())
    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)



# Single-experiment training entry point, DDP-aware (train.py)


class DistributedBatchSamplerWrapper(Sampler):
    """Splits a BucketBatchSampler's batches across DDP ranks, one shard of batches per process."""
    def __init__(self, batch_sampler, num_replicas, rank, shuffle=True):
        """Stores the underlying sampler and this process's rank/world size."""
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        """Sets the epoch used to seed shuffling (must match across all ranks)."""
        self.epoch = epoch
        if hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        """Yields this rank's shard of batches (seed matched across ranks so shuffling agrees)."""
        rng = random.Random(self.epoch + 42)
        batches = list(self.batch_sampler)
        if self.shuffle:
            rng.shuffle(batches)

        if len(batches) % self.num_replicas != 0:
            padding_size = self.num_replicas - (len(batches) % self.num_replicas)
            batches += batches[:padding_size]

        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]

    def __len__(self):
        """Returns this rank's share of the total batch count."""
        import math
        return math.ceil(len(self.batch_sampler) / self.num_replicas)

def str2bool(v):
    """Parses common truthy/falsy strings into a bool (for argparse)."""
    if isinstance(v, bool): return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def parse_args():
    """Defines and parses the training script's CLI arguments."""
    parser = argparse.ArgumentParser(description="Unified Seq2Seq NMT Training Interface")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--bidirectional", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--tf_ratio", type=float, default=0.5)
    parser.add_argument("--attention_type", type=str, default="none", choices=["none", "luong", "bahdanau"])
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--embedding_source", type=str, default="scratch", choices=["scratch", "word2vec", "glove"])
    parser.add_argument("--freeze_emb", type=str2bool, default=False)
    parser.add_argument("--src_lang", type=str, default="de")
    parser.add_argument("--trg_lang", type=str, default="en")
    parser.add_argument("--resume", type=str2bool, default=True, help="Resume from existing checkpoint if present")

    parser.add_argument("--eval_max_samples", type=int, default=1000,
                        help="Max samples for backfill test evaluation script (default: 1000)")
    parser.add_argument("--val_max_samples", type=int, default=None,
                        help="Max samples for per-epoch validation split (default: None for full val)")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, clip, device, tf_ratio, scaler=None, grad_accum_steps=1):
    """Trains one epoch, with mixed precision and gradient accumulation. Returns the mean loss."""
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
        
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                output_dim = output.shape[-1]
                
                if output.shape[1] == trg.shape[1]:
                    output = output[:, :-1].reshape(-1, output_dim)
                else:
                    output = output.reshape(-1, output_dim)
                    
                trg_eval = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_eval) / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            output_dim = output.shape[-1]
            
            if output.shape[1] == trg.shape[1]:
                output = output[:, :-1].reshape(-1, output_dim)
            else:
                output = output.reshape(-1, output_dim)
                
            trg_eval = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_eval) / grad_accum_steps
            
            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
        epoch_loss += loss.item() * grad_accum_steps
    
    total_loss = epoch_loss / len(dataloader)
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.item() / dist.get_world_size()
    
    return total_loss

def evaluate_validation(model, dataloader, criterion, device):
    """Computes mean validation loss (no teacher forcing, no gradient updates)."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.amp.autocast(device_type=device.type):
                    output = model(src, trg, teacher_forcing_ratio=0.0)
                    output_dim = output.shape[-1]
                    
                    if output.shape[1] == trg.shape[1]:
                        output = output[:, :-1].reshape(-1, output_dim)
                    else:
                        output = output.reshape(-1, output_dim)
                        
                    trg_eval = trg[:, 1:].reshape(-1)
                    loss = criterion(output, trg_eval)
            else:
                output = model(src, trg, teacher_forcing_ratio=0.0)
                output_dim = output.shape[-1]
                
                if output.shape[1] == trg.shape[1]:
                    output = output[:, :-1].reshape(-1, output_dim)
                else:
                    output = output.reshape(-1, output_dim)
                    
                trg_eval = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_eval)
            epoch_loss += loss.item()
            
    total_loss = epoch_loss / len(dataloader)
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.item() / dist.get_world_size()
        
    return total_loss

def train_main():
    """CLI entry point: trains one experiment end-to-end (DDP-aware) and saves the best checkpoint."""
    args = parse_args()


    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        
    set_seed(42 + rank)
    
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cfg_data = load_config()
    processed_dir = cfg_data.get("data", {}).get("processed_dir", "data/processed")
    
    train_csv = os.path.join(processed_dir, f"train_{args.src_lang}_{args.trg_lang}.csv")
    val_csv = os.path.join(processed_dir, f"val_{args.src_lang}_{args.trg_lang}.csv")

    if not os.path.exists(train_csv):
        legacy_suffix = "_sv" if (args.src_lang == "sv" or args.trg_lang == "sv") else ""
        legacy_train = os.path.join(processed_dir, f"train{legacy_suffix}.csv")
        legacy_val = os.path.join(processed_dir, f"val{legacy_suffix}.csv")
        
        if os.path.exists(legacy_train):
            train_csv, val_csv = legacy_train, legacy_val
        else:
            train_csv = os.path.join(processed_dir, "train.csv")
            val_csv = os.path.join(processed_dir, "val.csv")

    if rank == 0:
        print(f"📁 Resolving train split: {train_csv}")
        print(f"📁 Resolving val split:   {val_csv}")

    raw_train_loader, src_vocab, trg_vocab = get_dataloader(
        train_csv, batch_size=args.batch_size, shuffle=True, 
        src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type
    )
    raw_val_loader, _, _ = get_dataloader(
        val_csv, batch_size=args.batch_size, shuffle=False, 
        src_vocab=src_vocab, trg_vocab=trg_vocab, 
        src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type
    )
    
    # Validation Subsampling Logic (Optional)
    if args.val_max_samples and args.val_max_samples < len(raw_val_loader.dataset):
        if rank == 0:
            print(f"⚡ Subsampling validation set: randomly sampling {args.val_max_samples}/{len(raw_val_loader.dataset)} items.")
        random.seed(42)
        val_indices = random.sample(range(len(raw_val_loader.dataset)), args.val_max_samples)
        raw_val_loader = DataLoader(
            Subset(raw_val_loader.dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=raw_val_loader.collate_fn,
            num_workers=raw_val_loader.num_workers,
            pin_memory=raw_val_loader.pin_memory,
            persistent_workers=(raw_val_loader.num_workers > 0)
        )

    if is_distributed:
        train_sampler = DistributedBatchSamplerWrapper(
            raw_train_loader.batch_sampler, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedBatchSamplerWrapper(
            raw_val_loader.batch_sampler, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        train_loader = DataLoader(
            raw_train_loader.dataset,
            batch_sampler=train_sampler,
            collate_fn=raw_train_loader.collate_fn,
            num_workers=raw_train_loader.num_workers,
            pin_memory=raw_train_loader.pin_memory,
            persistent_workers=(raw_train_loader.num_workers > 0)
        )
        val_loader = DataLoader(
            raw_val_loader.dataset,
            batch_sampler=val_sampler,
            collate_fn=raw_val_loader.collate_fn,
            num_workers=raw_val_loader.num_workers,
            pin_memory=raw_val_loader.pin_memory,
            persistent_workers=(raw_val_loader.num_workers > 0)
        )
    else:
        train_loader = raw_train_loader
        val_loader = raw_val_loader
        train_sampler = None
        val_sampler = None
    
    pretrained_src_emb, pretrained_trg_emb = None, None
    silent_logging = rank > 0
    
    # source and target vocab each resolve to their own language's embedding file
    data_dir = os.path.join(ROOT_DIR, "data")
    if args.embedding_source == "word2vec":
        pretrained_src_emb = generate_word2vec_embeddings(
            src_vocab, train_csv, lang=args.src_lang, emb_dim=args.emb_dim,
            silent=silent_logging, token_type=args.token_type, data_dir=data_dir,
        )
        pretrained_trg_emb = generate_word2vec_embeddings(
            trg_vocab, train_csv, lang=args.trg_lang, emb_dim=args.emb_dim,
            silent=silent_logging, token_type=args.token_type, data_dir=data_dir,
        )
    elif args.embedding_source == "glove":
        pretrained_src_emb = load_glove_embeddings(
            src_vocab, emb_dim=300, silent=silent_logging,
            token_type=args.token_type, glove_dir=data_dir, lang=args.src_lang,
        )
        pretrained_trg_emb = load_glove_embeddings(
            trg_vocab, emb_dim=300, silent=silent_logging,
            token_type=args.token_type, glove_dir=data_dir, lang=args.trg_lang,
        )
        
    num_directions = 2 if args.bidirectional else 1
    encoder = Encoder(
        len(src_vocab), args.emb_dim, args.hidden_dim, 2, args.dropout, 
        args.rnn_type, args.bidirectional, pretrained_src_emb, args.freeze_emb, 
        300 if args.embedding_source == "glove" else None
    )
    decoder = Decoder(
        len(trg_vocab), args.emb_dim, args.hidden_dim * num_directions, args.hidden_dim, 2, 
        args.dropout, args.rnn_type, args.attention_type, pretrained_trg_emb, args.freeze_emb, 
        300 if args.embedding_source == "glove" else None
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = (total_params * 4) / (1024 ** 2)
        
        sample_src, sample_trg = next(iter(train_loader))
        
        src_bytes = sample_src.element_size() * sample_src.nelement()
        trg_bytes = sample_trg.element_size() * sample_trg.nelement()
        total_batch_bytes = src_bytes + trg_bytes
        
        src_mb = src_bytes / (1024 ** 2)
        trg_mb = trg_bytes / (1024 ** 2)
        total_batch_mb = total_batch_bytes / (1024 ** 2)

        char_index_bytes = 8
        char_emb_bytes = args.emb_dim * 4
        
        global_batch_size = args.batch_size * world_size

        print("\n" + "─" * 75)
        print(f"📐 [DYNAMIC MODEL & BATCH ANALYSIS]")
        print(f" ├─ Experiment ID:              {args.experiment}")
        print(f" ├─ Tokenizer Mode:             {args.token_type.upper()}")
        print(f" ├─ Micro-Batch Size (p/GPU):   {args.batch_size}")
        print(f" ├─ Grad Accumulation Steps:    {args.grad_accum_steps}")
        print(f" ├─ Global Batch Size (Total):   {global_batch_size * args.grad_accum_steps} sequence(s) across {world_size} rank(s)")
        print(f" ├─ Single Char/Token ID Size:  {char_index_bytes} bytes (int64 tensor index)")
        print(f" ├─ Single Char Embedding Size: {char_emb_bytes} bytes (Emb={args.emb_dim} float32 vector)")
        print(f" ├─ Dynamic 'src' Tensor Shape: {list(sample_src.shape)} -> {src_mb:.6f} MB")
        print(f" ├─ Dynamic 'trg' Tensor Shape: {list(sample_trg.shape)} -> {trg_mb:.6f} MB")
        print(f" ├─ Total Batch Pair Footprint: {total_batch_mb:.6f} MB ({total_batch_bytes:,} bytes)")
        print(f" ├─ Total Trainable Parameters: {total_params:,}")
        print(f" └─ Total Model Memory (FP32):  {model_size_mb:.2f} MB")
        print("─" * 75 + "\n")
    
    if is_distributed:
        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        
    # torch.compile needs CUDA compute capability >= 7.0 (fails late otherwise on older GPUs)
    supports_compile = hasattr(torch, "compile") and (
        device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
    )
    if supports_compile:
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            if rank == 0:
                print(f"⚠️ torch.compile skipped or failed: {e}")
    elif rank == 0 and hasattr(torch, "compile") and device.type == "cuda":
        print("ℹ️ torch.compile skipped: GPU compute capability < 7.0 (no Triton support).")
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    best_val_loss = float("inf")
    start_train_time = time.time()
    loss_history = {"train": [], "val": []}
    
    exp_tag = args.experiment if f"_{args.rnn_type}" in args.experiment else f"{args.experiment}_{args.rnn_type}"
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_tag}.pt")
    config_json_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_tag}.json")
    start_epoch = 0

    if args.resume and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"🔄 Resuming model weights from existing checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        
        # strip torch.compile/DDP wrapper prefixes from the state dict keys
        clean_state_dict = {
            k.replace("_orig_mod.", "").replace("module.", ""): v 
            for k, v in checkpoint['model_state_dict'].items()
        }
        raw_model.load_state_dict(clean_state_dict)
        
        if 'best_val_loss' in checkpoint.get('config', {}):
            best_val_loss = checkpoint['config']['best_val_loss']
        if 'loss_history' in checkpoint and isinstance(checkpoint['loss_history'], dict):
            loss_history = checkpoint['loss_history']
            start_epoch = len(loss_history.get("train", []))
    
    if start_epoch >= args.epochs:
        if rank == 0:
            print(f"📦 Checkpoint already fully trained ({start_epoch}/{args.epochs} epochs). Skipping epoch loop.")
            # mark completed in case a prior run crashed before setting it
            if os.path.exists(config_json_path):
                try:
                    with open(config_json_path, 'r') as f:
                        c_data = json.load(f)
                    if not c_data.get("completed"):
                        c_data["completed"] = True
                        with open(config_json_path, 'w') as f:
                            json.dump(c_data, f, indent=4)
                except Exception:
                    pass
    else:
        for epoch in range(start_epoch, args.epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)
                
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, args.clip, device, 
                args.tf_ratio, scaler, grad_accum_steps=args.grad_accum_steps
            )
            val_loss = evaluate_validation(model, val_loader, criterion, device)
            
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            
            if rank == 0:
                print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if rank == 0:
                    config_dict = vars(args).copy()
                    config_dict.update({
                        "train_time": f"{time.time() - start_train_time:.1f}s", 
                        "best_val_loss": best_val_loss, 
                        "val_loss": best_val_loss,
                        "epochs_trained": len(loss_history["train"]),
                        "loss_history": loss_history
                    })
                    
                    # Strip wrapper layers (DDP / torch.compile) before saving state dict
                    raw_model = model.module if hasattr(model, "module") else model
                    if hasattr(raw_model, "_orig_mod"):
                        raw_model = raw_model._orig_mod

                    clean_state_dict = {
                        k.replace("_orig_mod.", "").replace("module.", ""): v 
                        for k, v in raw_model.state_dict().items()
                    }

                    torch.save({
                        'config': config_dict, 
                        'model_state_dict': clean_state_dict, 
                        'src_vocab': src_vocab, 
                        'trg_vocab': trg_vocab,
                        'loss_history': loss_history
                    }, checkpoint_path)
                    
                    with open(config_json_path, 'w') as f:
                        json.dump(config_dict, f, indent=4)
            elif rank == 0:
                # no early stopping - all configs in a study train the same epoch budget
                print(f"↪️ No improvement this epoch (best remains {best_val_loss:.4f}). Continuing - training the full requested budget regardless.")

                # keep loss_history/epochs_trained current even on non-improving epochs
                if os.path.exists(config_json_path):
                    try:
                        with open(config_json_path, 'r') as f:
                            c_data = json.load(f)
                        c_data["loss_history"] = loss_history
                        c_data["epochs_trained"] = len(loss_history["train"])
                        with open(config_json_path, 'w') as f:
                            json.dump(c_data, f, indent=4)
                    except Exception:
                        pass

        # mark completed=True only once the full epoch budget has actually finished -
        # is_cache_valid() checks this to decide whether a restart can skip this experiment
        if rank == 0 and os.path.exists(config_json_path):
            try:
                with open(config_json_path, 'r') as f:
                    c_data = json.load(f)
                c_data["loss_history"] = loss_history
                c_data["epochs_trained"] = len(loss_history["train"])
                c_data["completed"] = True
                with open(config_json_path, 'w') as f:
                    json.dump(c_data, f, indent=4)
            except Exception:
                pass

    if rank == 0:
        if os.path.exists(checkpoint_path) and not args.experiment.startswith("TUNE_"):
            try:
                import subprocess
                import re
                import sys
                
                print(f"\n⌛ Automated Backfill: Executing evaluation metrics extraction...")
                cmd = [
                    sys.executable, SELF_PATH, "--task", "evaluate", "evaluate",
                    "--checkpoint", checkpoint_path,
                    "--max_samples", str(args.eval_max_samples)
                ]
                
                start_eval_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                inference_duration = time.time() - start_eval_time
                
                bleu_match = re.search(r"BLEU:\s*([\d\.]+)", result.stdout)
                meteor_match = re.search(r"METEOR:\s*([\d\.]+)", result.stdout)
                
                bleu_score = float(bleu_match.group(1)) if bleu_match else None
                meteor_score = float(meteor_match.group(1)) if meteor_match else None
                
                if bleu_score is not None:
                    if os.path.exists(config_json_path):
                        with open(config_json_path, 'r') as f:
                            c_data = json.load(f)
                        
                        c_data["bleu"] = bleu_score
                        c_data["Target Metric (BLEU)"] = bleu_score
                        c_data["bleu_score"] = bleu_score
                        c_data["overall_corpus_bleu"] = bleu_score
                        
                        if meteor_score is not None:
                            c_data["meteor"] = meteor_score
                            c_data["mean_meteor"] = meteor_score
                            
                        # Save inference duration into JSON metadata ledger
                        c_data["inference_time"] = f"{inference_duration:.2f}s"
                            
                        with open(config_json_path, 'w') as f:
                            json.dump(c_data, f, indent=4)
                            
                        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        checkpoint_payload['config'].update({
                            "bleu": bleu_score,
                            "Target Metric (BLEU)": bleu_score,
                            "bleu_score": bleu_score,
                            "overall_corpus_bleu": bleu_score,
                            "inference_time": f"{inference_duration:.2f}s"
                        })
                        if meteor_score is not None:
                            checkpoint_payload['config'].update({
                                "meteor": meteor_score,
                                "mean_meteor": meteor_score
                            })
                        torch.save(checkpoint_payload, checkpoint_path)
                        print(f"✅ Backfill Successful: Saved BLEU={bleu_score} and inference_time={inference_duration:.2f}s inside local JSON ledger.")
            except Exception as e:
                print(f"⚠️ Automated metrics backfill skipped: {e}")

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()



# Master orchestrator - all 5 ablation studies + tuning stages (run_studies.py)



def get_batch_size(study, token_type):
    """Resolves batch size: per-token-type config override, then generic override, then default."""
    training_cfg = config.get("training", {})

    per_type_batch = training_cfg.get(f"batch_size_{token_type}")
    if per_type_batch is not None:
        return str(per_type_batch)

    config_batch = training_cfg.get("batch_size")
    if config_batch is not None:
        return str(config_batch)

    return "256" if token_type == "char" else "128"


class AsyncEvaluationQueue:
    """Offloads evaluation and ledger synchronization to background execution threads."""

    def __init__(self, max_workers=2):
        """Sets up the background thread pool."""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit_evaluation(self, experiment_id, rnn_type, token_type):
        """Queues evaluation in the background without holding a global lock during computation."""

        def _task():
            """Runs the evaluation and syncs its result into the ledger."""
            print(
                f"\n⚡ [Async Eval Started] -> {experiment_id} ({rnn_type})"
            )
            run_auto_evaluation(experiment_id, rnn_type)
            with eval_lock:
                sync_ledger_to_token_type(token_type)
            print(
                f"✅ [Async Eval Finished] -> {experiment_id} ({rnn_type})"
            )

        future = self.executor.submit(_task)
        self.futures.append(future)

    def sync_study(self):
        """Blocks until all queued evaluations for the current study complete."""
        if self.futures:
            print(
                "\n⏳ [Study Barrier] Waiting for background evaluation tasks"
                " to complete..."
            )
            wait(self.futures)
            self.futures.clear()
            print(
                "🎯 [Study Barrier Cleared] All study models evaluated"
                " successfully.\n"
            )

    def shutdown(self):
        """Waits for all queued evaluations to finish and stops the thread pool."""
        self.executor.shutdown(wait=True)


def get_vocab_sizes(token_type="word"):
    """Dynamically retrieves vocabulary size from binary dataset cache or defaults to baseline."""
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    cache_dir = os.path.join(processed_dir, ".matrix_cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".pt") and token_type in fname:
                try:
                    payload = torch.load(
                        os.path.join(cache_dir, fname),
                        map_location="cpu",
                        weights_only=False,
                    )
                    if isinstance(payload, dict) and "src_vocab" in payload:
                        return len(payload["src_vocab"]), len(
                            payload["trg_vocab"]
                        )
                except Exception:
                    pass
    return (8192, 8192) if token_type == "word" else (256, 256)


def print_study_model_and_batch_info(
    study_name,
    exp_id,
    token_type,
    rnn_type,
    bidirectional,
    attention_type,
    emb_dim,
    hidden_dim,
    batch_size,
):
    """Analytically computes and outputs Model Size and Batch Size parameters."""
    src_vocab_len, trg_vocab_len = get_vocab_sizes(token_type)
    bidi_bool = str(bidirectional).lower() == "true"
    num_directions = 2 if bidi_bool else 1
    emb_d, hid_d = int(emb_dim), int(hidden_dim)
    gates = 4 if rnn_type == "LSTM" else (3 if rnn_type == "GRU" else 1)

    enc_emb = src_vocab_len * emb_d
    enc_l1 = gates * (
        (emb_d * hid_d + hid_d * hid_d + 2 * hid_d) * num_directions
    )
    enc_l2 = gates * (
        (hid_d * num_directions * hid_d + hid_d * hid_d + 2 * hid_d)
        * num_directions
    )
    enc_params = enc_emb + enc_l1 + enc_l2

    dec_emb = trg_vocab_len * emb_d
    enc_out_dim = hid_d * num_directions
    dec_rnn_in = emb_d + (enc_out_dim if attention_type != "none" else 0)
    dec_l1 = gates * (dec_rnn_in * hid_d + hid_d * hid_d + 2 * hid_d)
    dec_l2 = gates * (hid_d * hid_d + hid_d * hid_d + 2 * hid_d)
    dec_fc = hid_d * trg_vocab_len + trg_vocab_len

    attn_params = 0
    if attention_type == "luong":
        attn_params = (enc_out_dim * hid_d) + hid_d
    elif attention_type == "bahdanau":
        attn_params = (
            (hid_d * hid_d + hid_d) + (enc_out_dim * hid_d + hid_d) + hid_d
        )

    total_params = enc_params + dec_emb + dec_l1 + dec_l2 + dec_fc + attn_params
    model_size_mb = (total_params * 4) / (1024**2)

    seq_len = 256 if token_type == "char" else 64
    batch_num = int(batch_size)
    batch_memory_mb = (batch_num * seq_len * 8) / (1024**2)

    print("\n" + "─" * 75)
    print(f"📐 [DYNAMIC STUDY ANALYSIS] - {study_name} ({exp_id})")
    print(f" ├─ Tokenizer Mode:           {token_type.upper()}")
    print(
        f" ├─ Architecture:             {rnn_type} (BiDirect: {bidi_bool},"
        f" Attention: {attention_type})"
    )
    print(f" ├─ Dimensions:               Emb={emb_dim} | Hidden={hidden_dim}")
    print(f" ├─ Batch Size (N samples):   {batch_num} sequences / batch")
    print(
        f" ├─ Batch Shape Estimate:     [{batch_num}, {seq_len}]"
        f" ({batch_memory_mb:.4f} MB per batch tensor)"
    )
    print(f" ├─ Total Model Parameters:   ~{total_params:,} parameters")
    print(f" └─ Model Memory Footprint:   ~{model_size_mb:.2f} MB")
    print("─" * 75 + "\n")


def run_cmd(args_list):
    """Executes distributed PyTorch training sub-process via PyTorch DDP launcher."""
    if "--grad_accum_steps" not in args_list:
        args_list = ["--grad_accum_steps", "4"] + args_list

    i = 0
    kv = {}
    positional = []
    while i < len(args_list):
        item = args_list[i]
        if item.startswith("-"):
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                kv[item] = args_list[i + 1]
                i += 2
            else:
                kv[item] = None
                i += 1
        else:
            positional.append(item)
            i += 1

    cleaned_args = []
    for k, v in kv.items():
        cleaned_args.append(k)
        if v is not None:
            cleaned_args.append(v)
    args_list = cleaned_args + positional

    epochs = config.get("training", {}).get("epochs", 1)
    if "--epochs" in args_list:
        try:
            epochs = int(args_list[args_list.index("--epochs") + 1])
        except (ValueError, IndexError):
            pass

    nproc = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        SELF_PATH,
        "--task",
        "train",
    ] + args_list

    print(
        f"\n🚀 Launching DDP Execution Unit ({nproc} processes):"
        f" {' '.join(command)}"
    )

    start_time = time.time()
    subprocess.run(command, check=True)
    duration = time.time() - start_time
    print(
        f"⏱️ Done. Duration: {duration:.2f}s | Avg/Epoch:"
        f" {duration/max(1, epochs):.2f}s"
    )


def run_auto_evaluation(experiment_id, rnn_type):
    """Executes model evaluation with GPU acceleration when available."""
    target_model = os.path.join(OUTPUT_DIR, f"best_model_{experiment_id}_{rnn_type}.pt")
    if os.path.exists(target_model):
        cmd = [
            sys.executable,
            SELF_PATH,
            "--task",
            "evaluate",
            "evaluate",
            "--checkpoint",
            target_model,
        ]
        env = os.environ.copy()
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            print(f"⚠️ Evaluation failed for {experiment_id}.")


def sync_ledger_to_token_type(token_type):
    """Parses master ledger and segregates metrics into isolated study ledger files."""
    global_ledger = os.path.join(
        ROOT_DIR, f"evaluation_ledger_{token_type}.json"
    )
    if not os.path.exists(global_ledger):
        return

    try:
        with open(global_ledger, "r", encoding="utf-8") as f:
            g_data = json.load(f)

        remaining_g_data = {}
        for k, v in g_data.items():
            if k.upper().startswith(token_type.upper()):
                match = re.search(r"_(A|B|C|D|E)\d*|_(PIVOT)", k.upper())
                study_suffix = (
                    match.group(1) or match.group(2) if match else "MISC"
                )

                study_ledger_path = os.path.join(
                    ROOT_DIR, f"evaluation_ledger_{token_type}_{study_suffix}.json"
                )

                study_data = {}
                if os.path.exists(study_ledger_path):
                    try:
                        with open(
                            study_ledger_path, "r", encoding="utf-8"
                        ) as sf:
                            study_data = json.load(sf)
                    except Exception:
                        study_data = {}

                study_data[k] = v
                with open(study_ledger_path, "w", encoding="utf-8") as sf:
                    json.dump(study_data, sf, indent=4)
            else:
                remaining_g_data[k] = v

        if remaining_g_data:
            with open(global_ledger, "w", encoding="utf-8") as f:
                json.dump(remaining_g_data, f, indent=4)
        else:
            if os.path.exists(global_ledger):
                os.remove(global_ledger)
    except Exception as e:
        print(f"⚠️ Error occurred breaking ledger down to study files: {e}")


def get_best_hyperparameters(stage, token_type, rnn_type=None):
    """Parses validation loss metrics from hyperparameter tuning sweeps."""
    csv_path = os.path.join(
        ROOT_DIR, f"tuning_results_{token_type}_{stage}.csv"
    )
    profile = config.get("profiles", {}).get(token_type, {})
    default_args = [
        "--lr",
        str(profile.get("lr", 0.001)),
        "--dropout",
        str(profile.get("dropout", 0.3)),
        "--emb_dim",
        str(profile.get("emb_dim", 256)),
        "--hidden_dim",
        str(profile.get("hidden_dim", 512)),
    ]

    if not os.path.exists(csv_path):
        tune_configs = glob.glob(
            os.path.join(
                OUTPUT_DIR, f"best_config_TUNE_{token_type.upper()}_*.json"
            )
        )
        if tune_configs:
            best_loss = float("inf")
            best_cfg = None
            for cfg_f in tune_configs:
                try:
                    with open(cfg_f, "r", encoding="utf-8") as f:
                        c = json.load(f)
                    if (
                        rnn_type
                        and c.get("rnn_type", "").upper() != rnn_type.upper()
                    ):
                        continue
                    v_loss = float(
                        c.get("best_val_loss", c.get("val_loss", 999.0))
                    )
                    if v_loss < best_loss:
                        best_loss = v_loss
                        best_cfg = c
                except Exception:
                    pass
            if best_cfg:
                lr = best_cfg.get("lr", profile.get("lr", 0.001))
                dropout = best_cfg.get("dropout", profile.get("dropout", 0.3))
                emb_dim = int(
                    best_cfg.get("emb_dim", profile.get("emb_dim", 256))
                )
                hidden_dim = int(
                    best_cfg.get("hidden_dim", profile.get("hidden_dim", 512))
                )
                print(
                    "🎯 Optimization Checkpoint Found! Applying Tuned"
                    f" Parameters: --lr {lr} --dropout {dropout} --emb_dim"
                    f" {emb_dim} --hidden_dim {hidden_dim}"
                )
                return [
                    "--lr",
                    str(lr),
                    "--dropout",
                    str(dropout),
                    "--emb_dim",
                    str(emb_dim),
                    "--hidden_dim",
                    str(hidden_dim),
                ]

        print(
            f"ℹ️ Tuning ledger missing at {csv_path}. Falling back to standard"
            " profiles."
        )
        return default_args

    try:
        df = pd.read_csv(csv_path)
        valid = df[df["status"].astype(str).str.strip() == "Success"].copy()
        if valid.empty:
            print(
                f"⚠️ Tuning file {csv_path} contains no successful sweeps."
                " Falling back to defaults."
            )
            return default_args

        if rnn_type and "rnn_type" in valid.columns:
            cell_runs = valid[
                valid["rnn_type"].astype(str).str.upper().str.strip()
                == rnn_type.upper().strip()
            ]
            if not cell_runs.empty:
                valid = cell_runs

        valid["val_loss"] = pd.to_numeric(valid["val_loss"], errors="coerce")
        valid = valid.dropna(subset=["val_loss"])

        if valid.empty:
            print(
                "⚠️ No valid numerical optimization scores for"
                f" {rnn_type or 'all'}. Falling back to defaults."
            )
            return default_args

        valid = valid.sort_values(by="val_loss", ascending=True)
        best_run = valid.iloc[0]

        lr = best_run.get("learning_rate", profile.get("lr", 0.001))
        dropout = best_run.get("dropout", profile.get("dropout", 0.3))
        emb_dim = int(
            float(best_run.get("emb_dim", profile.get("emb_dim", 256)))
        )
        hidden_dim = int(
            float(best_run.get("hidden_dim", profile.get("hidden_dim", 512)))
        )

        print(
            "🎯 Optimization Checkpoint Found! Applying Tuned Parameters: --lr"
            f" {lr} --dropout {dropout} --emb_dim {emb_dim} --hidden_dim"
            f" {hidden_dim}"
        )
        return [
            "--lr",
            str(lr),
            "--dropout",
            str(dropout),
            "--emb_dim",
            str(emb_dim),
            "--hidden_dim",
            str(hidden_dim),
        ]

    except Exception as e:
        print(
            "⚠️ Hyperparameter parser interception exception:"
            f" {e}. Default parameter profile applied."
        )
        return default_args


def get_best_empirical_settings(token_type):
    """Retrieves top-performing empirical architectural settings from past study runs."""
    profile = config.get("profiles", {}).get(token_type, {})
    defaults = {
        "rnn_type": "LSTM",
        "bidirectional": "True",
        "embedding_source": "scratch",
        "freeze_emb": "False",
        "attention_type": "none",
        "emb_dim": str(profile.get("emb_dim", 256)),
    }

    ledger = {}
    pattern = os.path.join(ROOT_DIR, f"evaluation_ledger_{token_type}_*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                ledger.update(json.load(f))
        except Exception:
            pass

    pattern_cfg = os.path.join(
        OUTPUT_DIR, f"best_config_{token_type.upper()}_*.json"
    )
    for filepath in glob.glob(pattern_cfg):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            exp_key = cdata.get(
                "experiment",
                os.path.basename(filepath)
                .replace("best_config_", "")
                .split(".json")[0],
            )
            if exp_key not in ledger:
                ledger[exp_key] = cdata
        except Exception:
            pass

    if not ledger:
        return defaults
    try:
        prefix = token_type.upper()

        def get_composite_score(node):
            """Scores an experiment by BLEU+METEOR if available, else negative val loss."""
            metrics = node.get("metrics", {})
            bleu = float(
                metrics.get("overall_corpus_bleu", node.get("bleu", 0.0))
            )
            meteor = float(metrics.get("mean_meteor", node.get("meteor", 0.0)))
            if bleu > 0 or meteor > 0:
                return bleu + (meteor * 100.0)
            val_loss = float(
                node.get("best_val_loss", node.get("val_loss", 999.0))
            )
            return -val_loss

        best_a = -float("inf")
        for exp in [f"{prefix}_A{i}" for i in range(1, 7)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_a:
                    best_a = score
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])

        best_b = -float("inf")
        for exp in [f"{prefix}_B{i}" for i in range(1, 13)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_b:
                    best_b = score
                    defaults["embedding_source"] = ledger[k].get(
                        "embedding_source", defaults["embedding_source"]
                    )
                    defaults["freeze_emb"] = str(
                        ledger[k].get("freeze_emb", defaults["freeze_emb"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )

        best_c = -float("inf")
        for exp in [f"{prefix}_C{i}" for i in range(1, 7)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_c:
                    best_c = score
                    defaults["attention_type"] = ledger[k].get(
                        "attention_type", defaults["attention_type"]
                    )
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])
    except Exception:
        pass
    return defaults


def execute_preprocessing(token_type="word", mock_mode=False):
    """Executes offline dataset preprocessing and binary caching routines."""
    cmd = [
        sys.executable,
        SELF_PATH,
        "--task",
        "preprocess",
        "--token_type",
        token_type,
    ]
    if mock_mode:
        cmd.append("--mock")

    print(
        "⚡ Running preprocessing routine"
        f" (token_type={token_type}, mock={mock_mode})..."
    )
    subprocess.run(cmd, check=True)


def execute_tuning(
    stage="coarse", token_type="word", epochs=5, num_trials=12, configs_per_rnn=None
):
    """Runs a hyperparameter sweep: "coarse" across RNN/GRU/LSTM, "fine" on Study C's winning architecture."""
    print(
        "\n"
        + "═" * 75
        + f"\n🔍 RUNNING HYPERPARAMETER TUNING ({stage.upper()} -"
        f" {token_type.upper()} | {epochs} Epochs)\n"
        + "═" * 75
    )

    lrs = [0.0001, 0.0003, 0.0005, 0.001]
    dropouts = [0.2, 0.3, 0.4]
    emb_dims = [128, 256, 512] if token_type == "word" else [32, 64, 128]
    hidden_dims = [256, 512, 1024]

    batch_size = get_batch_size("TUNE", token_type)
    results_csv = os.path.join(
        ROOT_DIR, f"tuning_results_{token_type}_{stage}.csv"
    )

    fieldnames = [
        "run_id",
        "stage",
        "token_type",
        "rnn_type",
        "attention_type",
        "bidirectional",
        "learning_rate",
        "dropout",
        "emb_dim",
        "hidden_dim",
        "val_loss",
        "status",
    ]

    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    random.seed(42)

    selected_trials = []

    if stage == "fine":
        winner_c = get_best_empirical_settings(token_type)
        winner_rnn = winner_c["rnn_type"]
        winner_attn = winner_c["attention_type"]
        winner_bidi = winner_c["bidirectional"]
        winner_emb_src = winner_c["embedding_source"]
        winner_freeze = winner_c["freeze_emb"]

        print("─" * 75)
        print(f"🏆 [FINE TUNE CONFIGURATION - STUDY C WINNER]")
        print(f" ├─ Winner RNN Model:      {winner_rnn}")
        print(f" ├─ Winner Attention:      {winner_attn}")
        print(f" ├─ Winner Bidirectional:  {winner_bidi}")
        print(f" ├─ Winner Embedding Src:  {winner_emb_src}")
        print(f" └─ Winner Freeze Emb:     {winner_freeze}")
        print("─" * 75)

        all_combos = list(itertools.product(lrs, dropouts, emb_dims, hidden_dims))
        random.shuffle(all_combos)

        trial_combos = all_combos[:num_trials] if num_trials and num_trials < len(all_combos) else all_combos
        for lr, drop, emb_d, hid_d in trial_combos:
            selected_trials.append((lr, drop, emb_d, hid_d, winner_rnn, winner_attn, winner_bidi, winner_emb_src, winner_freeze))

        print(f"📋 Configured {len(selected_trials)} fine-tuning trials for Study C Winner Architecture ({winner_rnn} + {winner_attn}).")

    else:
        # Coarse sweep over all baseline cell types
        rnn_types = ["LSTM", "GRU", "RNN"]
        if configs_per_rnn is not None:
            trials_per_rnn = configs_per_rnn
        else:
            trials_per_rnn = max(1, num_trials // len(rnn_types))

        def _diverse_picks(values, n):
            """Cycles through a shuffled list of values so n picks cover as many distinct values as possible."""
            order = list(values)
            random.shuffle(order)
            return [order[i % len(order)] for i in range(n)]

        selected_combos = []
        for rnn in rnn_types:
            lr_picks = _diverse_picks(lrs, trials_per_rnn)
            dropout_picks = _diverse_picks(dropouts, trials_per_rnn)
            emb_dim_picks = _diverse_picks(emb_dims, trials_per_rnn)
            hidden_dim_picks = _diverse_picks(hidden_dims, trials_per_rnn)
            for i in range(trials_per_rnn):
                selected_combos.append(
                    (lr_picks[i], dropout_picks[i], emb_dim_picks[i], hidden_dim_picks[i], rnn)
                )

        if len(selected_combos) < num_trials and configs_per_rnn is None:
            all_combos = list(itertools.product(lrs, dropouts, emb_dims, hidden_dims, rnn_types))
            remaining = [c for c in all_combos if c not in selected_combos]
            random.shuffle(remaining)
            selected_combos.extend(remaining[:num_trials - len(selected_combos)])

        for lr, drop, emb_d, hid_d, rnn in selected_combos:
            selected_trials.append((lr, drop, emb_d, hid_d, rnn, "none", "False", "scratch", "False"))

        print(f"📋 Configured {len(selected_trials)} coarse trials across cell types {rnn_types}.")

    best_loss = float("inf")
    best_params = None

    for idx, (lr, drop, emb_d, hid_d, rnn, attn, bidi, emb_src, freeze) in enumerate(selected_trials, 1):
        exp_id = f"TUNE_{token_type.upper()}_{stage.upper()}_{idx}"
        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn}.pt")
        cfg_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{rnn}.json")

        print(
            f"\n🧪 [Trial {idx}/{len(selected_trials)}] -> LR={lr}, Dropout={drop},"
            f" Emb={emb_d}, Hidden={hid_d}, Cell={rnn}, Attn={attn}, BiDir={bidi}"
        )

        cmd = [
            "--experiment",
            exp_id,
            "--rnn_type",
            rnn,
            "--attention_type",
            attn,
            "--bidirectional",
            bidi,
            "--embedding_source",
            emb_src,
            "--freeze_emb",
            freeze,
            "--token_type",
            token_type,
            "--lr",
            str(lr),
            "--dropout",
            str(drop),
            "--emb_dim",
            str(emb_d),
            "--hidden_dim",
            str(hid_d),
            "--batch_size",
            batch_size,
            "--epochs",
            str(epochs),
            "--src_lang",
            "en",
            "--trg_lang",
            "de",
        ]

        try:
            if is_cache_valid(ckpt_path, cfg_path):
                print(
                    f"📦 [Cache Hit] Trial {idx} ({exp_id}) already completed."
                    " Skipping training."
                )
            else:
                run_cmd(cmd)

            status = "Success"

            val_loss = float("inf")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    val_loss = float(
                        cdata.get("best_val_loss", cdata.get("val_loss", 999.0))
                    )

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {
                    "lr": lr,
                    "dropout": drop,
                    "emb_dim": emb_d,
                    "hidden_dim": hid_d,
                    "rnn_type": rnn,
                    "attention_type": attn,
                    "bidirectional": bidi,
                    "embedding_source": emb_src,
                    "freeze_emb": freeze,
                    "val_loss": val_loss,
                }

        except Exception as e:
            print(f"⚠️ Trial {idx} failed: {e}")
            status = "Failed"
            val_loss = float("inf")

        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "run_id": exp_id,
                "stage": stage,
                "token_type": token_type,
                "rnn_type": rnn,
                "attention_type": attn,
                "bidirectional": bidi,
                "learning_rate": lr,
                "dropout": drop,
                "emb_dim": emb_d,
                "hidden_dim": hid_d,
                "val_loss": val_loss,
                "status": status,
            })

    if best_params:
        summary_json = os.path.join(
            OUTPUT_DIR,
            f"best_config_TUNE_{token_type.upper()}_{stage.upper()}.json",
        )
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)
        print(
            f"\n🏆 Best Tuning Parameters Saved -> {summary_json} (Val"
            f" Loss: {best_loss:.4f})"
        )


def run_automated_post_processing(token_type, rnn_type):
    """Executes evaluation aggregation, pivot evaluation, and attention heatmap generation."""
    env = os.environ.copy()

    de_en_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_D2_{rnn_type}.pt"
    )
    en_sv_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_E1_{rnn_type}.pt"
    )
    if os.path.exists(de_en_model) and os.path.exists(en_sv_model):
        try:
            subprocess.run(
                [
                    sys.executable,
                    SELF_PATH,
                    "--task",
                    "pivot",
                    "--de_en_model",
                    de_en_model,
                    "--en_sv_model",
                    en_sv_model,
                    "--text",
                    "maschinelles lernen macht unglaublichen spass",
                ],
                check=True,
                env=env,
            )
        except Exception:
            pass

        print(
            "\n📊 Launching Formal Quantitative Pivot Dataset Evaluation (DE ➔"
            " EN ➔ SV)..."
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    SELF_PATH,
                    "--task",
                    "pivot",
                    "--de_en_model",
                    de_en_model,
                    "--en_sv_model",
                    en_sv_model,
                    "--evaluate",
                    "--token_type",
                    token_type,
                    "--experiment",
                    f"{token_type.upper()}_PIVOT",
                ],
                check=True,
                env=env,
            )
            sync_ledger_to_token_type(token_type)
        except Exception as e:
            print(
                "⚠️ Quantitative pivot dataset evaluation interrupted or"
                f" unsupported: {e}"
            )

    try:
        generate_all_reports(token_type)
    except Exception as e:
        print(f"⚠️ Error compiling reports: {e}")

    attn_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_C4_{rnn_type}.pt"
    )
    if not os.path.exists(attn_model):
        attn_model = os.path.join(
            OUTPUT_DIR, f"best_model_{token_type.upper()}_C3_{rnn_type}.pt"
        )

    if os.path.exists(attn_model):
        try:
            visualize_attention(attn_model)
        except Exception as e:
            print(f"⚠️ Error rendering attention heatmap: {e}")


def execute_study_a(epochs, token_type, eval_queue: AsyncEvaluationQueue):
    """Executes Study A: Recurrent Architecture Benchmarking (RNN vs GRU vs LSTM, Uni vs Bi)."""
    configs = [
        ("A1", "RNN", "False"),
        ("A2", "RNN", "True"),
        ("A3", "GRU", "False"),
        ("A4", "GRU", "True"),
        ("A5", "LSTM", "False"),
        ("A6", "LSTM", "True"),
    ]
    batch_size = get_batch_size("A", token_type)
    for exp, cell, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"
        hparams = get_best_hyperparameters("coarse", token_type, rnn_type=cell)

        emb_dim = (
            hparams[hparams.index("--emb_dim") + 1]
            if "--emb_dim" in hparams
            else "256"
        )
        hidden_dim = (
            hparams[hparams.index("--hidden_dim") + 1]
            if "--hidden_dim" in hparams
            else "512"
        )

        print_study_model_and_batch_info(
            study_name="Study A (Architecture Benchmarking)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=cell,
            bidirectional=bidi,
            attention_type="none",
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt")
        cfg_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json")

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    cell,
                    "--bidirectional",
                    bidi,
                    "--token_type",
                    token_type,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_b(
    epochs, rnn_type, bidirectional, token_type, eval_queue: AsyncEvaluationQueue
):
    """Executes Study B: Input Embedding Representation & Dimensionality Benchmarking (EN -> DE)."""
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = (
        [
            ("B1", "scratch", "False", "256"),
            ("B2", "word2vec", "True", "300"),
            ("B3", "word2vec", "False", "300"),
            ("B4", "scratch", "True", "256"),
            ("B5", "glove", "True", "300"),
            ("B6", "glove", "False", "300"),
        ]
        if token_type == "word"
        else [
            ("B7", "scratch", "False", "32"),
            ("B8", "scratch", "False", "64"),
            ("B9", "scratch", "False", "128"),
            ("B10", "onehot", "True", "128"),
        ]
    )
    batch_size = get_batch_size("B", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, freeze, emb_dim in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study B (Embedding Representation Analysis - EN->DE)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type="none",
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt")
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    "scratch" if src == "onehot" else src,
                    "--freeze_emb",
                    freeze,
                    "--emb_dim",
                    emb_dim,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_c(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study C: Attention Mechanism Optimization (Luong vs Bahdanau vs None) (EN -> DE)."""
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = [
        ("C1", rnn_type, "none", "False"),
        ("C2", rnn_type, "none", bidirectional),
        ("C3", rnn_type, "luong", bidirectional),
        ("C4", rnn_type, "bahdanau", bidirectional),
        ("C5", "RNN", "luong", "True"),
        ("C6", "RNN", "bahdanau", "True"),
    ]

    batch_size = get_batch_size("C", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, cell, attn, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study C (Attention Mechanism Optimization - EN->DE)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=cell,
            bidirectional=bidi,
            attention_type=attn,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt")
        cfg_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json")

        if not is_cache_valid(ckpt_path, cfg_path):
            cmd_hparams = (
                get_best_hyperparameters("coarse", token_type, rnn_type=cell)
                if cell == "RNN"
                else hparams
            )
            run_cmd(
                cmd_hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    cell,
                    "--attention_type",
                    attn,
                    "--bidirectional",
                    bidi,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--emb_dim",
                    emb_dim,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_d(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    attention_type,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study D: Translation Direction Optimization (EN ➔ DE vs DE ➔ EN)."""
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    configs = [("D1", "en", "de"), ("D2", "de", "en")]
    batch_size = get_batch_size("D", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, trg in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study D (Language Direction Optimization)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type=attention_type,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(
            OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"
        )
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--attention_type",
                    attention_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--src_lang",
                    src,
                    "--trg_lang",
                    trg,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_e(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    attention_type,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study E: Generalization & Swedish Pivot Channel Construction (EN ➔ SV)."""
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    configs = [("E1", "en", "sv")]
    batch_size = get_batch_size("E", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, trg in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study E (Generalization & Pivot Pipeline)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type=attention_type,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(
            OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"
        )
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--attention_type",
                    attention_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--src_lang",
                    src,
                    "--trg_lang",
                    trg,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def run_pipeline_for_token_type(target_token_type, args):
    """Runs tuning + Studies A-E end-to-end for one token type (word or char)."""
    TUNE_1_EPOCHS = args.epochs if args.epochs is not None else 4
    TUNE_2_EPOCHS = args.epochs if args.epochs is not None else 5
    STUDY_A_EPOCHS = args.epochs if args.epochs is not None else 8
    STUDY_B_EPOCHS = args.epochs if args.epochs is not None else 8
    # Study C's winner drives fine-tuning and D/E, so it gets extra epochs
    STUDY_C_EPOCHS = args.epochs if args.epochs is not None else 10
    # D2/E1 also feed the pivot evaluation chain, so they need full training too
    STUDY_DE_EPOCHS = args.epochs if args.epochs is not None else 10

    print("\n" + "═" * 80)
    print(f"🚀 EXECUTING PIPELINE FOR TOKEN LEVEL: {target_token_type.upper()}")
    print(
        f"   Mode: {args.study.upper()}"
        f" | Epoch Strategy: [Tune-coarse: {TUNE_1_EPOCHS} ({args.tune_trials} trials),"
        f" A: {STUDY_A_EPOCHS}, B: {STUDY_B_EPOCHS}, C: {STUDY_C_EPOCHS},"
        f" Tune-fine: {TUNE_2_EPOCHS} ({args.fine_tune_trials} trials), D/E: {STUDY_DE_EPOCHS}]"
        f" | GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    print("═" * 80 + "\n")

    if not args.no_preprocess and args.study != "postprocess":
        execute_preprocessing(token_type=target_token_type, mock_mode=args.mock)

    eval_queue = AsyncEvaluationQueue(max_workers=2)

    # 1. First Tuning Pass (Coarse Sweep) - 5 Epochs
    if args.study in ["tune", "all"] and args.tune_stage == "coarse":
        execute_tuning(
            stage="coarse",
            token_type=target_token_type,
            epochs=TUNE_1_EPOCHS,
            num_trials=args.tune_trials,
            configs_per_rnn=args.configs_per_rnn,
        )

    # 2. Studies A, B, C - 6 Epochs
    if args.study in ["all", "A"]:
        execute_study_a(STUDY_A_EPOCHS, target_token_type, eval_queue)

    best_settings = get_best_empirical_settings(target_token_type)

    if args.study in ["all", "B"]:
        execute_study_b(
            STUDY_B_EPOCHS,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            target_token_type,
            eval_queue,
        )

    best_settings = get_best_empirical_settings(target_token_type)

    if args.study in ["all", "C"]:
        execute_study_c(
            STUDY_C_EPOCHS,
            target_token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["emb_dim"],
            eval_queue,
        )

    # 3. Second Tuning Pass (Fine Sweep) - 6 Epochs
    if args.study in ["all", "fine_tune"] or (args.study == "tune" and args.tune_stage == "fine"):
        execute_tuning(
            stage="fine",
            token_type=target_token_type,
            epochs=TUNE_2_EPOCHS,
            num_trials=args.fine_tune_trials,
            configs_per_rnn=None,
        )

    best_settings = get_best_empirical_settings(target_token_type)

    # 4. Studies D and E - 6 Epochs
    if args.study in ["all", "D"]:
        execute_study_d(
            STUDY_DE_EPOCHS,
            target_token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["attention_type"],
            best_settings["emb_dim"],
            eval_queue,
        )

    if args.study in ["all", "E"]:
        execute_study_e(
            STUDY_DE_EPOCHS,
            target_token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["attention_type"],
            best_settings["emb_dim"],
            eval_queue,
        )

    eval_queue.shutdown()

    if args.study in ["all", "postprocess"]:
        run_automated_post_processing(target_token_type, best_settings["rnn_type"])


def run_studies_main():
    """CLI entry point: runs tuning + Studies A-E for word, char, or both."""
    setup_logging(log_filename="run_studies.log", log_dir=OUTPUT_DIR)

    parser = argparse.ArgumentParser(
        description="Master Empirical NMT Orchestrator Interface"
    )
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=["all", "A", "B", "C", "D", "E", "tune", "fine_tune", "postprocess"],
        help="Specify study suite to run or execute 'all'",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="word",
        choices=["word", "char", "both"],
        help="Tokenization level: 'word', 'char', or 'both' (sequential execution)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Execute in rapid mock mode with small synthetic sample dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch scheduling across all stages with explicit integer",
    )
    parser.add_argument(
        "--tune_stage",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="Hyperparameter tuning stage",
    )
    parser.add_argument(
        "--tune_trials",
        type=int,
        default=12,
        help="Number of coarse hyperparameter search trials, split across RNN/GRU/LSTM (e.g. 12 trials)",
    )
    parser.add_argument(
        "--fine_tune_trials",
        type=int,
        default=6,
        help="Number of fine hyperparameter search trials for the single Study C winner "
             "architecture. Deliberately decoupled from --tune_trials: fine-tuning re-searches "
             "the same lr/dropout/emb_dim/hidden_dim grid coarse tuning already sampled, just "
             "for one fixed architecture instead of three, so it needs fewer trials.",
    )
    parser.add_argument(
        "--configs_per_rnn",
        type=int,
        default=None,
        help="Explicitly force N trials per cell type in coarse mode",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Skip data preprocessing step if cached dataset files exist",
    )
    parser.add_argument(
        "--auto_shutdown",
        action="store_true",
        help="Stop this RunPod pod via the RunPod API once the pipeline finishes "
             "successfully, to avoid paying for idle GPU time. Safe with data on a "
             "persistent Network Volume. Only fires on a clean full completion - an "
             "exception anywhere in the pipeline skips it. Off by default.",
    )

    args = parser.parse_args()

    print("\n" + "═" * 80)
    print("🚀 NMT PERFORMANCE INFRASTRUCTURE ORCHESTRATOR INITIALIZED")
    print("═" * 80 + "\n")

    # Handle sequential pipeline execution if 'both' option is chosen
    if args.token_type == "both":
        print("🔄 [MODE SWITCH] Sequential pipeline execution for WORD and CHAR levels triggered.")
        for t_type in ["word", "char"]:
            run_pipeline_for_token_type(t_type, args)
    else:
        run_pipeline_for_token_type(args.token_type, args)

    print("\n" + "═" * 80)
    print(
        "🎉 MASTER ORCHESTRATION PIPELINE COMPLETED SUCCESSFULLY ON GPU/CPU"
        " CLUSTER"
    )
    print("═" * 80 + "\n")

    if args.auto_shutdown:
        from auto_shutdown import stop_this_pod
        stop_this_pod()



# Global config load + base seeding (was run_studies.py module-level init)

config = load_config(CONFIG_PATH)
set_seed(config.get("system", {}).get("seed", 42))
eval_lock = threading.Lock()


# Unified CLI dispatcher
def main():
    """Reads --task and dispatches to the matching sub-main, passing the rest of argv through untouched."""
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument(
        "--task",
        choices=["preprocess", "build-pivot-eval", "train", "evaluate", "pivot", "study"],
        default="study",
        help="Which stage of the pipeline to run (default: study, i.e. the full "
             "run_studies.py orchestrator).",
    )
    task_args, remaining = task_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if task_args.task == "preprocess":
        preprocess_main()
    elif task_args.task == "build-pivot-eval":
        build_pivot_eval_main()
    elif task_args.task == "train":
        train_main()
    elif task_args.task == "evaluate":
        evaluate_main()
    elif task_args.task == "pivot":
        pivot_main()
    elif task_args.task == "study":
        run_studies_main()


if __name__ == "__main__":
    main()
