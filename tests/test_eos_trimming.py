import ast
import unittest
from pathlib import Path

EOS_IDX = 3
EVALUATORS = (
    Path(__file__).resolve().parents[1] / "src" / "evaluate.py",
    Path(__file__).resolve().parents[1] / "nmt_pipeline.py",
)


def load_eos_helper(path):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "strip_terminal_eos"
    ]
    if len(functions) != 1:
        raise AssertionError(f"Expected one strip_terminal_eos function in {path}")

    function = functions[0]
    isolated_module = ast.Module(body=[function], type_ignores=[])
    namespace = {"EOS_IDX": EOS_IDX}
    exec(compile(isolated_module, str(path), "exec"), namespace)
    return ast.dump(function, include_attributes=False), namespace["strip_terminal_eos"]


class TerminalEosTrimmingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = [(path, *load_eos_helper(path)) for path in EVALUATORS]

    def test_modular_and_single_file_helpers_match(self):
        self.assertEqual(self.helpers[0][1], self.helpers[1][1])

    def test_removes_one_trailing_eos(self):
        for path, _ast, helper in self.helpers:
            with self.subTest(path=path.name):
                self.assertEqual(helper([8, 9, EOS_IDX]), [8, 9])
                self.assertEqual(helper([EOS_IDX, EOS_IDX]), [EOS_IDX])

    def test_preserves_sequences_without_terminal_eos(self):
        for path, _ast, helper in self.helpers:
            with self.subTest(path=path.name):
                self.assertEqual(helper([8, EOS_IDX, 9]), [8, EOS_IDX, 9])
                self.assertEqual(helper([]), [])


if __name__ == "__main__":
    unittest.main()
