"""Structural tests for pipeline config loading."""

import ast
from pathlib import Path
import unittest


class TestConfigLoadingEncoding(unittest.TestCase):
    def _init_from_config_open_calls(self, path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "config_path"
        ]

    def test_pipeline_config_files_are_read_as_utf8(self):
        for path in [
            "pipelines/idea2video_pipeline.py",
            "pipelines/script2video_pipeline.py",
        ]:
            with self.subTest(path=path):
                calls = self._init_from_config_open_calls(path)
                self.assertTrue(calls)
                self.assertTrue(
                    any(
                        keyword.arg == "encoding"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == "utf-8"
                        for call in calls
                        for keyword in call.keywords
                    )
                )


if __name__ == "__main__":
    unittest.main()
