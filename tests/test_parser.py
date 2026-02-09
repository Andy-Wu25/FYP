from __future__ import annotations

import unittest
from pathlib import Path

from code_similarity_tool.code_parser import detect_lang, extract_code_elements


class ParserTest(unittest.TestCase):
    def test_detect_lang(self) -> None:
        self.assertEqual(detect_lang(Path("x.py")), "python")
        self.assertEqual(detect_lang(Path("x.java")), "java")
        self.assertIsNone(detect_lang(Path("x.txt")))

    def test_extract_python_function(self) -> None:
        src = b"def hello(a, b):\n    return a + b\n"
        elements = extract_code_elements(Path("sample.py"), src)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["name"], "hello")
        self.assertEqual(elements[0]["kind"], "function")


if __name__ == "__main__":
    unittest.main()
