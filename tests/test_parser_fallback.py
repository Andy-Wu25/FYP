from __future__ import annotations

import unittest
from pathlib import Path

from code_similarity_tool.code_parser import detect_lang, extract_code_elements


class ParserFallbackTest(unittest.TestCase):
    def test_detect_lang_supports_typescript_and_csharp(self) -> None:
        self.assertEqual(detect_lang(Path("x.ts")), "typescript")
        self.assertEqual(detect_lang(Path("x.cs")), "csharp")

    def test_unknown_language_falls_back_to_whole_file_embedding(self) -> None:
        src = b"line one\nline two"
        elements = extract_code_elements(Path("sample.foolang"), src)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["kind"], "file")
        self.assertEqual(elements[0]["start_line"], 1)
        self.assertEqual(elements[0]["end_line"], 2)
        self.assertEqual(elements[0]["name"], "<file:sample.foolang>")


if __name__ == "__main__":
    unittest.main()
