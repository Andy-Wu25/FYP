from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code_similarity_tool.runtime import load_runtime_context


class RuntimeCollectionsTest(unittest.TestCase):
    def test_collection_defaults_derive_from_base_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "CODE_SIM_COLLECTION": "demo",
                "CODE_SIM_PRIVATE_COLLECTION": "",
                "CODE_SIM_PUBLIC_COLLECTION": "",
            }
            with patch.dict(os.environ, env, clear=False):
                ctx = load_runtime_context(root)

            self.assertEqual(ctx.private_collection_name, "demo_private")
            self.assertEqual(ctx.public_collection_name, "demo_public")


if __name__ == "__main__":
    unittest.main()
