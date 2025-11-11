#!/usr/bin/env python3
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, Set, Tuple, List

from .config import load_config, save_config

EXCLUDED_DIRS = {".git", "venv", "node_modules", "__pycache__"}

CHECKED = "☑"
UNCHECKED = "☐"

class SourceSelectorApp:
    def __init__(self, root: tk.Tk, repo_root: Path):
        self.root = root
        self.repo_root = repo_root

        # state
        self.show_only_code = tk.BooleanVar(value=True)
        self.item_path: Dict[str, Path] = {}       # tree item id -> absolute path
        self.item_is_dir: Dict[str, bool] = {}
        self.checked: Dict[str, bool] = {}         # tree item id -> checked?
        self.path_to_item: Dict[Path, str] = {}    # absolute path -> tree item id

        self.cfg = load_config(self.repo_root)
        # Preload any existing selections to pre-check in UI
        self.pre_selected_files: Set[Path] = {
            (self.repo_root / f).resolve() for f in self.cfg.get("include_files", [])
        }
        self.pre_selected_dirs: Set[Path] = {
            (self.repo_root / d).resolve() for d in self.cfg.get("include_dirs", [])
        }

        self._build_ui()
        self._populate_tree()

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title("Code Similarity – Source Selection")
        self.root.geometry("900x640")

        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(side="top", fill="x")

        ttk.Checkbutton(
            top,
            text="Show only Python/Java files",
            variable=self.show_only_code,
            command=self._rebuild_tree
        ).pack(side="left")

        ttk.Button(top, text="Expand All", command=self._expand_all).pack(side="left", padx=(12, 0))
        ttk.Button(top, text="Collapse All", command=self._collapse_all).pack(side="left", padx=(6, 0))

        # Tree
        self.tree = ttk.Treeview(self.root, columns=("check", "name"), show="tree")
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Bind click in the tree to toggle
        self.tree.bind("<Button-1>", self._on_click)

        # Bottom bar
        bottom = ttk.Frame(self.root, padding=(8, 6))
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(bottom, text="Select None", command=self._select_none).pack(side="left", padx=(6, 0))
        ttk.Button(bottom, text="Apply", command=self._apply).pack(side="right")

    # ---------- Populate ----------
    def _rebuild_tree(self):
        for i in self.tree.get_children(""):
            self.tree.delete(i)
        self.item_path.clear()
        self.item_is_dir.clear()
        self.checked.clear()
        self.path_to_item.clear()
        self._populate_tree()

    def _populate_tree(self):
        # Build root node (repo)
        root_id = self.tree.insert("", "end", text=f"{self.repo_root.name} {UNCHECKED}", open=True)
        self.item_path[root_id] = self.repo_root
        self.item_is_dir[root_id] = True
        self.checked[root_id] = False
        self.path_to_item[self.repo_root] = root_id

        self._add_children(self.repo_root, root_id)
        # Expand one level by default
        for child in self.tree.get_children(root_id):
            self.tree.item(child, open=True)

    def _add_children(self, directory: Path, parent_id: str):
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return

        for p in entries:
            name = p.name
            if name in EXCLUDED_DIRS:
                continue

            if p.is_dir():
                node_id = self.tree.insert(parent_id, "end", text=f"{name} {UNCHECKED}", open=False)
                self.item_path[node_id] = p
                self.item_is_dir[node_id] = True
                self.checked[node_id] = (p.resolve() in self.pre_selected_dirs)
                self.path_to_item[p.resolve()] = node_id

                # If pre-selected dir, cascade to current children later when expanding
                self._add_children(p, node_id)
                self._refresh_label(node_id)
            else:
                if self.show_only_code.get() and not self._is_supported_code(p):
                    continue
                node_id = self.tree.insert(parent_id, "end", text=f"{name} {UNCHECKED}", open=False)
                self.item_path[node_id] = p
                self.item_is_dir[node_id] = False
                self.checked[node_id] = (p.resolve() in self.pre_selected_files)
                self.path_to_item[p.resolve()] = node_id
                self._refresh_label(node_id)

    # ---------- Helpers ----------
    def _is_supported_code(self, p: Path) -> bool:
        ext = p.suffix.lower()
        return ext in {".py", ".java"}

    def _refresh_label(self, item_id: str):
        checked = self.checked.get(item_id, False)
        path = self.item_path[item_id]
        icon = CHECKED if checked else UNCHECKED
        self.tree.item(item_id, text=f"{path.name} {icon}")

    def _toggle_item(self, item_id: str, desired: bool | None = None):
        """Toggle checkbox state. If folder, cascade to descendants."""
        current = self.checked.get(item_id, False)
        new_state = (not current) if desired is None else desired
        self.checked[item_id] = new_state
        self._refresh_label(item_id)

        if self.item_is_dir.get(item_id, False):
            # cascade to children
            for child in self.tree.get_children(item_id):
                self._set_state_recursive(child, new_state)

    def _set_state_recursive(self, item_id: str, state: bool):
        self.checked[item_id] = state
        self._refresh_label(item_id)
        for child in self.tree.get_children(item_id):
            self._set_state_recursive(child, state)

    def _collect_checked(self) -> Tuple[List[Path], List[Path]]:
        """Return (dirs, files) that are checked."""
        inc_dirs: List[Path] = []
        inc_files: List[Path] = []
        for item_id, is_checked in self.checked.items():
            if not is_checked:
                continue
            p = self.item_path[item_id]
            if self.item_is_dir.get(item_id, False):
                inc_dirs.append(p)
            else:
                inc_files.append(p)
        # De-duplicate / sort
        inc_dirs = sorted({d.resolve() for d in inc_dirs})
        inc_files = sorted({f.resolve() for f in inc_files})
        return inc_dirs, inc_files

    # ---------- Events ----------
    def _on_click(self, event):
        # Identify the item; toggle when clicking on row (simple + robust)
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return
        self._toggle_item(row_id)

    def _expand_all(self):
        for iid in self.tree.get_children(""):
            self._expand_recursive(iid, True)

    def _collapse_all(self):
        for iid in self.tree.get_children(""):
            self._expand_recursive(iid, False)

    def _expand_recursive(self, item_id: str, open_: bool):
        self.tree.item(item_id, open=open_)
        for child in self.tree.get_children(item_id):
            self._expand_recursive(child, open_)

    def _select_all(self):
        for iid in list(self.item_path.keys()):
            self.checked[iid] = True
            self._refresh_label(iid)

    def _select_none(self):
        for iid in list(self.item_path.keys()):
            self.checked[iid] = False
            self._refresh_label(iid)
            
    def _apply(self):
        inc_dirs, inc_files = self._collect_checked()

        def rel(p: Path) -> str:
            return str(p.resolve().relative_to(self.repo_root.resolve())).replace("\\", "/")

        self.cfg["include_dirs"] = [rel(d) for d in inc_dirs]
        self.cfg["include_files"] = [rel(f) for f in inc_files]

        save_config(self.repo_root, self.cfg)
        messagebox.showinfo("Saved", f"Selection saved to {self.repo_root / '.git' / '.code-sim-config.json'}")
        self.root.destroy()

def main():
    # Tool is at repo_root/code_similarity_tool/select_sources.py
    repo_root = Path(__file__).resolve().parents[1]  # …/repo
    root = tk.Tk()
    app = SourceSelectorApp(root, repo_root)
    root.mainloop()

if __name__ == "__main__":
    main()
