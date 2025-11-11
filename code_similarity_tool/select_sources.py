#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from .config import load_config, save_config

def main():
    repo_root = Path(__file__).resolve().parents[2]  # .../repo
    cfg = load_config(repo_root)

    root = tk.Tk()
    root.title("Code Similarity – Source Selection")
    root.geometry("800x600")

    # build tree
    tree = ttk.Treeview(root, columns=("type",), show="tree")
    tree.pack(fill="both", expand=True)

    # TODO: populate tree with repo files/dirs (excluding .git, venv...)
    #       attach checkboxes; track selections in sets

    def on_apply():
        # TODO: compute include_dirs, include_files from checked items
        cfg["include_dirs"] = sorted([...])     # repo-root relative
        cfg["include_files"] = sorted([...])
        save_config(repo_root, cfg)
        messagebox.showinfo("Saved", "Selection saved to .git/.code-sim-config.json")
        root.destroy()

    btn = ttk.Button(root, text="Apply", command=on_apply)
    btn.pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    main()
