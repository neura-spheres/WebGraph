"""
WebGraph — Desktop GUI
Run with: python gui.py
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

BASE_DIR = Path(__file__).parent


# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#0f1117"
BG2       = "#1a1d27"
BG3       = "#252836"
ACCENT    = "#6c63ff"
ACCENT2   = "#4ecca3"
DANGER    = "#ff6b6b"
WARNING   = "#ffd93d"
TEXT      = "#e0e0e0"
TEXT_DIM  = "#7a7a8e"
BORDER    = "#2e3148"

BTN_SETUP   = "#3d5a80"
BTN_CRAWL   = "#6c63ff"
BTN_INDEX   = "#2ecc71"
BTN_PR      = "#e67e22"
BTN_EXPORT  = "#9b59b6"
BTN_SERVE   = "#1abc9c"
BTN_ALL     = "#e74c3c"
BTN_STOP    = "#ff6b6b"

LOG_COLOURS = {
    "INFO":    "#4ecca3",
    "WARNING": "#ffd93d",
    "ERROR":   "#ff6b6b",
    "DEBUG":   "#7a7a8e",
    ">>":      "#6c63ff",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _btn(parent, text, color, command, width=16, pady=10):
    b = tk.Button(
        parent, text=text, bg=color, fg="white",
        font=("Segoe UI", 10, "bold"),
        relief="flat", cursor="hand2",
        activebackground=color, activeforeground="white",
        padx=10, pady=pady, width=width,
        command=command,
    )
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b


def _lighten(hex_color, amount=30):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = min(255, r+amount), min(255, g+amount), min(255, b+amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _label(parent, text, size=10, bold=False, color=TEXT, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, bg=BG2, fg=color,
                    font=("Segoe UI", size, weight), **kw)


def _entry(parent, textvariable=None, width=10):
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, width=width)
    if textvariable:
        e.config(textvariable=textvariable)
    return e


# ── Main Application ──────────────────────────────────────────────────────────
class NeuraSearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WebGraph Control Panel")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(960, 700)

        self._proc   = None        # current subprocess
        self._thread = None        # reader thread
        self._running = False

        # ── settings vars ──
        self.var_limit   = tk.StringVar(value="50000")
        self.var_workers = tk.StringVar(value="10")
        self.var_depth   = tk.StringVar(value="6")
        self.var_port    = tk.StringVar(value="8000")
        self.var_status  = tk.StringVar(value="Idle")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # centre window
        self.update_idletasks()
        w, h = 1080, 760
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG2, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="WebGraph", bg=BG2, fg=ACCENT,
                 font=("Segoe UI", 22, "bold")).pack(side="left", padx=20)
        tk.Label(hdr, text="Control Panel", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 13)).pack(side="left")
        self._status_pill = tk.Label(hdr, textvariable=self.var_status,
                                     bg=BG3, fg=ACCENT2,
                                     font=("Segoe UI", 10, "bold"),
                                     padx=14, pady=4, relief="flat")
        self._status_pill.pack(side="right", padx=20)

        # body: left panel + log
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        left = tk.Frame(body, bg=BG2, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG2)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_log(right)

    def _build_left(self, parent):
        # ── Seeds ──────────────────────────────────────────────────────────────
        self._section(parent, "Seed URLs")
        seed_frame = tk.Frame(parent, bg=BG2)
        seed_frame.pack(fill="x", padx=14, pady=(0, 8))

        self._seed_entry = tk.Entry(
            seed_frame, bg=BG3, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 10),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self._seed_entry.pack(side="left", fill="x", expand=True, ipady=5)
        self._seed_entry.bind("<Return>", lambda e: self._add_seed())

        add_btn = tk.Button(
            seed_frame, text="Add", bg=ACCENT, fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat",
            cursor="hand2", padx=8, pady=5,
            command=self._add_seed,
        )
        add_btn.pack(side="left", padx=(6, 0))

        # listbox for seeds
        lb_frame = tk.Frame(parent, bg=BG2)
        lb_frame.pack(fill="x", padx=14, pady=(0, 4))

        sb = tk.Scrollbar(lb_frame, orient="vertical", bg=BG3, troughcolor=BG)
        self._seed_list = tk.Listbox(
            lb_frame, bg=BG3, fg=TEXT, selectbackground=ACCENT,
            font=("Segoe UI", 9), relief="flat", height=5,
            yscrollcommand=sb.set,
            highlightthickness=1, highlightbackground=BORDER,
            activestyle="none",
        )
        sb.config(command=self._seed_list.yview)
        self._seed_list.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

        rem_btn = tk.Button(
            parent, text="Remove selected", bg=BG3, fg=DANGER,
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            padx=6, pady=3, command=self._remove_seed,
        )
        rem_btn.pack(anchor="e", padx=14, pady=(0, 10))

        # ── Crawl Settings ─────────────────────────────────────────────────────
        self._section(parent, "Crawl Settings")
        grid = tk.Frame(parent, bg=BG2)
        grid.pack(fill="x", padx=14, pady=(0, 12))

        settings = [
            ("Page limit",    self.var_limit,   "50000"),
            ("Workers",       self.var_workers, "10"),
            ("Max depth",     self.var_depth,   "6"),
            ("API port",      self.var_port,    "8000"),
        ]
        for i, (lbl, var, _) in enumerate(settings):
            tk.Label(grid, text=lbl, bg=BG2, fg=TEXT_DIM,
                     font=("Segoe UI", 9)).grid(row=i, column=0, sticky="w", pady=4)
            e = _entry(grid, textvariable=var, width=12)
            e.grid(row=i, column=1, sticky="e", pady=4, padx=(10, 0))
        grid.columnconfigure(0, weight=1)

        # ── Language Filter ────────────────────────────────────────────────────
        self._section(parent, "Language Filter")
        lang_hint = tk.Label(parent, text='Enter codes separated by commas\n(empty = no filter, e.g. "en,id")',
                             bg=BG2, fg=TEXT_DIM, font=("Segoe UI", 8), justify="left")
        lang_hint.pack(anchor="w", padx=14, pady=(0, 4))

        self._lang_entry = tk.Entry(
            parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 10),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self._lang_entry.insert(0, "en,id")
        self._lang_entry.pack(fill="x", padx=14, ipady=5, pady=(0, 14))

        # ── Command Buttons ────────────────────────────────────────────────────
        self._section(parent, "Commands")
        btn_grid = tk.Frame(parent, bg=BG2)
        btn_grid.pack(fill="x", padx=14, pady=(4, 0))

        buttons = [
            ("Setup NLTK",  BTN_SETUP,  self._run_setup),
            ("Crawl",       BTN_CRAWL,  self._run_crawl),
            ("Index",       BTN_INDEX,  self._run_index),
            ("PageRank",    BTN_PR,     self._run_pagerank),
            ("Export",      BTN_EXPORT, self._run_export),
            ("Serve API",   BTN_SERVE,  self._run_serve),
            ("Run All",     BTN_ALL,    self._run_all),
        ]
        self._cmd_buttons = []
        for i, (txt, col, cmd) in enumerate(buttons):
            b = _btn(btn_grid, txt, col, cmd, width=13, pady=8)
            b.grid(row=i//2, column=i%2, padx=4, pady=4, sticky="ew")
            self._cmd_buttons.append(b)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # stop button
        self._stop_btn = _btn(parent, "STOP", BTN_STOP, self._stop, width=28, pady=9)
        self._stop_btn.pack(fill="x", padx=14, pady=(10, 4))
        self._stop_btn.config(state="disabled")

    def _build_log(self, parent):
        header = tk.Frame(parent, bg=BG2)
        header.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(header, text="Console Output", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        clear_btn = tk.Button(
            header, text="Clear", bg=BG3, fg=TEXT_DIM,
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            padx=8, pady=2, command=self._clear_log,
        )
        clear_btn.pack(side="right")

        log_frame = tk.Frame(parent, bg=BG3,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self._log = scrolledtext.ScrolledText(
            log_frame, bg=BG3, fg=TEXT,
            font=("Cascadia Code", 9) if self._font_exists("Cascadia Code")
                 else ("Consolas", 9),
            relief="flat", state="disabled",
            wrap="word", padx=10, pady=8,
        )
        self._log.pack(fill="both", expand=True)

        # colour tags
        for tag, colour in LOG_COLOURS.items():
            self._log.tag_config(tag, foreground=colour)
        self._log.tag_config("TIMESTAMP", foreground=TEXT_DIM)

    # ── Utility ───────────────────────────────────────────────────────────────
    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(f, text=title, bg=BG2, fg=ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                               expand=True, padx=(8, 0))

    def _font_exists(self, name):
        try:
            import tkinter.font as tkf
            return name in tkf.families()
        except Exception:
            return False

    def _add_seed(self):
        url = self._seed_entry.get().strip()
        if url:
            self._seed_list.insert(tk.END, url)
            self._seed_entry.delete(0, tk.END)

    def _remove_seed(self):
        sel = self._seed_list.curselection()
        for i in reversed(sel):
            self._seed_list.delete(i)

    def _get_seeds(self):
        return list(self._seed_list.get(0, tk.END))

    def _get_langs(self):
        raw = self._lang_entry.get().strip()
        if not raw:
            return []
        return [l.strip() for l in raw.split(",") if l.strip()]

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_write(self, text):
        self._log.config(state="normal")
        import re, datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log.insert(tk.END, f"[{ts}] ", "TIMESTAMP")

        # detect log level keyword
        matched = False
        for kw in ("ERROR", "WARNING", "INFO", "DEBUG"):
            if kw in text:
                self._log.insert(tk.END, text + "\n", kw)
                matched = True
                break
        if not matched:
            self._log.insert(tk.END, text + "\n", ">>")

        self._log.see(tk.END)
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", tk.END)
        self._log.config(state="disabled")

    # ── Process management ────────────────────────────────────────────────────
    def _set_running(self, running: bool, label="Running…"):
        self._running = running
        state = "disabled" if running else "normal"
        for b in self._cmd_buttons:
            b.config(state=state)
        self._stop_btn.config(state="normal" if running else "disabled")
        self.var_status.set(label if running else "Idle")
        self._status_pill.config(fg=WARNING if running else ACCENT2)

    def _stream_proc(self, cmd, label):
        """Run cmd in a subprocess and stream its output to the log."""
        self._log_write(f">> {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(BASE_DIR),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    self.after(0, self._log_write, line)
            self._proc.wait()
            rc = self._proc.returncode
            msg = f"Process finished (exit code {rc})"
            self.after(0, self._log_write, msg)
        except Exception as exc:
            self.after(0, self._log_write, f"ERROR: {exc}")
        finally:
            self.after(0, self._set_running, False)

    def _launch(self, cmd, label):
        if self._running:
            messagebox.showwarning("Busy", "A command is already running.\nStop it first.")
            return
        self._set_running(True, label)
        self._thread = threading.Thread(
            target=self._stream_proc, args=(cmd, label), daemon=True
        )
        self._thread.start()

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._log_write(">> Stopping process…")
            try:
                import signal
                if sys.platform == "win32":
                    self._proc.terminate()
                else:
                    self._proc.send_signal(signal.SIGINT)
            except Exception as e:
                self._log_write(f"WARNING: Could not send stop signal: {e}")
                self._proc.terminate()
        self._set_running(False)

    def _python(self):
        return sys.executable

    # ── Command launchers ─────────────────────────────────────────────────────
    def _run_setup(self):
        self._launch([self._python(), "main.py", "setup"], "Setup…")

    def _run_crawl(self):
        seeds = self._get_seeds()
        if not seeds:
            messagebox.showwarning("No Seeds", "Add at least one seed URL before crawling.")
            return
        langs = self._get_langs()
        cmd = [
            self._python(), "main.py", "crawl",
            *seeds,
            "--limit",   self.var_limit.get()   or "50000",
            "--workers", self.var_workers.get() or "10",
            "--depth",   self.var_depth.get()   or "6",
        ]
        # patch config before launch
        self._patch_config_languages(langs)
        self._launch(cmd, "Crawling…")

    def _run_index(self):
        self._launch([self._python(), "main.py", "index"], "Indexing…")

    def _run_pagerank(self):
        self._launch([self._python(), "main.py", "pagerank"], "PageRank…")

    def _run_export(self):
        self._launch([self._python(), "main.py", "export"], "Exporting…")

    def _run_serve(self):
        cmd = [
            self._python(), "main.py", "serve",
            "--port", self.var_port.get() or "8000",
        ]
        self._launch(cmd, "Serving…")

    def _run_all(self):
        seeds = self._get_seeds()
        if not seeds:
            messagebox.showwarning("No Seeds", "Add at least one seed URL.")
            return
        langs = self._get_langs()
        cmd = [
            self._python(), "main.py", "all",
            "--seeds", *seeds,
            "--limit",   self.var_limit.get()   or "50000",
            "--workers", self.var_workers.get() or "10",
            "--depth",   self.var_depth.get()   or "6",
            "--port",    self.var_port.get()     or "8000",
        ]
        self._patch_config_languages(langs)
        self._launch(cmd, "Running all…")

    # ── Config patching ───────────────────────────────────────────────────────
    def _patch_config_languages(self, langs):
        """Rewrite CRAWL_LANGUAGES in config.py to match UI setting."""
        try:
            config_path = BASE_DIR / "config.py"
            text = config_path.read_text(encoding="utf-8")
            import re
            new_val = repr(langs)
            new_text = re.sub(
                r'^(CRAWL_LANGUAGES\s*:\s*list\s*=\s*).*$',
                rf'\g<1>{new_val}   # set by GUI',
                text,
                flags=re.MULTILINE,
            )
            config_path.write_text(new_text, encoding="utf-8")
            self._log_write(f"INFO  config.py: CRAWL_LANGUAGES = {new_val}")
        except Exception as e:
            self._log_write(f"WARNING: Could not patch config.py: {e}")

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        if self._running:
            if not messagebox.askyesno("Quit", "A command is still running. Stop it and quit?"):
                return
            self._stop()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = NeuraSearchGUI()
    app.mainloop()
