"""Desktop UI. Run with: python indicbert_browser_ui.py"""
from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from browser_pipeline import RunSettings, prepare_frame, run_pipeline, atomic_write, PendingTranslations
    from batch_runner import BatchSettings, preview_batches, run_batch_pipeline
except ModuleNotFoundError as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Missing Python package", f"Missing: {exc.name}\nRun setup_windows.bat first, or:\npython -m pip install -r requirements.txt")
    root.destroy()
    raise SystemExit(1)

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "ui_settings.json"


class Application:
    def __init__(self, root):
        self.root = root
        root.title("IndicBERT · Google Translate · Browser Recovery")
        root.geometry(f"1140x{min(920, root.winfo_screenheight() - 80)}")
        root.minsize(920, 700)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.last_folder = None
        self.closing = False
        self.preview_ids = {}
        self.skipped_count = 0
        defaults = {}
        if SETTINGS_FILE.exists():
            try:
                defaults = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        candidates = list(APP_DIR.glob("combined4_clean*.xlsx"))
        self.input_var = tk.StringVar(value=defaults.get("input_file", str(candidates[0]) if len(candidates) == 1 else ""))
        self.output_var = tk.StringVar(value=defaults.get("output_dir", str(APP_DIR / "results")))
        self.start_var = tk.StringVar(value=str(defaults.get("start_row", 1)))
        self.end_var = tk.StringVar(value=str(defaults.get("end_row", 10)))
        self.batch_mode_var = tk.BooleanVar(value=defaults.get("batch_mode", True))
        self.batch_size_var = tk.StringVar(value=str(defaults.get("batch_size", 1000)))
        self.max_batches_var = tk.StringVar(value=str(defaults.get("max_batches", 100)))
        self.full_dataset_var = tk.BooleanVar(value=defaults.get("full_dataset", False))
        self.language_var = tk.StringVar(value=defaults.get("source_language", "auto"))
        self.skip_failed_var = tk.BooleanVar(value=defaults.get("skip_failed_rows", True))
        interval = defaults.get("interval", 2) if defaults.get("ui_version") == 2 else 2
        self.interval_var = tk.StringVar(value=str(interval))
        self.chrome_var = tk.StringVar(value=defaults.get("chrome_binary", ""))
        self.driver_var = tk.StringVar(value=defaults.get("chromedriver_path", ""))
        self.status_var = tk.StringVar(value="Choose an Excel file and a row range.")
        self.range_var = tk.StringVar()
        self.overall_var = tk.StringVar(value="Batch mode: 1,000 rows × 100 batches by default.")
        self.batch_label_var = tk.StringVar(value="")
        self.start_var.trace_add("write", self._range_label)
        self.end_var.trace_add("write", self._range_label)
        self.batch_size_var.trace_add("write", self._range_label)
        self.max_batches_var.trace_add("write", self._range_label)
        self._build()
        self._mode_fields()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)

    def _build(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#142d4e")
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=30)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        main = ttk.Frame(self.root, padding=22)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(10, weight=1)
        ttk.Label(main, text="Translate posts and save batch results", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(main, text="Original cleaning and sentiment. Automatic browser recovery; pending rows retry after other batches.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 12))
        ttk.Label(main, text="Input Excel").grid(row=2, column=0, sticky="w", padx=(0, 14))
        self.input_entry = ttk.Entry(main, textvariable=self.input_var)
        self.input_entry.grid(row=2, column=1, sticky="ew")
        self.input_button = ttk.Button(main, text="Browse…", command=self._choose_input)
        self.input_button.grid(row=2, column=2, padx=(10, 0))
        ttk.Label(main, text="Output folder").grid(row=3, column=0, sticky="w", pady=8)
        self.output_entry = ttk.Entry(main, textvariable=self.output_var)
        self.output_entry.grid(row=3, column=1, sticky="ew")
        self.output_button = ttk.Button(main, text="Browse…", command=self._choose_output)
        self.output_button.grid(row=3, column=2, padx=(10, 0))
        range_frame = ttk.Frame(main)
        range_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 3))
        ttk.Label(range_frame, text="Start row").pack(side="left")
        self.start_entry = ttk.Entry(range_frame, textvariable=self.start_var, width=10)
        self.start_entry.pack(side="left", padx=(8, 24))
        ttk.Label(range_frame, text="End row").pack(side="left")
        self.end_entry = ttk.Entry(range_frame, textvariable=self.end_var, width=10)
        self.end_entry.pack(side="left", padx=(8, 24))
        ttk.Label(range_frame, textvariable=self.range_var).pack(side="left")
        ttk.Label(main, text="Both ends included. Data row 1 is the first record (Excel row 2); the header is excluded.").grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 12))
        batches = ttk.LabelFrame(main, text="Batch run", padding=8)
        batches.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.batch_check = ttk.Checkbutton(batches, text="Batch mode", variable=self.batch_mode_var, command=self._mode_fields)
        self.batch_check.grid(row=0, column=0, padx=(0, 14))
        ttk.Label(batches, text="Rows per batch").grid(row=0, column=1)
        self.batch_size_entry = ttk.Entry(batches, textvariable=self.batch_size_var, width=8)
        self.batch_size_entry.grid(row=0, column=2, padx=(6, 16))
        ttk.Label(batches, text="Max batches").grid(row=0, column=3)
        self.max_batches_entry = ttk.Entry(batches, textvariable=self.max_batches_var, width=8)
        self.max_batches_entry.grid(row=0, column=4, padx=(6, 16))
        self.full_check = ttk.Checkbutton(batches, text="Full dataset (ignore max)", variable=self.full_dataset_var, command=self._mode_fields)
        self.full_check.grid(row=0, column=5, sticky="w")
        ttk.Label(batches, text="Batch mode starts at Start row. Its end is calculated from the batch settings; End row applies only to range mode.").grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))
        options = ttk.LabelFrame(main, text="Translation settings", padding=10)
        options.grid(row=7, column=0, columnspan=3, sticky="ew")
        ttk.Label(options, text="Source language").grid(row=0, column=0, sticky="w")
        self.language_combo = ttk.Combobox(options, textvariable=self.language_var, values=("auto", "hi", "bn", "en"), width=10, state="readonly")
        self.language_combo.grid(row=0, column=1, padx=(8, 24))
        ttk.Label(options, text="English output · Request gap (seconds)").grid(row=0, column=2, sticky="w")
        self.interval_entry = ttk.Entry(options, textvariable=self.interval_var, width=8)
        self.interval_entry.grid(row=0, column=3, padx=8)
        ttk.Label(options, text="Chrome / driver paths are optional; leave blank for automatic discovery.").grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 4))
        ttk.Label(options, text="Chrome executable").grid(row=2, column=0, sticky="w")
        self.chrome_entry = ttk.Entry(options, textvariable=self.chrome_var, width=40)
        self.chrome_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=8)
        ttk.Label(options, text="ChromeDriver").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.driver_entry = ttk.Entry(options, textvariable=self.driver_var, width=40)
        self.driver_entry.grid(row=3, column=1, columnspan=3, sticky="ew", padx=8, pady=(5, 0))
        ttk.Label(options, text="Retry: Detect language → English in fresh Incognito, normal Chrome, then Search.").grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.skip_check = ttk.Checkbutton(options, text="Skip untranslated rows after all attempts (save reason; leave scores blank)", variable=self.skip_failed_var)
        self.skip_check.grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))
        options.columnconfigure(2, weight=1)
        controls = ttk.Frame(main)
        controls.grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        self.preview_button = ttk.Button(controls, text="Preview cleaned text", command=lambda: self._start(preview=True))
        self.preview_button.pack(side="left")
        self.start_button = ttk.Button(controls, text="Start / Resume", command=self._start)
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Button(controls, text="Open output folder", command=self._open_folder).pack(side="right")
        progress_area = ttk.Frame(main)
        progress_area.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(progress_area, textvariable=self.overall_var).pack(anchor="w")
        self.overall_bar = ttk.Progressbar(progress_area, mode="determinate")
        self.overall_bar.pack(fill="x", pady=(3, 4))
        ttk.Label(progress_area, textvariable=self.batch_label_var).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_area, mode="determinate")
        self.progress_bar.pack(fill="x")
        ttk.Label(progress_area, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))
        tabs = ttk.Notebook(main)
        tabs.grid(row=10, column=0, columnspan=3, sticky="nsew")
        table_frame = ttk.Frame(tabs)
        log_frame = ttk.Frame(tabs)
        tabs.add(table_frame, text="Cleaned text & English (up to 50 rows)")
        tabs.add(log_frame, text="Run log")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table = ttk.Treeview(table_frame, columns=("data", "excel", "status", "cleaned", "english"), show="headings")
        for name, label, width in (("data", "Data row", 70), ("excel", "Excel row", 70), ("status", "Status", 95), ("cleaned", "Original cleaning result", 365), ("english", "Google English result", 365)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, minwidth=60, stretch=name in ("cleaned", "english"))
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.table.bind("<Double-1>", self._show_row)
        ttk.Label(table_frame, text="Double-click a row to read its complete cleaned and translated text.").grid(row=2, column=0, sticky="w", pady=5)
        self.log_widget = tk.Text(log_frame, wrap="word", font=("Consolas", 10), background="#ffffff", state="disabled")
        self.log_widget.pack(fill="both", expand=True)
        self.editable = [self.input_entry, self.output_entry, self.start_entry, self.end_entry,
                         self.interval_entry, self.chrome_entry, self.driver_entry,
                         self.input_button, self.output_button, self.preview_button, self.start_button,
                         self.batch_check, self.batch_size_entry, self.max_batches_entry, self.full_check,
                         self.skip_check]

    def _mode_fields(self):
        batch = self.batch_mode_var.get()
        self.end_entry.configure(state="disabled" if batch else "normal")
        self.batch_size_entry.configure(state="normal" if batch else "disabled")
        self.max_batches_entry.configure(state="normal" if batch and not self.full_dataset_var.get() else "disabled")
        self.full_check.configure(state="normal" if batch else "disabled")
        self._range_label()

    def _range_label(self, *_):
        try:
            if self.batch_mode_var.get():
                size, maximum = int(self.batch_size_var.get()), int(self.max_batches_var.get())
                if size < 1 or maximum < 1 or int(self.start_var.get()) < 1:
                    raise ValueError()
                self.range_var.set(f"All remaining rows, {size:,} per batch" if self.full_dataset_var.get()
                                   else f"Up to {maximum:,} × {size:,} = {maximum * size:,} records")
                return
            start, end = int(self.start_var.get()), int(self.end_var.get())
            self.range_var.set(f"{end - start + 1:,} selected records" if 1 <= start <= end else "Enter a valid range")
        except ValueError:
            self.range_var.set("Enter whole row numbers")

    def _choose_input(self):
        path = filedialog.askopenfilename(filetypes=[("Excel workbook", "*.xlsx *.xls")])
        if path:
            self.input_var.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def _settings(self):
        if not self.output_var.get().strip():
            raise ValueError("Choose an output folder.")
        batch = self.batch_mode_var.get()
        cls = BatchSettings if batch else RunSettings
        extra = {"batch_size": int(self.batch_size_var.get()), "max_batches": int(self.max_batches_var.get()),
                 "full_dataset": self.full_dataset_var.get()} if batch else {}
        settings = cls(
            input_file=Path(self.input_var.get().strip()).expanduser().resolve(),
            output_dir=Path(self.output_var.get().strip()).expanduser().resolve(),
            start_row=int(self.start_var.get()), end_row=10 if batch else int(self.end_var.get()),
            source_language=self.language_var.get(), interval=float(self.interval_var.get()),
            skip_failed_rows=self.skip_failed_var.get(),
            chrome_binary=self.chrome_var.get().strip(), chromedriver_path=self.driver_var.get().strip(),
            **extra,
        )
        settings.validate()
        return settings

    def _start(self, preview=False):
        if self.worker is not None and self.worker.is_alive():
            return
        self.skipped_count = 0
        try:
            settings = self._settings()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check settings", str(exc))
            return
        preferences = {k: str(v) if isinstance(v, Path) else v for k, v in settings.__dict__.items()}
        preferences.update({"ui_version": 2, "batch_mode": self.batch_mode_var.get(),
                            "batch_size": int(self.batch_size_var.get()), "max_batches": int(self.max_batches_var.get()),
                            "full_dataset": self.full_dataset_var.get()})
        try:
            atomic_write(SETTINGS_FILE, lambda path: path.write_text(json.dumps(preferences, indent=2), encoding="utf-8"))
        except OSError:
            pass
        self.stop_event.clear()
        self._busy(True)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(15)
        self.status_var.set("Reading dataset and applying original cleaning…")
        self._log(f"Python: {sys.executable}")
        if isinstance(settings, BatchSettings):
            self._log(f"Batch run from data row {settings.start_row}; {settings.batch_size} rows per batch; full dataset={settings.full_dataset}.")
        else:
            self._log(f"Selected data rows: {settings.start_row}–{settings.end_row}; Excel rows: {settings.start_row + 1}–{settings.end_row + 1}")
        for row in self.table.get_children():
            self.table.delete(row)
        self.preview_ids.clear()

        def log(*parts):
            self.events.put(("log", " ".join(str(p) for p in parts)))

        last_progress = [0.0]

        def progress(stage, current, total, row):
            # Bound the UI event queue when thousands of cached rows finish
            # rapidly. Keep the first 50 preview records and the final count.
            if stage == "translation" and current > 50 and current != total:
                now = time.monotonic()
                if now - last_progress[0] < 0.1:
                    return
                last_progress[0] = now
            self.events.put(("progress", (stage, current, total, row)))

        def work():
            try:
                if preview:
                    if isinstance(settings, BatchSettings):
                        report, folder, plan, total = preview_batches(settings, log, self.stop_event)
                        progress("plan", 0, total, {"folder": str(folder), "batches": len(plan), "last_row": plan[-1]["end"]})
                    else:
                        _, report, folder = prepare_frame(settings, log, self.stop_event)
                    progress("prepared", 0, len(report), report.head(50).to_dict("records"))
                    self.events.put(("preview_done", str(folder)))
                else:
                    runner = run_batch_pipeline if isinstance(settings, BatchSettings) else run_pipeline
                    output = runner(settings, self.stop_event, log, progress)
                    self.events.put(("done", str(output)))
            except InterruptedError as exc:
                log(str(exc))
                self.events.put(("stopped", str(exc)))
            except PendingTranslations as exc:
                log(str(exc))
                self.events.put(("pending", str(exc)))
            except Exception as exc:
                log(traceback.format_exc())
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _busy(self, busy):
        for widget in self.editable:
            widget.configure(state="disabled" if busy else "normal")
        self.language_combo.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if not busy:
            self._mode_fields()

    def _stop(self):
        self.stop_event.set()
        self.status_var.set("Stop requested. Finishing the current read/model operation and saving progress…")
        self.stop_button.configure(state="disabled")

    def _log(self, text):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text + "\n")
        if int(self.log_widget.index("end-1c").split(".")[0]) > 2000:
            self.log_widget.delete("1.0", "501.0")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _update_row(self, row):
        data_id = int(row["data_row"])
        if data_id not in self.preview_ids and len(self.preview_ids) >= 50:
            return
        values = (data_id, row["excel_row"], row["translation_status"], row["post_text_original"], row["post_text_original_english"])
        if data_id in self.preview_ids:
            self.table.item(self.preview_ids[data_id], values=values)
        else:
            self.preview_ids[data_id] = self.table.insert("", "end", values=values)

    def _poll(self):
        try:
            # Limit each UI turn so very large ranges do not freeze the window.
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._log(value)
                elif kind == "progress":
                    stage, current, total, row = value
                    if stage == "plan":
                        self.last_folder = Path(row["folder"])
                        self.overall_bar.configure(maximum=max(1, total), value=0)
                        self.overall_var.set(f"Plan: {row['batches']:,} batches, {total:,} records, through data row {row['last_row']:,}.")
                    elif stage == "overall":
                        self.overall_bar.configure(maximum=max(1, total), value=current)
                        self.overall_var.set(f"Saved/processed rows: {current:,}/{total:,} ({100 * current / max(1, total):.1f}%)")
                    elif stage == "batch":
                        self.last_folder = Path(row["folder"])
                        self.batch_label_var.set(f"Batch {current:,}/{total:,} · data rows {row['start']:,}–{row['end']:,}")
                        for item in self.table.get_children():
                            self.table.delete(item)
                        self.preview_ids.clear()
                    elif stage == "batch_saved":
                        self.batch_label_var.set(f"Batch {current:,}/{total:,} CSV saved: {Path(row).name}")
                    elif stage == "prepared":
                        self.progress_bar.stop()
                        self.progress_bar.configure(mode="determinate", maximum=max(1, total), value=0)
                        for item in row:
                            self._update_row(item)
                    elif stage == "model":
                        self.progress_bar.configure(mode="indeterminate")
                        self.progress_bar.start(15)
                        self.status_var.set("Running the original sentiment model; loaded model is reused between batches…")
                    elif stage == "translation":
                        self.progress_bar.stop()
                        self.progress_bar.configure(mode="determinate")
                        self.progress_bar.configure(maximum=max(1, total), value=current)
                        self.status_var.set(f"Translation: {current:,}/{total:,} selected rows processed")
                        self._update_row(row)
                    elif stage == "repair":
                        self.status_var.set(f"Retrying pending translations: pass {current}/{total}. Saved work is reused.")
                    elif stage == "skipped":
                        self.skipped_count = current
                        self._log(f"Skipped {current:,} untranslated rows; blank scores. Details: {row}")
                    elif stage == "sentiment":
                        self.progress_bar.stop()
                        self.progress_bar.configure(mode="determinate", maximum=max(1, total), value=current)
                        self.status_var.set(f"Original IndicBERT: batch {current:,}/{total:,}")
                else:
                    self.progress_bar.stop()
                    self._busy(False)
                    if kind == "done":
                        self.last_folder = Path(value).parent
                        note = f"\n\nSkipped translations: {self.skipped_count:,}. English/scores are blank; see skipped_rows.csv." if self.skipped_count else ""
                        self.status_var.set(f"Complete. Final output saved. Skipped translations: {self.skipped_count:,}.")
                        messagebox.showinfo("Complete", f"Final output saved:\n{value}" + note)
                    elif kind == "preview_done":
                        self.last_folder = Path(value)
                        self.status_var.set("Cleaned text preview ready. Click Start / Resume to translate and analyse.")
                    elif kind == "stopped":
                        self.status_var.set("Stopped. Click Start / Resume with the same settings to continue.")
                    elif kind == "pending":
                        self.status_var.set("Available work saved. Some translations remain pending; Start / Resume retries them.")
                        messagebox.showwarning("Pending translations saved", value)
                    elif kind == "error":
                        self.status_var.set("Run paused. See Run log; saved progress can be resumed.")
                        messagebox.showerror("Run stopped", value + "\n\nPrevious successful translations are saved. Check Run log and click Start / Resume when ready.")
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def _open_folder(self):
        folder = self.last_folder or Path(self.output_var.get()).expanduser()
        if not folder.is_dir():
            messagebox.showinfo("Output folder", "The folder will be created after Preview or Start.")
            return
        if sys.platform == "win32":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _show_row(self, _event):
        selection = self.table.selection()
        if not selection:
            return
        values = self.table.item(selection[0], "values")
        window = tk.Toplevel(self.root)
        window.title(f"Data row {values[0]} · Excel row {values[1]} · {values[2]}")
        window.geometry("850x570")
        text = tk.Text(window, wrap="word", font=("Segoe UI", 11), padx=15, pady=15)
        text.pack(fill="both", expand=True)
        text.insert("end", f"CLEANED SOURCE\n\n{values[3]}\n\nGOOGLE ENGLISH RESULT\n\n{values[4]}")
        text.configure(state="disabled")

    def _close(self):
        if self.worker is not None and self.worker.is_alive():
            self.closing = True
            self._stop()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
