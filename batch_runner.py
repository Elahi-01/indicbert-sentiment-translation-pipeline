"""Bounded batches, durable per-row progress and ordered CSV consolidation."""
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import hashlib
import json
import os
import sqlite3
import sys

import pandas as pd
import analysis_logic
from browser_pipeline import (
    RunSettings, GoogleTranslateBrowser, TranslationCache, TranslationFailure,
    BrowserBlocked, atomic_write, save_csv, digest_file, check_stop,
    split_post, validate_translation, PreservingTranslator, TRANSLATION_POLICY,
    SUCCESS_STATUSES, FINAL_STATUSES, browser_error_is_transient, TranslationUnavailable,
    PendingTranslations,
    untranslated_output_failure,
)

INFERENCE_ROWS = 16  # Same inference batch size as the original code.
VALUE_COLUMNS = [
    "post_text_original_english", "post_text_original_indic_neg",
    "post_text_original_indic_pos", "post_text_original_indic_compound",
    "post_text_original_indic_label", "post_text_original_english_textblob_polarity",
    "post_text_original_english_vader_neg", "post_text_original_english_vader_neu",
    "post_text_original_english_vader_pos", "post_text_original_english_vader_compound",
]


@dataclass(frozen=True)
class BatchSettings(RunSettings):
    batch_size: int = 1000
    max_batches: int = 100
    full_dataset: bool = False
    batch_retries: int = 2

    def validate(self):
        # End row is a range-mode field. Batch mode derives its effective end.
        RunSettings.validate(replace(self, end_row=max(self.start_row, self.end_row)))
        if self.batch_size < 1 or self.max_batches < 1 or self.batch_retries < 0:
            raise ValueError("Batch size and max batches must be positive; retries cannot be negative.")


def batch_plan(start_row, last_row, batch_size):
    if start_row < 1 or last_row < start_row or batch_size < 1:
        raise ValueError("Invalid batch plan.")
    return [
        {"number": number, "start": start, "end": min(start + batch_size - 1, last_row)}
        for number, start in enumerate(range(start_row, last_row + 1, batch_size), 1)
    ]


def prepare_batch_input(settings, log=print, stop_event=None):
    settings.validate()
    check_stop(stop_event)
    log("Reading the input Excel once for this session...")
    fingerprint = digest_file(settings.input_file)
    source = pd.read_excel(settings.input_file)  # Original pandas NA defaults.
    if fingerprint != digest_file(settings.input_file):
        raise ValueError("Input changed during reading. Close Excel and retry.")
    check_stop(stop_event)
    missing = {"post_text", "datetime"} - set(source.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if settings.start_row > len(source):
        raise ValueError(f"Start row is beyond the dataset's {len(source):,} data rows.")
    last_row = len(source) if settings.full_dataset else min(
        len(source), settings.start_row + settings.batch_size * settings.max_batches - 1
    )
    frame = source.iloc[settings.start_row - 1:last_row].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["month_year"] = frame["datetime"].dt.to_period("M")
    frame["post_text_original"] = frame["post_text"].apply(analysis_logic.clean_text)
    frame["post_text_original"] = frame["post_text_original"].apply(analysis_logic.remove_hashtags)
    frame["post_text_original"] = frame["post_text_original"].apply(analysis_logic.remove_underscore_words)
    frame["post_text_original"] = frame["post_text_original"].apply(analysis_logic.remove_repeated_punctuation)
    plan = batch_plan(settings.start_row, last_row, settings.batch_size)
    identity = {
        "format_version": 2,
        "input_sha256": fingerprint,
        "start_row": settings.start_row,
        "end_row": last_row,
        "batch_size": settings.batch_size,
        "source_language": settings.source_language,
        "target_language": "en",
        "analysis_sha256": digest_file(Path(analysis_logic.__file__)),
        "translation_policy": TRANSLATION_POLICY,
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    folder = Path(settings.output_dir) / f"batch_run_{settings.start_row}_{last_row}_b{settings.batch_size}_{signature}"
    folder.mkdir(parents=True, exist_ok=True)
    log(f"Plan: {len(plan):,} batches; data rows {settings.start_row:,}–{last_row:,}; {len(frame):,} records.")
    if not settings.full_dataset and last_row < len(source):
        log(f"{len(source) - last_row:,} later records are outside this plan. Use Full dataset to include them.")
    return frame, plan, folder, identity


def report_for(frame):
    report = pd.DataFrame({
        "data_row": frame.index + 1, "excel_row": frame.index + 2,
        "post_text_original": frame["post_text_original"],
    })
    report["post_text_original_english"] = ""
    report["translation_status"] = "pending"
    report["analysis_status"] = "pending"
    report["translation_error"] = ""
    return report


def preview_batches(settings, log=print, stop_event=None):
    frame, plan, folder, _ = prepare_batch_input(settings, log, stop_event)
    first = plan[0]
    part = frame.loc[first["start"] - 1:first["end"] - 1].copy()
    report = report_for(part)
    save_csv(report[["data_row", "excel_row", "post_text_original"]], folder / "preview_cleaned_posts.csv")
    return report, folder, plan, len(frame)


class JobLock:
    """OS lock releases after a process crash; no stale-PID override required."""
    def __init__(self, path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = Path(self.path).open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if not self.handle.tell():
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("This batch job is already running in another window.") from exc
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class JobStore:
    def __init__(self, path, identity):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS row_progress (
                data_row INTEGER PRIMARY KEY, source_text TEXT NOT NULL,
                english TEXT, translation_status TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '', result_json TEXT);
            CREATE TABLE IF NOT EXISTS batches (
                number INTEGER PRIMARY KEY, first_row INTEGER NOT NULL,
                last_row INTEGER NOT NULL, filename TEXT NOT NULL,
                sha256 TEXT NOT NULL, row_count INTEGER NOT NULL);
        """)
        encoded = json.dumps(identity, sort_keys=True)
        old = self.get_meta("identity")
        if old is not None and old != encoded:
            self.db.close()
            raise ValueError("Saved progress belongs to a different input or calculation version.")
        self.set_meta("identity", encoded)

    def get_meta(self, key):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, value))
        self.db.commit()

    def rows(self, first, last):
        return {row["data_row"]: dict(row) for row in self.db.execute(
            "SELECT * FROM row_progress WHERE data_row BETWEEN ? AND ? ORDER BY data_row", (first, last))}

    def translated(self, data_row, source, english, status):
        self.db.execute("""INSERT INTO row_progress
            (data_row, source_text, english, translation_status, error)
            VALUES (?, ?, ?, ?, '') ON CONFLICT(data_row) DO UPDATE SET
            source_text=excluded.source_text, english=excluded.english,
            translation_status=excluded.translation_status, error=''""",
            (data_row, source, english, status))
        self.db.commit()  # Each whole-post translation is durable immediately.

    def failed(self, data_row, source, error):
        self.db.execute("""INSERT INTO row_progress
            (data_row, source_text, english, translation_status, error)
            VALUES (?, ?, NULL, 'pending', ?) ON CONFLICT(data_row)
            DO UPDATE SET error=excluded.error""", (data_row, source, error))
        self.db.commit()

    def skipped(self, data_row, source, error):
        self.db.execute("""INSERT INTO row_progress
            (data_row, source_text, english, translation_status, error)
            VALUES (?, ?, '', 'skipped', ?) ON CONFLICT(data_row) DO UPDATE SET
            english='', translation_status='skipped', error=excluded.error,
            result_json=NULL""", (data_row, source, error))
        self.db.commit()

    def results(self, frame):
        """Commit a completed original 16-row inference group atomically."""
        columns = frame.columns.tolist()
        previous = self.get_meta("output_columns")
        if previous is not None and json.loads(previous) != columns:
            raise ValueError("Output column schema changed; no CSV was combined.")
        missing = set(VALUE_COLUMNS) - set(columns)
        if missing:
            raise ValueError(f"Sentiment results missing columns: {sorted(missing)}")
        values = []
        for index, row in frame[VALUE_COLUMNS].iterrows():
            payload = {}
            for col, value in row.items():
                if pd.isna(value):
                    payload[col] = None
                elif hasattr(value, "item"):
                    payload[col] = value.item()
                else:
                    payload[col] = value
            # Python's float JSON round-trip keeps the original score precision.
            values.append((json.dumps(payload, ensure_ascii=False, allow_nan=False), int(index) + 1))
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('output_columns', ?)", (json.dumps(columns),))
            for payload, row_id in values:
                cursor = self.db.execute("UPDATE row_progress SET result_json=? WHERE data_row=? AND english IS NOT NULL", (payload, row_id))
                if cursor.rowcount != 1:
                    raise ValueError(f"Cannot checkpoint analysis before translation: data row {row_id}")

    def completed_count(self):
        return self.db.execute("SELECT COUNT(*) FROM row_progress WHERE result_json IS NOT NULL").fetchone()[0]

    def batch_record(self, number):
        row = self.db.execute("SELECT * FROM batches WHERE number=?", (number,)).fetchone()
        return dict(row) if row else None

    def finish_batch(self, batch, path):
        self.db.execute("INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?)",
                        (batch["number"], batch["start"], batch["end"], path.name,
                         digest_file(path), batch["end"] - batch["start"] + 1))
        self.db.commit()  # Only AFTER the complete CSV was fsynced and replaced.

    def close(self):
        self.db.close()


def restore_report(report, states):
    for index, row in report.iterrows():
        state = states.get(int(row["data_row"]))
        if state is None:
            continue
        if state["source_text"] != row["post_text_original"]:
            raise ValueError("A saved row no longer matches its original cleaned source.")
        if state["english"] is not None:
            if row["post_text_original"] and state["translation_status"] != "skipped":
                validate_translation(row["post_text_original"], state["english"])
            report.at[index, "post_text_original_english"] = state["english"]
            report.at[index, "translation_status"] = state["translation_status"]
        report.at[index, "translation_error"] = state["error"]
        if state["result_json"] is not None:
            report.at[index, "analysis_status"] = "skipped" if state["translation_status"] == "skipped" else "saved"


def translate_batch(frame, batch, settings, folder, store, cache, stop_event,
                    log, progress, browser_factory):
    report = report_for(frame)
    restore_report(report, store.rows(batch["start"], batch["end"]))
    translator = PreservingTranslator(settings, cache, stop_event, log, browser_factory)
    current = None
    consecutive_failures = 0
    status_file = folder / f"batch_{batch['number']:03d}_translation.csv"
    try:
        for count, (index, row) in enumerate(report.iterrows(), 1):
            current = index
            check_stop(stop_event)
            text = row["post_text_original"]
            if row["translation_status"] not in FINAL_STATUSES:
                if not text:
                    english, status = "", "empty"
                else:
                    try:
                        english, status = translator.translate(text)
                    except (InterruptedError, BrowserBlocked):
                        raise
                    except Exception as exc:
                        if not browser_error_is_transient(exc):
                            raise
                        error = f"{type(exc).__name__}: {exc}"
                        if settings.skip_failed_rows:
                            store.skipped(int(row["data_row"]), text, error)
                            report.at[index, "translation_status"] = "skipped"
                            report.at[index, "analysis_status"] = "skipped"
                        else:
                            store.failed(int(row["data_row"]), text, error)
                            report.at[index, "translation_status"] = "retry_pending"
                        report.at[index, "translation_error"] = error
                        translator.close()
                        consecutive_failures = 0 if settings.skip_failed_rows and untranslated_output_failure(exc) else consecutive_failures + 1
                        save_csv(report, status_file)
                        log(f"Data row {int(row['data_row'])}: {'skipped after all attempts (including Detect language)' if settings.skip_failed_rows else 'saved for later retry'}; continuing with other rows.")
                        if progress:
                            progress("translation", count, len(report), report.loc[index].to_dict())
                        if consecutive_failures >= settings.max_consecutive_failures:
                            raise TranslationUnavailable(
                                f"{consecutive_failures} consecutive texts failed on all routes. Possible internet/service outage; saved progress can be resumed.") from exc
                        continue
                consecutive_failures = 0
                data_row = int(row["data_row"])
                store.translated(data_row, text, english, status)
                report.at[index, "post_text_original_english"] = english
                report.at[index, "translation_status"] = status
                report.at[index, "translation_error"] = ""
            if count % 100 == 0:
                save_csv(report, status_file)
            if progress:
                progress("translation", count, len(report), report.loc[index].to_dict())
    except BaseException as exc:
        if current is not None and report.at[current, "translation_status"] not in FINAL_STATUSES:
            row = report.loc[current]
            error = f"{type(exc).__name__}: {exc}"
            store.failed(int(row["data_row"]), row["post_text_original"], error)
            report.at[current, "translation_error"] = error
            report.at[current, "translation_status"] = "interrupted" if isinstance(exc, InterruptedError) else "failed"
        raise
    finally:
        try:
            save_csv(report, status_file)
        finally:
            translator.close()
    frame["post_text_original_english"] = report["post_text_original_english"].tolist()
    return frame, report


def valid_batch(record, batch, folder):
    if record is None:
        return False
    path = folder / record["filename"]
    return (record["first_row"] == batch["start"] and record["last_row"] == batch["end"]
            and record["row_count"] == batch["end"] - batch["start"] + 1
            and path.is_file() and digest_file(path) == record["sha256"])


def merge_batches(plan, folder, store, stop_event=None):
    """Copy CSV fields as text, in numeric batch order, without pandas coercion."""
    output = folder / f"FULL_rows_{plan[0]['start']}_{plan[-1]['end']}.csv"
    columns = json.loads(store.get_meta("output_columns") or "null")
    if not columns:
        raise ValueError("No complete output schema is available for merging.")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    def write(path):
        with path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(columns)
            for batch in plan:
                check_stop(stop_event)
                record = store.batch_record(batch["number"])
                if not valid_batch(record, batch, folder):
                    raise ValueError(f"Batch {batch['number']} is incomplete or its CSV changed. Resume to rebuild it.")
                source = folder / record["filename"]
                count = 0
                with source.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    if next(reader, None) != columns:
                        raise ValueError(f"Batch {batch['number']} CSV columns do not match.")
                    for row in reader:
                        check_stop(stop_event)
                        if len(row) != len(columns):
                            raise ValueError(f"Batch {batch['number']} has an invalid CSV row.")
                        writer.writerow(row)
                        count += 1
                if count != record["row_count"] or digest_file(source) != record["sha256"]:
                    raise ValueError(f"Batch {batch['number']} changed during consolidation.")
    atomic_write(output, write)
    store.set_meta("full_csv_sha256", digest_file(output))
    store.set_meta("status", "complete")
    return output


def save_pending_snapshot(full_frame, folder, store):
    """Durable diagnostics; pending rows are never represented as complete."""
    first, last = int(full_frame.index[0]) + 1, int(full_frame.index[-1]) + 1
    states = store.rows(first, last)
    report = report_for(full_frame)
    restore_report(report, states)
    pending = report.loc[~report["translation_status"].isin(FINAL_STATUSES)].copy()
    save_csv(pending, folder / "pending_rows.csv")
    save_csv(report.loc[report["translation_status"].eq("skipped")], folder / "skipped_rows.csv")
    return report, states, len(pending)


def partial_result_frame(frame, report, states):
    result = frame.copy()
    payloads = []
    for index in result.index:
        state = states.get(int(index) + 1, {})
        payload = json.loads(state["result_json"]) if state.get("result_json") is not None else {}
        payload["post_text_original_english"] = state.get("english")
        payloads.append(payload)
    for column in VALUE_COLUMNS:
        result[column] = [payload.get(column) for payload in payloads]
    # Complete outputs retain the original schema. Partial diagnostics explicitly
    # identify missing rows and groups whose sentiment has not run.
    result["data_row"] = report["data_row"]
    result["translation_status"] = report["translation_status"]
    result["analysis_status"] = report["analysis_status"]
    result["translation_error"] = report["translation_error"]
    return result


def save_partial_snapshot(full_frame, folder, store):
    report, states, remaining = save_pending_snapshot(full_frame, folder, store)
    result = partial_result_frame(full_frame, report, states)
    path = folder / f"PARTIAL_rows_{int(result.index[0]) + 1}_{int(result.index[-1]) + 1}.csv"
    save_csv(result, path)
    return path, remaining


def run_batch_pipeline(settings, stop_event=None, log=print, progress=None,
                       browser_factory=GoogleTranslateBrowser,
                       sentiment_function=analysis_logic.add_original_sentiment_columns):
    full_frame, plan, folder, identity = prepare_batch_input(settings, log, stop_event)
    total = len(full_frame)
    if progress:
        progress("plan", 0, total, {"folder": str(folder), "batches": len(plan), "last_row": plan[-1]["end"]})
    with JobLock(folder / "run.lock"):
        store = JobStore(folder / "checkpoint.sqlite3", identity)
        cache = None
        try:
            cache = TranslationCache(Path(settings.output_dir) / "translation_cache.sqlite3", settings.source_language)
            completed = store.completed_count()
            log(f"Resume: {completed:,}/{total:,} rows already have saved sentiment results.")
            if progress:
                progress("overall", completed, total, None)
            # First visit all batches. Later passes visit incomplete batches only;
            # the row store/cache makes each such visit retry pending text only.
            for repair_pass in range(settings.pending_retry_passes + 1):
                if repair_pass:
                    log(f"Pending-row retry pass {repair_pass}/{settings.pending_retry_passes}. Completed rows and groups will be reused.")
                    if progress:
                        progress("repair", repair_pass, settings.pending_retry_passes, None)
                for batch in plan:
                    check_stop(stop_event)
                    record = store.batch_record(batch["number"])
                    if valid_batch(record, batch, folder):
                        if not repair_pass:
                            log(f"Batch {batch['number']}/{len(plan)} already complete; skipping {record['filename']}.")
                        continue
                    part = full_frame.loc[batch["start"] - 1:batch["end"] - 1].copy()
                    log(f"Batch {batch['number']}/{len(plan)}: data rows {batch['start']:,}–{batch['end']:,}.")
                    if progress:
                        progress("batch", batch["number"], len(plan), {"start": batch["start"], "end": batch["end"], "folder": str(folder)})
                        progress("prepared", 0, len(part), report_for(part).head(50).to_dict("records"))
                    # A slightly longer request gap on a repair pass helps after
                    # ordinary transient errors. No retry route bypasses a block.
                    timed = replace(settings, interval=settings.interval * (2**repair_pass))
                    part, report = translate_batch(part, batch, timed, folder, store, cache, stop_event, log, progress, browser_factory)
                    states = store.rows(batch["start"], batch["end"])
                    groups = (len(part) + INFERENCE_ROWS - 1) // INFERENCE_ROWS
                    for offset in range(0, len(part), INFERENCE_ROWS):
                        check_stop(stop_event)
                        chunk = part.iloc[offset:offset + INFERENCE_ROWS].copy()
                        group_states = [states.get(int(index) + 1, {}) for index in chunk.index]
                        done = [state.get("result_json") is not None for state in group_states]
                        if all(done):
                            continue
                        if any(done):
                            raise ValueError("A saved 16-row analysis checkpoint is inconsistent. No rows were overwritten.")
                        if any(state.get("english") is None for state in group_states):
                            # Keep the original 16-row grouping. Do not analyse
                            # untranslated rows or regroup neighbours around them.
                            continue
                        if progress:
                            progress("model", 0, 1, None)
                        result = sentiment_function(chunk, log, stop_event, None)
                        if result.index.tolist() != chunk.index.tolist() or len(result) != len(chunk):
                            raise ValueError("Sentiment output row order changed. No results were checkpointed.")
                        skipped_indexes = [index for index, state in zip(chunk.index, group_states)
                                           if state.get("translation_status") == "skipped"]
                        for column in VALUE_COLUMNS:
                            if column != "post_text_original_english":
                                result.loc[skipped_indexes, column] = None
                        store.results(result)
                        completed += len(chunk)
                        if progress:
                            progress("sentiment", offset // INFERENCE_ROWS + 1, groups, None)
                            progress("overall", completed, total, None)
                    states = store.rows(batch["start"], batch["end"])
                    for index in report.index:
                        if states.get(int(index) + 1, {}).get("result_json") is not None:
                            report.at[index, "analysis_status"] = "skipped" if states[int(index) + 1]["translation_status"] == "skipped" else "saved"
                    save_csv(report, folder / f"batch_{batch['number']:03d}_translation.csv")
                    if any(states.get(int(index) + 1, {}).get("result_json") is None for index in part.index):
                        log(f"Batch {batch['number']} has pending rows. Other batches will continue; this batch will be retried later.")
                        save_csv(report.loc[~report["translation_status"].isin(FINAL_STATUSES)],
                                 folder / f"batch_{batch['number']:03d}_pending.csv")
                        save_csv(partial_result_frame(part, report, states),
                                 folder / f"batch_{batch['number']:03d}_partial.csv")
                        continue
                    payloads = [json.loads(states[int(index) + 1]["result_json"]) for index in part.index]
                    for column in VALUE_COLUMNS:
                        part[column] = [payload[column] for payload in payloads]
                    columns = json.loads(store.get_meta("output_columns"))
                    path = folder / f"batch_{batch['number']:03d}_rows_{batch['start']}_{batch['end']}.csv"
                    save_csv(part[columns], path)
                    store.finish_batch(batch, path)
                    # Clear the diagnostic queue for this batch, preserving headers.
                    save_csv(report.iloc[:0], folder / f"batch_{batch['number']:03d}_pending.csv")
                    (folder / f"batch_{batch['number']:03d}_partial.csv").unlink(missing_ok=True)
                    log(f"Batch {batch['number']}/{len(plan)} saved: {path.name}")
                    if progress:
                        progress("batch_saved", batch["number"], len(plan), str(path))
                _, _, remaining = save_pending_snapshot(full_frame, folder, store)
                if all(valid_batch(store.batch_record(b["number"]), b, folder) for b in plan):
                    break
                log(f"Pending translations after this pass: {remaining:,}.")
                if repair_pass < settings.pending_retry_passes:
                    # Bounded, cancellable backoff; never an endless retry loop.
                    delay = min(30, 5 * (repair_pass + 1))
                    if stop_event is not None:
                        if stop_event.wait(delay):
                            check_stop(stop_event)
                    else:
                        import time
                        time.sleep(delay)
            check_stop(stop_event)
            if not all(valid_batch(store.batch_record(b["number"]), b, folder) for b in plan):
                output, remaining = save_partial_snapshot(full_frame, folder, store)
                store.set_meta("status", "pending_translations")
                raise PendingTranslations(
                    f"Processed the entire planned range; {remaining:,} rows still need translation. "
                    f"Saved available work: {output}. Pending rows: {folder / 'pending_rows.csv'}. "
                    "Click Start / Resume with the same settings to retry only unfinished work. FULL CSV will be produced when all rows finish.")
            log("All planned batches are complete. Combining verified CSVs in row order...")
            output = merge_batches(plan, folder, store, stop_event)
            skipped_count = store.db.execute("SELECT COUNT(*) FROM row_progress WHERE translation_status='skipped'").fetchone()[0]
            if skipped_count:
                store.set_meta("status", "complete_with_skips")
                log(f"Finished with {skipped_count:,} skipped translations. Their English/sentiment fields are blank; see skipped_rows.csv.")
                if progress:
                    progress("skipped", skipped_count, total, str(folder / "skipped_rows.csv"))
            # Do not leave a stale incomplete result looking current after recovery.
            partial = folder / f"PARTIAL_rows_{plan[0]['start']}_{plan[-1]['end']}.csv"
            if partial.exists():
                partial.unlink()
            log(f"Full CSV saved: {output}")
            if progress:
                progress("overall", total, total, None)
            return output
        except (BrowserBlocked, TranslationUnavailable, InterruptedError):
            # Also save a readable partial view on a deliberate pause. SQLite
            # already contains each success, including the most recent row.
            save_partial_snapshot(full_frame, folder, store)
            raise
        finally:
            if cache is not None:
                cache.close()
            store.close()
