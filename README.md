# IndicBERT Google Translate UI & Fault-Tolerant Batch Pipeline

A desktop GUI application and fault-tolerant NLP pipeline for cleaning multilingual social media datasets, translating non-English posts into English via an automated 4-stage Selenium Chrome recovery workflow, and computing multi-model sentiment scores using **IndicBERTv2**, **TextBlob**, and **VADER**.

---

## Key Features

* **Baseline-Identical Cleaning & Sentiment Logic:** Preserves exact byte-level compatibility with the original `analysis_logic.py` and `original_indicbert.py` calculations, including Unicode-aware hashtag removal (supporting Indic vowel signs, virama, and zero-width joiners), underscore word removal, and repeated punctuation normalization.
* **Mixed-Text English Preservation (`preserve-english-v2`):** Automatically segments mixed-language posts so existing English words, numbers, and surrounding punctuation remain byte-exact in their original order while only non-Latin script spans are translated[cite: 1].
* **Smart Script Hinting & 4-Stage Browser Recovery:**
  1. **Attempt 1 (Incognito + Script Hint):** Inspects Unicode ranges when `Source language` is set to `auto`, applying `sl=hi` for Devanagari and `sl=bn` for Bengali in a Chrome Incognito session to prevent Google Translate from misdetecting Indic scripts as English[cite: 1].
  2. **Attempt 2 (Fresh Incognito + Auto-Detect):** If the first attempt fails (e.g., Sanskrit text written in Devanagari script), opens a fresh Chrome Incognito session using `Detect language` (`sl=auto` $\rightarrow$ `tl=en`) on the exact same cleaned text[cite: 1].
  3. **Attempt 3 (Fresh Normal Chrome + Auto-Detect):** Launches an isolated temporary normal Chrome profile with `Detect language` $\rightarrow$ `English`[cite: 1].
  4. **Attempt 4 (Google Search Translator Widget):** Queries Google Search in a fresh normal Chrome session and extracts verified translations strictly from the embedded translation widget (`#tw-source-text-ta` / `#tw-target-text`), ignoring general search results or AI Overviews[cite: 1].
* **Durable SQLite Checkpointing & Two-Tier Caching:** Every translated post and every completed 16-row inference group is immediately committed to `checkpoint.sqlite3` (WAL mode), while `translation_cache.sqlite3` caches both segment-level and post-level translations across runs[cite: 1].
* **Graceful Skip & Outage Protection:** With `Skip untranslated rows after all attempts` enabled by default, rows that fail all four browser routes are logged to `skipped_rows.csv` with blank English and sentiment fields so batch processing continues uninterrupted[cite: 1]. If Google displays a CAPTCHA or rate-limit page (`/sorry/`), the pipeline immediately saves progress and pauses without attempting session-switching bypasses[cite: 1].
* **Multi-Model Sentiment Suite:**
  * **IndicBERTv2 (`ai4bharat/IndicBERTv2-alpha-SentimentClassification`):** Evaluates cleaned original text (`post_text_original`) in strict 16-row batches (`max_length=256`) to produce `indic_neg`, `indic_pos`, `indic_compound`, and `indic_label`[cite: 1].
  * **TextBlob & VADER:** Evaluates translated English text (`post_text_original_english`) to produce `textblob_polarity`, `vader_neg`, `vader_neu`, `vader_pos`, and `vader_compound`[cite: 1].

---

## Project Structure

| File | Description |
| :--- | :--- |
| `indicbert_browser_ui.py` | Tkinter desktop GUI featuring batch/range configuration, dual progress bars, live 50-row preview table, and persistent user preferences (`ui_settings.json`)[cite: 1]. |
| `browser_pipeline.py` | Core translation engine implementing Unicode script segmentation, 4-stage Selenium Chrome recovery, DOM freshness checks, and SQLite translation caching[cite: 1]. |
| `batch_runner.py` | Batch orchestrator handling OS file locking (`JobLock`), SQLite progress store (`JobStore`), pending/skip queues, 16-row inference grouping, and atomic CSV merging[cite: 1]. |
| `analysis_logic.py` | Original text preprocessing functions and IndicBERTv2, TextBlob, and VADER sentiment analysis pipeline[cite: 1]. |
| `original_indicbert.py` | Baseline standalone script (`deep_translator` + IndicBERTv2) retained for reference and byte-verification[cite: 1]. |
| `tests_recovery.py` | Offline regression test suite (48 tests) covering browser fallback routes, Sanskrit/Hindi detection, cache validation, and crash recovery[cite: 1]. |
| `setup_windows.bat` | Windows setup script that creates a `.venv` virtual environment and installs all packages from `requirements.txt`[cite: 1]. |
| `run_windows.bat` | Windows launcher script that starts `indicbert_browser_ui.py` using the local `.venv` interpreter[cite: 1]. |
| `requirements.txt` | Pinned Python dependencies (`pandas`, `torch`, `transformers`, `selenium`, `textblob`, `vaderSentiment`, etc.)[cite: 1]. |
| `VERIFY_RESULTS.md` | Verification summary and SHA-256 baseline checksum documentation[cite: 1]. |
| `README_BN.md` | Bengali documentation and operational guide[cite: 1]. |

---

## Getting Started (Windows)

### Prerequisites
* **Python 3.11 or 3.12** installed with **Add Python to PATH** enabled[cite: 1].
* **Google Chrome** installed[cite: 1].

### Fresh Installation
1. Extract the project ZIP archive to a local folder[cite: 1].
2. Double-click `setup_windows.bat` to create the `.venv` virtual environment and install all required dependencies[cite: 1].
3. Once setup finishes, double-click `run_windows.bat` to launch the desktop UI[cite: 1].

---

## Usage Guide

1. **Select Input & Output:** Browse for your input Excel workbook (`.xlsx` or `.xls` containing required `post_text` and `datetime` columns) and choose an output directory[cite: 1].
2. **Configure Execution Mode:**
   * **Batch Mode (Default ON):** Processes records starting from `Start row` using `Rows per batch` (default `1000`) and `Max batches` (default `100`, covering up to `100,000` data rows)[cite: 1]. Check **Full dataset (ignore max)** to process all remaining rows to the end of the file[cite: 1].
   * **Range Mode (Batch Mode OFF):** Processes a single bounded slice from `Start row` to `End row` (inclusive, where Data row 1 corresponds to Excel row 2 excluding the header) and exports a single `.xlsx` workbook[cite: 1].
3. **Configure Translation Settings:**
   * **Source language:** Keep set to `auto` for automatic Devanagari (`hi`) and Bengali (`bn`) script hinting with `Detect language` fallback, or select an explicit language code (`hi`, `bn`, `en`)[cite: 1].
   * **Request gap:** Default is `2` seconds between live browser submissions; cached, duplicate, empty, and English-only posts bypass Google Translate completely[cite: 1].
   * **Skip untranslated rows after all attempts:** Enabled by default (`ON`) to record persistently untranslatable rows in `skipped_rows.csv` with blank scores rather than blocking batch completion[cite: 1].
4. **Preview & Run:**
   * Click **Preview cleaned text** to inspect the first 50 cleaned records and verify row bounds[cite: 1].
   * Click **Start / Resume** to begin or resume translation and sentiment analysis[cite: 1].

---

## Output Files & Checkpoints

### Output Directory Artifacts

| File | Purpose |
| :--- | :--- |
| `batch_001_rows_1_1000.csv` | Final verified CSV for a completed batch containing all original columns and sentiment scores[cite: 1]. |
| `batch_001_translation.csv` | Per-row diagnostic log tracking `translation_status`, `analysis_status`, and `translation_error` for the batch[cite: 1]. |
| `batch_001_pending.csv` | Queue of unresolved rows within the current batch (cleared upon batch completion)[cite: 1]. |
| `batch_001_partial.csv` | Snapshot of an incomplete batch containing available results with missing sentiment fields left blank[cite: 1]. |
| `pending_rows.csv` | Consolidated queue of unresolved translations across the entire planned job[cite: 1]. |
| `skipped_rows.csv` | Audit log of skipped rows listing `data_row`, `excel_row`, `post_text_original`, status, and failure reason[cite: 1]. |
| `PARTIAL_rows_1_100000.csv` | Full-range diagnostic snapshot saved when a job pauses or finishes with pending rows (automatically removed once `FULL_...csv` is built)[cite: 1]. |
| `FULL_rows_1_100000.csv` | Final consolidated dataset merged in exact row order across all batches (skipped rows have blank English and sentiment columns)[cite: 1]. |
| `checkpoint.sqlite3` | Job-specific SQLite database storing durable per-row translations, 16-row inference group results, and batch SHA-256 records[cite: 1]. |
| `translation_cache.sqlite3` | Shared SQLite cache in the output root storing verified segment and full-post translations for instant reuse[cite: 1]. |

### Generated Sentiment & Metadata Columns

* **Preprocessing Columns:** `month_year`, `post_text_original`, `post_text_original_english`[cite: 1]
* **IndicBERTv2 Columns (on `post_text_original`):** `post_text_original_indic_neg`, `post_text_original_indic_pos`, `post_text_original_indic_compound`, `post_text_original_indic_label`[cite: 1]
* **TextBlob Column (on `post_text_original_english`):** `post_text_original_english_textblob_polarity`[cite: 1]
* **VADER Columns (on `post_text_original_english`):** `post_text_original_english_vader_neg`, `post_text_original_english_vader_neu`, `post_text_original_english_vader_pos`, `post_text_original_english_vader_compound`[cite: 1]

---

## Running Verification Tests

To execute the 48 offline regression tests without downloading model weights or opening a live browser, run[cite: 1]:

```bat
.venv\Scripts\python.exe -m unittest -v tests_recovery