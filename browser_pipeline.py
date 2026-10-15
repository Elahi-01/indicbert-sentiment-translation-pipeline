"""Original cleaning/sentiment with verified browser translation and recovery."""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
import hashlib
import os
import re
import sqlite3
import tempfile
import time
import unicodedata

import pandas as pd
from analysis_logic import (
    clean_text, remove_hashtags, remove_underscore_words,
    remove_repeated_punctuation, add_original_sentiment_columns,
)

SOURCE_SELECTORS = ('textarea[aria-label="Source text"]', 'textarea')
OUTPUT_SELECTORS = (
    'span.HwtZe[lang="en"] span.ryNqvb',
    'span[lang="en"] span.ryNqvb',
)
MODEL_NAME = "ai4bharat/IndicBERTv2-alpha-SentimentClassification"
FOREIGN_SCRIPT = re.compile(r"[\u0900-\u0DFF\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF]")
TRANSLATION_POLICY = "preserve-english-v2"
RAW_CACHE_ENGINE = "chrome-google-v1"
PRESERVED_CACHE_ENGINE = "chrome-google-preserve-english-v2"


class TranslationFailure(RuntimeError):
    pass


class BrowserBlocked(TranslationFailure):
    pass


class TranslationUnavailable(TranslationFailure):
    """Several consecutive failures suggest a service/network-wide outage."""


class PendingTranslations(TranslationFailure):
    """All available work was processed, but some rows still need translation."""


def browser_error_is_transient(exc):
    if type(exc).__name__ in ("NoSuchDriverException", "SessionNotCreatedException"):
        return False  # Chrome/driver installation must be repaired, not retried per row.
    return isinstance(exc, (TranslationFailure, TimeoutError, ConnectionError)) or type(exc).__module__.startswith(("selenium.", "urllib3.", "http.client"))


def untranslated_output_failure(exc):
    # A returned foreign-script output is a text/language failure. It does not
    # imply a service outage just because several difficult posts occur in a row.
    return "The non-English input was not converted to English." in str(exc)


LANGUAGE_NAMES = {"hi": "Hindi", "bn": "Bengali", "auto": "Detect language"}
SUCCESS_STATUSES = ("translated", "cached", "empty", "unchanged_english")
FINAL_STATUSES = SUCCESS_STATUSES + ("skipped",)


def effective_source_language(text, configured="auto"):
    if configured != "auto":
        return configured
    # The user's dataset contains Hindi/Bengali. Avoid Google's demonstrated
    # English mis-detection. Other scripts still use Google's language detector.
    counts = {"hi": 0, "bn": 0}
    for char in text:
        if "\u0900" <= char <= "\u097f" or "\ua8e0" <= char <= "\ua8ff":
            counts["hi"] += 1
        elif "\u0980" <= char <= "\u09ff":
            counts["bn"] += 1
    language = max(counts, key=counts.get)
    return language if counts[language] else "auto"


@dataclass(frozen=True)
class RunSettings:
    input_file: Path
    output_dir: Path
    start_row: int = 1
    end_row: int = 10
    source_language: str = "auto"
    interval: float = 2.0
    timeout: float = 60.0
    max_part_chars: int = 4500
    chrome_binary: str = ""
    chromedriver_path: str = ""
    browser_restart_every: int = 250
    pending_retry_passes: int = 2
    max_consecutive_failures: int = 5
    skip_failed_rows: bool = True

    def validate(self):
        if not Path(self.input_file).is_file():
            raise ValueError("Choose an existing input Excel file.")
        if self.start_row < 1 or self.end_row < self.start_row:
            raise ValueError("Start row must be >= 1 and End row must be >= Start row.")
        if self.interval < 0 or self.timeout <= 0 or self.max_part_chars < 1:
            raise ValueError("Translation timing and part length must be valid.")
        if not re.fullmatch(r"auto|[a-z]{2,3}(?:-[A-Za-z]+)?", self.source_language):
            raise ValueError("Source language must be auto or a language code such as hi.")
        if self.browser_restart_every < 1 or self.pending_retry_passes < 0 or self.max_consecutive_failures < 1:
            raise ValueError("Browser restart and failure limits must be positive; retry passes cannot be negative.")


def check_stop(event):
    if event is not None and event.is_set():
        raise InterruptedError("Stopped by the user. Saved translations can be resumed.")


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.stem + "_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name))
        # Windows _commit/FlushFileBuffers requires write access. Opening this
        # handle as read-only works on Unix but can raise EBADF on Windows.
        with Path(name).open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        Path(name).unlink(missing_ok=True)


def save_csv(frame, path):
    atomic_write(path, lambda p: frame.to_csv(p, index=False, encoding="utf-8-sig"))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_frame(settings, log=print, stop_event=None):
    settings.validate()
    check_stop(stop_event)
    log(f"Reading Excel. Selected data rows {settings.start_row:,}–{settings.end_row:,} (inclusive).")
    before = digest_file(settings.input_file)
    # Same pandas defaults as the original. Do not reinterpret NA/empty values.
    original = pd.read_excel(settings.input_file)
    if before != digest_file(settings.input_file):
        raise ValueError("The input file changed during reading. Close Excel and run again.")
    check_stop(stop_event)
    missing = {"post_text", "datetime"} - set(original.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if settings.end_row > len(original):
        raise ValueError(f"End row is {settings.end_row:,}, but the dataset has {len(original):,} data rows.")
    # Data row 1 = iloc[0]. End row is inclusive in the UI.
    frame = original.iloc[settings.start_row - 1:settings.end_row].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["month_year"] = frame["datetime"].dt.to_period("M")
    frame["post_text_original"] = frame["post_text"].apply(clean_text)
    frame["post_text_original"] = frame["post_text_original"].apply(remove_hashtags)
    frame["post_text_original"] = frame["post_text_original"].apply(remove_underscore_words)
    frame["post_text_original"] = frame["post_text_original"].apply(remove_repeated_punctuation)
    report = pd.DataFrame({
        "data_row": frame.index + 1,
        "excel_row": frame.index + 2,
        "post_text_raw": frame["post_text"].apply(clean_text),
        "post_text_original": frame["post_text_original"],
    })
    report["post_text_original_english"] = ""
    report["translation_status"] = "pending"
    report["translation_error"] = ""
    folder = Path(settings.output_dir) / f"rows_{settings.start_row}_{settings.end_row}_{before[:12]}_{TRANSLATION_POLICY}"
    folder.mkdir(parents=True, exist_ok=True)
    save_csv(report[["data_row", "excel_row", "post_text_raw", "post_text_original"]], folder / "cleaned_posts.csv")
    log(f"Cleaned {len(frame):,} rows with the original cleaning rules (Unicode-aware hashtags).")
    return frame, report, folder


def split_post(text, maximum=4500):
    if maximum < 1:
        raise ValueError("Part length must be positive.")
    parts = []
    remaining = text
    while len(remaining) > maximum:
        cut = remaining.rfind(" ", 0, maximum + 1)
        if cut < 1:
            cut = maximum
        parts.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def script_flags(text):
    """Identify non-Latin letters and their marks without changing the text."""
    foreign, latin = [], []
    previous_foreign = False
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L"):
            # NFKD is used ONLY for classification, never for output. This also
            # preserves accented/styled Latin characters in English spans.
            decomposed = unicodedata.normalize("NFKD", character)
            is_latin = "LATIN" in unicodedata.name(character, "") or any(
                "A" <= c <= "Z" or "a" <= c <= "z" for c in decomposed
            )
            previous_foreign = not is_latin
            foreign.append(not is_latin)
            latin.append(is_latin)
        elif category.startswith("M"):
            is_foreign = previous_foreign or bool(FOREIGN_SCRIPT.fullmatch(character))
            foreign.append(is_foreign)
            latin.append(False)
        else:
            previous_foreign = False
            foreign.append(False)
            latin.append(False)
    return foreign, latin


def translation_segments(text):
    """Return (translate, text) pieces; literal English pieces stay byte-exact.

    Adjacent Hindi words retain their context, including intervening punctuation
    and numbers. Latin letters form boundaries. Pure non-Latin posts retain the
    previous whole-post translation path and can reuse old successful caches.
    """
    foreign, latin = script_flags(text)
    if not any(foreign):
        return [(False, text)]
    if not any(latin):
        return [(True, text)]
    segments = []
    cursor, start, end = 0, None, None

    def finish_span():
        nonlocal cursor, start, end
        if start is None:
            return
        if cursor < start:
            segments.append((False, text[cursor:start]))
        segments.append((True, text[start:end]))
        cursor = end
        start, end = None, None

    for index, (is_foreign, is_latin) in enumerate(zip(foreign, latin)):
        if is_latin:
            finish_span()
        elif is_foreign:
            if start is None:
                start = index
            end = index + 1
        elif (start is not None and unicodedata.category(text[index]).startswith("P")
              and FOREIGN_SCRIPT.fullmatch(text[index])):
            # Danda and other script-specific punctuation belong to the Hindi
            # span. Ordinary English punctuation outside it remains literal.
            end = index + 1
    finish_span()
    if cursor < len(text):
        segments.append((False, text[cursor:]))
    return segments


def validate_preserved_literals(source, result):
    segments = translation_segments(source)
    if not any(translate for translate, _ in segments):
        if result != source:
            raise TranslationFailure("An English-only post was changed.")
        return result
    cursor = 0
    for number, (translate, literal) in enumerate(segments):
        if translate or not literal:
            continue
        position = result.find(literal, cursor)
        if position < 0 or (number == 0 and position != 0):
            raise TranslationFailure("An original English span was changed or removed.")
        cursor = position + len(literal)
        if number == len(segments) - 1 and not result.endswith(literal):
            raise TranslationFailure("The original English suffix was changed.")
    return result


def validate_translation(source, result):
    result = str(result or "").strip()
    if not result:
        raise TranslationFailure("Google returned an empty result.")
    if any(script_flags(source)[0]):
        same = re.sub(r"\s+", " ", result) == re.sub(r"\s+", " ", source).strip()
        if same or not any(script_flags(result)[1]) or any(script_flags(result)[0]):
            raise TranslationFailure("The non-English input was not converted to English.")
    return result


class TranslationCache:
    def __init__(self, path, language):
        self.language = language
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS translations (
            engine TEXT NOT NULL, source_language TEXT NOT NULL,
            target_language TEXT NOT NULL, source_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            PRIMARY KEY(engine, source_language, target_language, source_text))""")
        self.connection.commit()

    def get(self, text, engine=RAW_CACHE_ENGINE):
        row = self.connection.execute(
            "SELECT translated_text FROM translations WHERE engine=? AND source_language=? AND target_language=? AND source_text=?",
            (engine, self.language, "en", text),
        ).fetchone()
        if row:
            try:
                if engine == PRESERVED_CACHE_ENGINE:
                    validate_translation(text, row[0])
                    result = validate_preserved_literals(text, row[0])
                else:
                    result = validate_translation(text, row[0])
                return result
            except TranslationFailure:
                return None
        return None

    def put(self, text, translated, engine=RAW_CACHE_ENGINE):
        if engine == PRESERVED_CACHE_ENGINE:
            validate_translation(text, translated)
            validate_preserved_literals(text, translated)
        else:
            translated = validate_translation(text, translated)
        self.connection.execute("INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?)",
                                (engine, self.language, "en", text, translated))
        self.connection.commit()

    def close(self):
        self.connection.close()


class GoogleTranslateBrowser:
    """Visible Chrome; ordinary failures recover using fresh sessions.

    The same input is verified at every attempt. Verification/rate-limit pages
    never trigger session switching. Normal mode uses a Selenium-owned profile,
    not the user's existing logged-in Chrome profile.
    """
    def __init__(self, settings, stop_event, log):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        self.settings, self.stop_event, self.log = settings, stop_event, log
        self.By, self.Keys = By, Keys
        self.driver = None
        self.last_submission = None
        self.route = "incognito"
        self.active_language = None
        self.session_successes = 0
        self.language_override = None
        # Create lazily inside translate(), so launch/navigation failures are
        # covered by the same recovery plan as a stale page.

    def _start(self, route):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        self.close()
        self.route = route
        options = webdriver.ChromeOptions()
        if route == "incognito":
            options.add_argument("--incognito")
        options.add_argument("--lang=en-US")
        options.add_argument("--window-size=1280,960")
        options.page_load_strategy = "eager"
        options.add_experimental_option("prefs", {"intl.accept_languages": "en-US,en"})
        if self.settings.chrome_binary:
            options.binary_location = self.settings.chrome_binary
        service = Service(executable_path=self.settings.chromedriver_path) if self.settings.chromedriver_path else Service()
        try:
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(min(self.settings.timeout, 30))
            self.driver.set_script_timeout(min(self.settings.timeout, 15))
            # Bound the HTTP command timeout as well as page/DOM waits, so a
            # frozen Chrome connection cannot block indefinitely.
            executor = self.driver.command_executor
            client_config = getattr(executor, "_client_config", None)
            if client_config is not None:
                client_config.timeout = min(self.settings.timeout, 30)
            self.active_language = None
            self.session_successes = 0
            self.log(f"Opened fresh Chrome: {route}.")
        except BaseException:
            self.close()
            raise

    def _navigate(self, url):
        from selenium.common.exceptions import TimeoutException
        try:
            self.driver.get(url)
        except TimeoutException:
            # The DOM can already be usable while an unrelated asset is slow.
            self.driver.execute_script("window.stop();")
        check_stop(self.stop_event)
        if "consent.google.com" in self.driver.current_url:
            buttons = self.driver.find_elements(self.By.XPATH, "//button[normalize-space()='Reject all']")
            if buttons:
                buttons[0].click()
            else:
                self.log("Close Google's cookie notice in the app's Chrome window.")
        self._blocked()

    def _open_language(self, language):
        if self.active_language == language:
            return
        # A fresh document is also an output freshness barrier: the previous
        # request's asynchronous result cannot be read from this document.
        self._navigate("about:blank")
        if self.route == "search":
            name = LANGUAGE_NAMES.get(language, language)
            query = "Google Translate to English" if language == "auto" else f"{name} to English translate"
            url = "https://www.google.com/search?" + urlencode({"q": query, "hl": "en"})
        else:
            url = "https://translate.google.com/?" + urlencode({
                "sl": language, "tl": "en", "hl": "en", "op": "translate"})
        self._navigate(url)
        self._wait(self._source, "Google's translator input was not found (page/cookie notice or widget changed).")
        if self.route == "search":
            self._wait(lambda: self._search_languages_match(language),
                       "Google Search's translator languages could not be verified.")
        self.active_language = language

    def _search_languages_match(self, language):
        # Only the translation widget is used; never a search result, AI
        # overview, transliteration, or unrelated English text on the page.
        target = self.driver.find_elements(self.By.CSS_SELECTOR, "#tw-tl")
        source = self.driver.find_elements(self.By.CSS_SELECTOR, "#tw-sl")
        if not target or not target[0].is_displayed() or target[0].text.strip().lower() != "english":
            return False
        if not source or not source[0].is_displayed():
            return False
        label = source[0].text.lower()
        expected = LANGUAGE_NAMES.get(language, language).lower()
        return language == "auto" or expected in label or (language == "bn" and "bangla" in label)

    def _pause(self, seconds):
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                check_stop(self.stop_event)
        else:
            time.sleep(seconds)

    def _blocked(self):
        check_stop(self.stop_event)
        if self.driver is None:
            return
        from selenium.common.exceptions import NoSuchElementException
        try:
            body = self.driver.find_element(self.By.TAG_NAME, "body").text.lower()
        except NoSuchElementException:
            body = ""
        if "/sorry/" in self.driver.current_url or any(s in body for s in (
            "unusual traffic from your computer network", "verify you are human",
            "our systems have detected unusual traffic", "too many requests",
        )):
            raise BrowserBlocked("Google displayed verification/rate limiting. Progress saved; wait and use Start / Resume later. No session switching was attempted to bypass it.")

    def _wait(self, predicate, message, timeout=None):
        from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
        deadline = time.monotonic() + (self.settings.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            self._blocked()
            try:
                result = predicate()
                if result:
                    return result
            except (StaleElementReferenceException, NoSuchElementException):
                pass
            self._pause(0.25)
        raise TranslationFailure(message)

    def _source(self):
        selectors = ('#tw-source-text-ta', '#tw-container textarea:not([readonly])') if self.route == "search" else SOURCE_SELECTORS
        for selector in selectors:
            elements = self.driver.find_elements(self.By.CSS_SELECTOR, selector)
            shown = [e for e in elements if e.is_displayed() and e.is_enabled()]
            if shown:
                return shown[0]
        return None

    def _output(self):
        selectors = ('#tw-target-text',) if self.route == "search" else OUTPUT_SELECTORS
        for selector in selectors:
            elements = self.driver.find_elements(self.By.CSS_SELECTOR, selector)
            result = " ".join(e.text.strip() for e in elements if e.is_displayed() and e.text.strip())
            if result:
                return result.strip()
        return ""

    def _clear(self):
        # Do not spend a full 60 seconds waiting for an obsolete output to
        # disappear. A stale output after clearing triggers a fresh session.
        short_wait = min(self.settings.timeout, 3)
        element = self._wait(self._source, "Google's input disappeared.", short_wait)
        element.click()
        element.send_keys(self.Keys.CONTROL, "a")
        element.send_keys(self.Keys.BACKSPACE)
        self._wait(lambda: self._source() is not None and not self._source().get_attribute("value"),
                   "Google's input did not clear.", short_wait)
        self._wait(lambda: not self._output(), "Previous translation did not clear; restarting Chrome.", short_wait)
        return self._source()

    def _translate_part(self, text):
        language = self.language_override or effective_source_language(text, self.settings.source_language)
        self._open_language(language)
        element = self._clear()
        if self.last_submission is not None:
            self._pause(max(0, self.settings.interval - (time.monotonic() - self.last_submission)))
        self._blocked()
        element.click()
        self.driver.execute_cdp_cmd("Input.insertText", {"text": text})
        self.last_submission = time.monotonic()
        if self.route == "search":
            element.send_keys(self.Keys.ENTER)
        self._wait(lambda: self._source() is not None and self._source().get_attribute("value") == text,
                   "The exact cleaned text did not reach Google's translator.")
        last_result, stable_since = "", time.monotonic()
        invalid_result = ""

        def completed():
            nonlocal last_result, stable_since, invalid_result
            source = self._source()
            if source is None or source.get_attribute("value") != text:
                return False
            result = self._output()
            busy = any(e.is_displayed() for e in self.driver.find_elements(self.By.CSS_SELECTOR, '[role="progressbar"]'))
            if busy or not result:
                last_result, stable_since = "", time.monotonic()
                return False
            if result != last_result:
                last_result, stable_since = result, time.monotonic()
                return False
            if time.monotonic() - stable_since < 2:
                return False
            try:
                return validate_translation(text, result)
            except TranslationFailure as exc:
                # An echoed Hindi intermediate result need not mean failure.
                # Give the real translation time to arrive before recovering.
                invalid_result = str(exc)
                if time.monotonic() - stable_since >= 8:
                    invalid_result += " Stable invalid output; opening a fresh session."
                    raise TranslationFailure(invalid_result) from exc
                return False

        try:
            return self._wait(completed, "No verified English output appeared before the timeout.")
        except BrowserBlocked:
            raise
        except TranslationFailure as exc:
            if invalid_result:
                raise TranslationFailure(invalid_result) from exc
            raise

    def translate(self, text):
        errors = []
        current_route = self.route
        recovery = [(current_route, False), ("incognito", True), ("normal", True), ("search", True)]
        for number, (route, force_new) in enumerate(recovery, 1):
            check_stop(self.stop_event)
            try:
                # A script is not a language: Devanagari can be Sanskrit as
                # well as Hindi. After the first attempt, let Google detect it.
                self.language_override = None if number == 1 else "auto"
                if force_new or self.driver is None or self.session_successes >= self.settings.browser_restart_every:
                    self._start(route)
                result = validate_translation(text, self._translate_part(text))
                self.session_successes += 1
                return result
            except (InterruptedError, BrowserBlocked):
                raise
            except Exception as exc:
                if not browser_error_is_transient(exc):
                    self.close()
                    raise
                errors.append(f"{route}: {type(exc).__name__}: {exc}")
                self.log(f"Translation attempt {number}/4 failed ({route}, source={self.language_override or 'configured/script hint'}): {exc}")
                self.close()
                if number < len(recovery):
                    self.log("Retrying the SAME text in a fresh session with Detect language → English...")
                    self._pause(min(number, 3))
        self.route = "incognito"
        raise TranslationFailure("All browser routes failed for this text. " + " | ".join(errors))

    def close(self):
        if self.driver is not None:
            driver, self.driver = self.driver, None
            try:
                driver.quit()
            except Exception:
                # quit() itself can fail on a crashed window. Stop only this
                # Selenium service; never taskkill the user's Chrome processes.
                service = getattr(driver, "service", None)
                if service is not None:
                    try:
                        service.stop()
                    except Exception:
                        pass
        self.active_language = None


class PreservingTranslator:
    """Translate only non-English spans; retain original English and ordering."""
    def __init__(self, settings, cache, stop_event=None, log=print,
                 browser_factory=GoogleTranslateBrowser):
        self.settings, self.cache = settings, cache
        self.stop_event, self.log = stop_event, log
        self.browser_factory = browser_factory
        self.browser = None
        self.requests = 0

    def _span(self, text):
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        parts = split_post(text, self.settings.max_part_chars)
        outputs = []
        for part in parts:
            check_stop(self.stop_event)
            result = self.cache.get(part)
            if result is None:
                if self.browser is None:
                    self.log("Opening one Chrome Incognito window for non-English text...")
                    self.browser = self.browser_factory(self.settings, self.stop_event, self.log)
                result = validate_translation(part, self.browser.translate(part))
                self.requests += 1
                self.cache.put(part, result)
            outputs.append(result)
        result = validate_translation(text, " ".join(outputs))
        if len(parts) > 1:
            self.cache.put(text, result)
        return result

    def translate(self, text):
        check_stop(self.stop_event)
        segments = translation_segments(text)
        if not any(translate for translate, _ in segments):
            return text, "unchanged_english"
        cached = self.cache.get(text, engine=PRESERVED_CACHE_ENGINE)
        if cached is not None:
            return cached, "cached"
        previous_requests = self.requests
        result = "".join(self._span(piece) if translate else piece
                         for translate, piece in segments)
        validate_translation(text, result)
        validate_preserved_literals(text, result)
        self.cache.put(text, result, engine=PRESERVED_CACHE_ENGINE)
        return result, "translated" if self.requests > previous_requests else "cached"

    def close(self):
        if self.browser is not None:
            self.browser.close()
            self.browser = None


def translate_frame(frame, report, folder, settings, stop_event=None, log=print,
                    progress=None, browser_factory=GoogleTranslateBrowser):
    cache = TranslationCache(Path(settings.output_dir) / "translation_cache.sqlite3", settings.source_language)
    translator = PreservingTranslator(settings, cache, stop_event, log, browser_factory)
    result_file = folder / "translation_results.csv"
    current = None
    consecutive_failures = 0
    if settings.skip_failed_rows and result_file.is_file():
        previous = pd.read_csv(result_file, keep_default_na=False)
        required = {"data_row", "post_text_original", "translation_status", "translation_error"}
        if required <= set(previous.columns):
            skipped = previous.loc[previous["translation_status"].eq("skipped")].set_index("data_row")
            for index, row in report.iterrows():
                data_row = int(row["data_row"])
                if data_row in skipped.index and skipped.loc[data_row, "post_text_original"] == row["post_text_original"]:
                    report.at[index, "translation_status"] = "skipped"
                    report.at[index, "translation_error"] = skipped.loc[data_row, "translation_error"]
    save_csv(report, result_file)
    try:
        for repair_pass in range(settings.pending_retry_passes + 1):
            if repair_pass:
                log(f"Range pending-row retry pass {repair_pass}/{settings.pending_retry_passes}.")
            for count, (index, row) in enumerate(report.iterrows(), 1):
                current = index
                check_stop(stop_event)
                if row["translation_status"] in FINAL_STATUSES:
                    continue
                text = row["post_text_original"]
                if not text:
                    report.at[index, "translation_status"] = "empty"
                else:
                    try:
                        translated, status = translator.translate(text)
                    except (InterruptedError, BrowserBlocked):
                        raise
                    except Exception as exc:
                        if not browser_error_is_transient(exc):
                            raise
                        report.at[index, "translation_status"] = "skipped" if settings.skip_failed_rows else "retry_pending"
                        report.at[index, "translation_error"] = f"{type(exc).__name__}: {exc}"
                        translator.close()
                        consecutive_failures = 0 if settings.skip_failed_rows and untranslated_output_failure(exc) else consecutive_failures + 1
                        save_csv(report, result_file)
                        log(f"Data row {int(row['data_row'])}: {'skipped after all attempts' if settings.skip_failed_rows else 'saved for later retry'}; moving to the next row.")
                        if consecutive_failures >= settings.max_consecutive_failures:
                            raise TranslationUnavailable(f"{consecutive_failures} consecutive texts failed; check internet/service and resume.") from exc
                    else:
                        report.at[index, "post_text_original_english"] = translated
                        report.at[index, "translation_status"] = status
                        report.at[index, "translation_error"] = ""
                        consecutive_failures = 0
                if count % 50 == 0:
                    save_csv(report, result_file)
                if progress:
                    progress("translation", count, len(report), report.loc[index].to_dict())
            pending = report.loc[~report["translation_status"].isin(FINAL_STATUSES)]
            save_csv(pending, folder / "pending_rows.csv")
            if pending.empty:
                break
            if repair_pass < settings.pending_retry_passes:
                if stop_event is not None:
                    if stop_event.wait(5):
                        check_stop(stop_event)
                else:
                    time.sleep(5)
    except BaseException as exc:
        if current is not None:
            # A pause after a successful row must not erase that row's result.
            if report.at[current, "translation_status"] not in FINAL_STATUSES:
                report.at[current, "translation_status"] = "interrupted" if isinstance(exc, InterruptedError) else "retry_pending"
                report.at[current, "translation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            save_csv(report, result_file)
            save_csv(report.loc[~report["translation_status"].isin(FINAL_STATUSES)], folder / "pending_rows.csv")
            save_csv(report.loc[report["translation_status"].eq("skipped")], folder / "skipped_rows.csv")
            log(f"Translation status and English text saved: {result_file}")
        finally:
            translator.close()
            cache.close()
    if not report["translation_status"].isin(FINAL_STATUSES).all():
        raise PendingTranslations(f"All selected rows were attempted. Pending translations are saved in {folder / 'pending_rows.csv'}. Start / Resume retries unfinished text; sentiment waits for verified English.")
    frame["post_text_original_english"] = report["post_text_original_english"].tolist()
    skipped_count = int(report["translation_status"].eq("skipped").sum())
    if skipped_count:
        log(f"Skipped {skipped_count:,} untranslated rows after all attempts; details: {folder / 'skipped_rows.csv'}")
        if progress:
            progress("skipped", skipped_count, len(report), str(folder / "skipped_rows.csv"))
    return frame


def run_pipeline(settings, stop_event=None, log=print, progress=None,
                 browser_factory=GoogleTranslateBrowser, sentiment_function=add_original_sentiment_columns):
    frame, report, folder = prepare_frame(settings, log, stop_event)
    if progress:
        progress("prepared", 0, len(report), report.head(50).to_dict("records"))
    frame = translate_frame(frame, report, folder, settings, stop_event, log, progress, browser_factory)
    check_stop(stop_event)
    log("Translation complete. Starting the original IndicBERT, TextBlob and VADER calculations...")
    if progress:
        progress("model", 0, 1, None)
    frame = sentiment_function(frame, log, stop_event, progress)
    skipped = report.index[report["translation_status"].eq("skipped")]
    # Keep the original inference groups/calculations. Empty English never
    # supplies TextBlob/VADER scores; discard all reported scores on skipped rows.
    for col in frame.columns:
        if col.startswith(("post_text_original_indic_", "post_text_original_english_")):
            frame.loc[skipped, col] = None
    check_stop(stop_event)
    output_file = folder / f"combined4_sentiment_translation_{settings.start_row}_{settings.end_row}.xlsx"
    atomic_write(output_file, lambda p: frame.to_excel(p, index=False))
    log(f"Saved {len(frame):,} selected rows with the original output columns: {output_file}")
    return output_file
