"""Offline regression tests. Run: python -m unittest -v tests_recovery.
Browser/network behaviour is simulated; no model weights are downloaded.
"""
import ast
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd
import browser_pipeline as bp
import batch_runner as br


class Clock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def pause(self, seconds):
        self.now += seconds


class Element:
    def __init__(self, driver, kind):
        self.driver, self.kind = driver, kind
    @property
    def text(self):
        if self.kind == 'body':
            return self.driver.body
        if self.kind == 'source_language':
            return 'Hindi'
        if self.kind == 'target_language':
            return 'English'
        return self.driver.output
    def is_displayed(self): return True
    def is_enabled(self): return True
    def get_attribute(self, name):
        return self.driver.input if name == 'value' else None
    def click(self): pass
    def send_keys(self, *keys):
        if 'BACKSPACE' in keys:
            self.driver.input = ''
            if not self.driver.stale:
                self.driver.output = ''


class Driver:
    def __init__(self, clock=None, stale=False, echo_first=False):
        self.clock = clock or Clock()
        self.stale, self.echo_first = stale, echo_first
        self.input, self._output, self.body = '', '', ''
        self.current_url = 'about:blank'
        self.urls = []
        self.closed = False
        self.submitted = None
    @property
    def output(self):
        if self.submitted is not None and self.echo_first:
            return self.input if self.clock.now - self.submitted < 3 else 'At Hashmi Ji\'s house in Motihari'
        return self._output
    @output.setter
    def output(self, value): self._output = value
    def get(self, url):
        self.current_url = url
        self.urls.append(url)
        self.input, self._output, self.submitted = '', '', None
    def find_element(self, by, value): return Element(self, 'body')
    def find_elements(self, by, selector):
        if selector in bp.SOURCE_SELECTORS or selector in ('#tw-source-text-ta', '#tw-container textarea:not([readonly])'):
            return [Element(self, 'source')]
        if selector in bp.OUTPUT_SELECTORS or selector == '#tw-target-text':
            return [Element(self, 'output')]
        if selector == '#tw-sl': return [Element(self, 'source_language')]
        if selector == '#tw-tl': return [Element(self, 'target_language')]
        return []
    def execute_cdp_cmd(self, name, args):
        self.input = args['text']
        self._output = 'At Hashmi Ji\'s house in Motihari'
        self.submitted = self.clock.now
    def execute_script(self, script): pass
    def quit(self): self.closed = True


def selenium_modules():
    names = ['selenium', 'selenium.webdriver', 'selenium.webdriver.common',
             'selenium.webdriver.common.by', 'selenium.webdriver.common.keys',
             'selenium.common', 'selenium.common.exceptions',
             'selenium.webdriver.chrome', 'selenium.webdriver.chrome.service']
    modules = {name: types.ModuleType(name) for name in names}
    modules['selenium.webdriver.common.by'].By = types.SimpleNamespace(CSS_SELECTOR='css', XPATH='xpath', TAG_NAME='tag')
    modules['selenium.webdriver.common.keys'].Keys = types.SimpleNamespace(CONTROL='CONTROL', BACKSPACE='BACKSPACE', ENTER='ENTER')
    for name in ('TimeoutException', 'StaleElementReferenceException', 'NoSuchElementException'):
        setattr(modules['selenium.common.exceptions'], name, type(name, (Exception,), {}))
    return modules


class BrowserRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.modules = selenium_modules()
        self.patch = patch.dict(sys.modules, self.modules)
        self.patch.start()
        self.settings = bp.RunSettings(Path('input.xlsx'), Path('.'), timeout=8, interval=0)
        self.log = []
        self.browser = bp.GoogleTranslateBrowser(self.settings, None, self.log.append)
        self.clock = Clock()
        self.browser._pause = self.clock.pause
    def tearDown(self): self.patch.stop()
    def install_start(self, behaviours=None):
        starts = []
        behaviours = behaviours or {}
        def start(route):
            self.browser.close()
            starts.append(route)
            self.browser.route = route
            self.browser.driver = Driver(self.clock, **behaviours.get(len(starts), {}))
            self.browser.active_language = None
            self.browser.session_successes = 0
        self.browser._start = start
        return starts
    def test_script_hint_hindi(self):
        self.assertEqual(bp.effective_source_language('मोतिहारी हाशमी जी के घर पर'), 'hi')
    def test_stable_invalid_output_recovers_before_full_timeout(self):
        self.browser.settings = bp.RunSettings(Path('input.xlsx'), Path('.'), timeout=60, interval=0)
        self.install_start()
        self.browser._start('incognito')
        self.browser._output = lambda: 'नमस्ते' if self.browser.driver.input else ''
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            with self.assertRaisesRegex(bp.TranslationFailure, 'Stable invalid output'):
                self.browser._translate_part('नमस्ते')
        self.assertLess(self.clock.now, 15)
    def test_script_hint_bengali(self):
        self.assertEqual(bp.effective_source_language('আমার দেশ'), 'bn')
    def test_explicit_source_is_kept(self):
        self.assertEqual(bp.effective_source_language('नमस्ते', 'bn'), 'bn')
    def test_other_script_still_auto(self):
        self.assertEqual(bp.effective_source_language('你好'), 'auto')
    def test_hindi_url_not_auto(self):
        starts = self.install_start()
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            self.browser.translate('मोतिहारी हाशमी जी के घर पर')
        urls = self.browser.driver.urls
        self.assertEqual(parse_qs(urlparse(urls[-1]).query)['sl'], ['hi'])
        self.assertEqual(starts, ['incognito'])
    def test_stale_page_reopens_incognito_same_text(self):
        starts = self.install_start()
        submitted = []
        def part(text):
            submitted.append(text)
            if len(submitted) == 1:
                raise bp.TranslationFailure('Previous translation did not clear')
            return 'Hello'
        self.browser._translate_part = part
        self.assertEqual(self.browser.translate('नमस्ते'), 'Hello')
        self.assertEqual(starts, ['incognito', 'incognito'])
        self.assertEqual(submitted, ['नमस्ते', 'नमस्ते'])
    def test_hindi_failure_uses_auto_detection_for_sanskrit_same_text(self):
        self.install_start()
        text = 'शरणागत दीनार्तपरित्राण परायणे। सर्वस्यातिहरे देवि नारायणी नमोस्तुते।।'
        attempts = []
        original_part = self.browser._translate_part
        def part(source):
            attempts.append((source, self.browser.language_override))
            if self.browser.language_override is None:
                raise bp.TranslationFailure('The non-English input was not converted to English.')
            return original_part(source)
        self.browser._translate_part = part
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            self.browser.translate(text)
        self.assertEqual(attempts, [(text, None), (text, 'auto')])
        query = parse_qs(urlparse(self.browser.driver.urls[-1]).query)
        self.assertEqual(query['sl'], ['auto'])
        self.assertEqual(query['tl'], ['en'])

    def test_next_text_uses_script_hint_again_after_detect_fallback(self):
        self.install_start()
        overrides = []
        def part(text):
            overrides.append(self.browser.language_override)
            if len(overrides) == 1:
                raise bp.TranslationFailure('invalid output')
            return 'Hello'
        self.browser._translate_part = part
        self.browser.translate('शरणागत')
        self.browser.translate('नमस्ते')
        self.assertEqual(overrides, [None, 'auto', None])
    def test_normal_chrome_fallback_and_reuse(self):
        starts = self.install_start()
        def part(text):
            if self.browser.route == 'incognito': raise bp.TranslationFailure('timeout')
            return 'Hello'
        self.browser._translate_part = part
        self.assertEqual(self.browser.translate('नमस्ते'), 'Hello')
        self.assertEqual(self.browser.translate('नया'), 'Hello')
        self.assertEqual(starts, ['incognito', 'incognito', 'normal'])
    def test_search_last_fallback(self):
        starts = self.install_start()
        def part(text):
            if self.browser.route != 'search': raise bp.TranslationFailure('no output')
            return 'Hello'
        self.browser._translate_part = part
        self.assertEqual(self.browser.translate('नमस्ते'), 'Hello')
        self.assertEqual(starts, ['incognito', 'incognito', 'normal', 'search'])
    def test_all_routes_exhausted_close_browser(self):
        starts = self.install_start()
        self.browser._translate_part = lambda text: (_ for _ in ()).throw(bp.TranslationFailure('no output'))
        with self.assertRaises(bp.TranslationFailure): self.browser.translate('नमस्ते')
        self.assertEqual(len(starts), 4)
        self.assertIsNone(self.browser.driver)
    def test_verification_never_switches_route(self):
        starts = self.install_start()
        self.browser._translate_part = lambda text: (_ for _ in ()).throw(bp.BrowserBlocked('verify'))
        with self.assertRaises(bp.BrowserBlocked): self.browser.translate('नमस्ते')
        self.assertEqual(starts, ['incognito'])
    def test_block_after_invalid_intermediate_not_downgraded(self):
        self.browser.active_language = 'hi'
        self.browser.driver = Driver()
        self.browser._open_language = lambda language: None
        self.browser._clear = lambda: Element(self.browser.driver, 'source')
        calls = [0]
        def wait(predicate, message, timeout=None):
            calls[0] += 1
            if calls[0] == 1: return True
            raise bp.BrowserBlocked('verify')
        self.browser._wait = wait
        with self.assertRaises(bp.BrowserBlocked): self.browser._translate_part('नमस्ते')
    def test_intermediate_hindi_echo_waits_for_english(self):
        self.install_start({1: {'echo_first': True}})
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            self.assertEqual(self.browser.translate('मोतिहारी हाशमी जी के घर पर'), "At Hashmi Ji's house in Motihari")
    def test_clear_stale_output_has_short_timeout(self):
        self.browser.driver = Driver(stale=True)
        self.browser.driver.input, self.browser.driver.output = 'previous', 'old output'
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            with self.assertRaises(bp.TranslationFailure): self.browser._clear()
        self.assertLessEqual(self.clock.now, 3.25)
    def test_rotation_keeps_working_route(self):
        starts = self.install_start()
        self.browser._translate_part = lambda text: 'Hello'
        self.browser.settings = bp.RunSettings(Path('input.xlsx'), Path('.'), browser_restart_every=2)
        for _ in range(3): self.browser.translate('नमस्ते')
        self.assertEqual(starts, ['incognito', 'incognito'])
    def test_programming_error_not_swallowed(self):
        starts = self.install_start()
        self.browser._translate_part = lambda text: (_ for _ in ()).throw(ValueError('bug'))
        with self.assertRaises(ValueError): self.browser.translate('नमस्ते')
        self.assertEqual(starts, ['incognito'])
    def test_stop_interrupts_recovery(self):
        starts = self.install_start()
        event = threading.Event()
        self.browser.stop_event = event
        def part(text):
            event.set()
            raise bp.TranslationFailure('timeout')
        self.browser._translate_part = part
        with self.assertRaises(InterruptedError): self.browser.translate('नमस्ते')
        self.assertEqual(starts, ['incognito'])
    def test_search_only_reads_widget_output(self):
        self.browser.route, self.browser.driver = 'search', Driver()
        self.browser.driver.output = 'Verified widget result'
        self.assertEqual(self.browser._output(), 'Verified widget result')
        self.assertTrue(self.browser._search_languages_match('hi'))
        self.assertFalse(self.browser._search_languages_match('bn'))
    def test_source_mismatch_never_accepts_result(self):
        self.install_start()
        self.browser.driver = Driver(self.clock)
        self.browser.active_language = 'hi'
        self.browser.driver.execute_cdp_cmd = lambda *args: None
        with patch.object(bp.time, 'monotonic', self.clock.monotonic):
            with self.assertRaises(bp.TranslationFailure): self.browser._translate_part('नमस्ते')

    def test_incognito_and_normal_are_actual_different_launch_options(self):
        launched = []
        class Options:
            def __init__(self): self.arguments = []
            def add_argument(self, arg): self.arguments.append(arg)
            def add_experimental_option(self, key, value): pass
        class Chrome(Driver):
            def __init__(self, service, options):
                super().__init__()
                launched.append(options)
                self.command_executor = types.SimpleNamespace(_client_config=types.SimpleNamespace(timeout=None))
            def set_page_load_timeout(self, timeout): self.page_timeout = timeout
            def set_script_timeout(self, timeout): self.script_timeout = timeout
        self.modules['selenium.webdriver'].ChromeOptions = Options
        self.modules['selenium.webdriver'].Chrome = Chrome
        self.modules['selenium.webdriver.chrome.service'].Service = lambda **kw: None
        self.browser._start('incognito')
        first = self.browser.driver
        self.browser._start('normal')
        self.assertTrue(first.closed)
        self.assertIn('--incognito', launched[0].arguments)
        self.assertNotIn('--incognito', launched[1].arguments)
        self.assertEqual(self.browser.driver.command_executor._client_config.timeout, 8)


class FakeBrowser:
    calls = []
    failures = {}
    block = False
    def __init__(self, settings, stop_event, log): pass
    def translate(self, text):
        self.__class__.calls.append(text)
        if self.__class__.block: raise bp.BrowserBlocked('verification')
        failures = self.__class__.failures.get(text, 0)
        if failures:
            self.__class__.failures[text] = failures - 1
            raise bp.TranslationFailure('stale output on all routes')
        return 'English ' + ''.join(c for c in text if c.isascii()).strip()
    def close(self): pass


def fake_sentiment(frame, log=print, stop_event=None, progress=None):
    result = frame.copy()
    for col in br.VALUE_COLUMNS:
        if col == 'post_text_original_english': continue
        result[col] = 'positive' if col.endswith('label') else 0.25
    return result


class PipelineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        FakeBrowser.calls, FakeBrowser.failures, FakeBrowser.block = [], {}, False
        self.input = self.root / 'input.xlsx'
        self.output = self.root / 'results'
        self.sentiment_groups = []
    def tearDown(self): self.temp.cleanup()
    def dataset(self, n=48):
        pd.DataFrame({'post_text': [f'नमस्ते {i}' for i in range(1, n+1)],
                      'datetime': ['2026-01-01']*n}).to_excel(self.input, index=False)
    def settings(self, **kwargs):
        base = dict(input_file=self.input, output_dir=self.output, batch_size=32,
                    max_batches=100, interval=0, pending_retry_passes=0, skip_failed_rows=False)
        base.update(kwargs)
        return br.BatchSettings(**base)
    def sentiment(self, frame, *args):
        self.sentiment_groups.append(frame.index.tolist())
        return fake_sentiment(frame, *args)
    def run_job(self, settings):
        return br.run_batch_pipeline(settings, log=lambda *a: None, browser_factory=FakeBrowser, sentiment_function=self.sentiment)
    def folder(self): return next(self.output.glob('batch_run_*'))
    def read_csv(self, path):
        with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
    def test_pending_row_does_not_stop_later_batches(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 1
        with self.assertRaises(bp.PendingTranslations): self.run_job(self.settings())
        self.assertEqual(len(FakeBrowser.calls), 48)
        self.assertTrue((self.folder()/'batch_002_rows_33_48.csv').exists())
        self.assertFalse((self.folder()/'FULL_rows_1_48.csv').exists())
        self.assertEqual([r['data_row'] for r in self.read_csv(self.folder()/'pending_rows.csv')], ['2'])

    def test_skipped_row_keeps_original_group_and_full_csv_with_blank_scores(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 100
        output = self.run_job(self.settings(skip_failed_rows=True))
        rows = self.read_csv(output)
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[1]['post_text_original_english'], '')
        for col in br.VALUE_COLUMNS:
            self.assertEqual(rows[1][col], '')
        self.assertEqual(rows[2]['post_text_original_english_vader_compound'], '0.25')
        self.assertEqual(self.sentiment_groups, [list(range(16)), list(range(16,32)), list(range(32,48))])
        skipped = self.read_csv(self.folder()/'skipped_rows.csv')
        self.assertEqual([r['data_row'] for r in skipped], ['2'])
        self.assertEqual(skipped[0]['translation_status'], 'skipped')
        self.assertEqual(skipped[0]['analysis_status'], 'skipped')
        self.assertTrue(skipped[0]['translation_error'])
        self.assertEqual(self.read_csv(self.folder()/'pending_rows.csv'), [])
        FakeBrowser.calls.clear()
        self.sentiment_groups.clear()
        self.run_job(self.settings(skip_failed_rows=True))
        self.assertEqual(FakeBrowser.calls, [])
        self.assertEqual(self.sentiment_groups, [])

    def test_skipped_rows_restore_before_analysis_after_abrupt_stop(self):
        self.dataset(16)
        FakeBrowser.failures['नमस्ते 2'] = 100
        def stop_before_analysis(*args):
            raise InterruptedError('power loss before scores')
        with self.assertRaises(InterruptedError):
            br.run_batch_pipeline(self.settings(skip_failed_rows=True), log=lambda *a:None,
                                  browser_factory=FakeBrowser, sentiment_function=stop_before_analysis)
        FakeBrowser.calls.clear()
        output = self.run_job(self.settings(skip_failed_rows=True))
        self.assertEqual(FakeBrowser.calls, [])
        self.assertEqual(self.read_csv(output)[1]['post_text_original_indic_pos'], '')

    def test_existing_pending_checkpoint_resumes_with_skip_without_redoing_saved_groups(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 100
        with self.assertRaises(bp.PendingTranslations):
            self.run_job(self.settings())
        old_folder = self.folder()
        self.sentiment_groups.clear()
        FakeBrowser.calls.clear()
        output = self.run_job(self.settings(skip_failed_rows=True))
        self.assertEqual(output.parent, old_folder)
        self.assertEqual(FakeBrowser.calls, ['नमस्ते 2'])
        self.assertEqual(self.sentiment_groups, [list(range(16))])
        self.assertEqual(len(self.read_csv(output)), 48)

    def test_many_foreign_outputs_skip_without_false_service_outage(self):
        self.dataset(8)
        class ForeignOutputBrowser(FakeBrowser):
            def translate(self, text):
                raise bp.TranslationFailure('All browser routes failed: The non-English input was not converted to English.')
        output = br.run_batch_pipeline(self.settings(skip_failed_rows=True), log=lambda *a:None,
                                       browser_factory=ForeignOutputBrowser, sentiment_function=self.sentiment)
        self.assertEqual(len(self.read_csv(output)), 8)
        self.assertEqual(len(self.read_csv(self.folder()/'skipped_rows.csv')), 8)

    def test_range_mode_skip_and_resume(self):
        self.dataset(4)
        FakeBrowser.failures['नमस्ते 2'] = 100
        settings = bp.RunSettings(self.input,self.output,end_row=4,interval=0)
        output = bp.run_pipeline(settings,log=lambda *a:None,browser_factory=FakeBrowser,sentiment_function=self.sentiment)
        result = pd.read_excel(output,keep_default_na=False)
        self.assertEqual(result.loc[1,'post_text_original_english'], '')
        self.assertEqual(result.loc[1,'post_text_original_english_vader_compound'], '')
        FakeBrowser.calls.clear()
        bp.run_pipeline(settings,log=lambda *a:None,browser_factory=FakeBrowser,sentiment_function=self.sentiment)
        self.assertEqual(FakeBrowser.calls, [])

    def test_incomplete_batch_has_separate_partial_csv(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 1
        with self.assertRaises(bp.PendingTranslations): self.run_job(self.settings())
        rows=self.read_csv(self.folder()/'batch_001_partial.csv')
        self.assertEqual(len(rows),32)
        self.assertEqual(rows[1]['analysis_status'],'pending')
        self.assertEqual(rows[16]['analysis_status'],'saved')
        self.run_job(self.settings())
        self.assertFalse((self.folder()/'batch_001_partial.csv').exists())
    def test_retry_pass_completes_ordered_full_csv(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 1
        with patch.object(bp.time, 'sleep', lambda seconds: None):
            output = self.run_job(self.settings(pending_retry_passes=2))
        rows = self.read_csv(output)
        self.assertEqual(len(rows), 48)
        self.assertEqual([r['post_text'] for r in rows], [f'नमस्ते {i}' for i in range(1,49)])
        self.assertEqual(FakeBrowser.calls.count('नमस्ते 2'), 2)
        self.assertEqual(FakeBrowser.calls.count('नमस्ते 3'), 1)
        self.assertEqual(self.read_csv(self.folder()/'pending_rows.csv'), [])
    def test_resume_only_failed_translation_and_original_group(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 1
        with self.assertRaises(bp.PendingTranslations): self.run_job(self.settings())
        self.assertEqual(self.sentiment_groups, [list(range(16,32)), list(range(32,48))])
        FakeBrowser.calls.clear()
        self.sentiment_groups.clear()
        output = self.run_job(self.settings())
        self.assertEqual(FakeBrowser.calls, ['नमस्ते 2'])
        self.assertEqual(self.sentiment_groups, [list(range(16))])
        self.assertEqual(len(self.read_csv(output)), 48)
        self.assertFalse((self.folder()/'PARTIAL_rows_1_48.csv').exists())
    def test_partial_rows_never_receive_missing_sentiment(self):
        self.dataset()
        FakeBrowser.failures['नमस्ते 2'] = 100
        with self.assertRaises(bp.PendingTranslations): self.run_job(self.settings())
        rows = self.read_csv(self.folder()/'PARTIAL_rows_1_48.csv')
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[1]['post_text_original_english'], '')
        self.assertEqual(rows[1]['post_text_original_english_vader_compound'], '')
        self.assertEqual(rows[2]['analysis_status'], 'pending')
        self.assertEqual(rows[16]['analysis_status'], 'saved')
    def test_consecutive_outage_pauses_instead_of_100000_failures(self):
        self.dataset()
        FakeBrowser.failures = {f'नमस्ते {i}': 100 for i in range(1,49)}
        with self.assertRaises(bp.TranslationUnavailable): self.run_job(self.settings())
        self.assertEqual(len(FakeBrowser.calls), 5)
        self.assertTrue((self.folder()/'PARTIAL_rows_1_48.csv').exists())
    def test_verification_preserves_progress_without_fallback(self):
        self.dataset()
        FakeBrowser.block = True
        with self.assertRaises(bp.BrowserBlocked): self.run_job(self.settings())
        self.assertEqual(len(FakeBrowser.calls), 1)
        self.assertTrue((self.folder()/'checkpoint.sqlite3').exists())
    def test_100_batch_order_and_exact_row_count(self):
        self.dataset(200)
        output = self.run_job(self.settings(batch_size=2, max_batches=100))
        self.assertEqual(len(list(self.folder().glob('batch_*_rows_*.csv'))), 100)
        self.assertEqual(len(self.read_csv(output)), 200)
    def test_subset_inclusive_and_empty_english_bypass(self):
        pd.DataFrame({'post_text': ['excluded', None, 'Keep Me!', 'नमस्ते 4', 'excluded'],
                      'datetime': ['2026-01-01']*5}).to_excel(self.input,index=False)
        output = self.run_job(self.settings(start_row=2,batch_size=3,max_batches=1))
        rows = self.read_csv(output)
        self.assertEqual([r['post_text_original_english'] for r in rows], ['', 'Keep Me!', 'English 4'])
        self.assertEqual(FakeBrowser.calls, ['नमस्ते 4'])
    def test_duplicate_hindi_uses_cache(self):
        pd.DataFrame({'post_text': ['नमस्ते 1']*3, 'datetime':['2026-01-01']*3}).to_excel(self.input,index=False)
        self.run_job(self.settings())
        self.assertEqual(FakeBrowser.calls, ['नमस्ते 1'])
    def test_mixed_literals_kept_exactly(self):
        self.dataset(1)
        cache = bp.TranslationCache(self.root/'cache.sqlite3', 'auto')
        try:
            translator = bp.PreservingTranslator(self.settings(), cache, browser_factory=FakeBrowser, log=lambda *a:None)
            source = 'Keep THIS. नमस्ते 9 then English!'
            result, status = translator.translate(source)
            self.assertEqual(result, 'Keep THIS. English 9 then English!')
            self.assertEqual(FakeBrowser.calls, ['नमस्ते'])
        finally: cache.close()
    def test_split_parts_survive_second_part_failure(self):
        self.dataset(1)
        settings = self.settings(max_part_chars=9)
        cache = bp.TranslationCache(self.root/'cache.sqlite3', 'auto')
        try:
            translator = bp.PreservingTranslator(settings, cache, browser_factory=FakeBrowser, log=lambda *a:None)
            source = 'नमस्ते 1 नमस्ते 2'
            parts = bp.split_post(source, 9)
            FakeBrowser.failures[parts[1]] = 1
            with self.assertRaises(bp.TranslationFailure): translator.translate(source)
            translator.close()
            translator.translate(source)
            self.assertEqual(FakeBrowser.calls.count(parts[0]), 1)
        finally: cache.close()
    def test_range_mode_retries_pending_without_sentiment_early(self):
        self.dataset(4)
        FakeBrowser.failures['नमस्ते 2'] = 1
        settings = bp.RunSettings(self.input,self.output,end_row=4,interval=0,pending_retry_passes=1,skip_failed_rows=False)
        with patch.object(bp.time, 'sleep', lambda seconds:None):
            output = bp.run_pipeline(settings, log=lambda *a:None, browser_factory=FakeBrowser,sentiment_function=self.sentiment)
        self.assertTrue(output.exists())
        self.assertEqual(FakeBrowser.calls.count('नमस्ते 2'),2)
        self.assertEqual(self.sentiment_groups,[list(range(4))])
    def test_rebuild_missing_csv_does_not_retranslate_or_reanalyse(self):
        self.dataset(32)
        output = self.run_job(self.settings())
        (output.parent/'batch_001_rows_1_32.csv').unlink()
        FakeBrowser.calls.clear()
        self.sentiment_groups.clear()
        output = self.run_job(self.settings())
        self.assertTrue(output.exists())
        self.assertEqual(FakeBrowser.calls, [])
        self.assertEqual(self.sentiment_groups, [])
    def test_abrupt_process_exit_reuses_saved_group(self):
        self.dataset(32)
        code = r'''import os,sys
from pathlib import Path
import tests_recovery as t
import batch_runner as b
calls=[0]
def sentiment(frame,*args):
 calls[0]+=1
 if calls[0]==2: os._exit(19)
 return t.fake_sentiment(frame,*args)
s=b.BatchSettings(Path(sys.argv[1]),Path(sys.argv[2]),batch_size=32,max_batches=1,interval=0,pending_retry_passes=0)
b.run_batch_pipeline(s,log=lambda *a:None,browser_factory=t.FakeBrowser,sentiment_function=sentiment)
'''
        child = subprocess.run([sys.executable,'-c',code,str(self.input),str(self.output)],cwd=Path(__file__).parent, capture_output=True,text=True)
        self.assertEqual(child.returncode,19,child.stderr)
        output = self.run_job(self.settings(max_batches=1))
        self.assertEqual(FakeBrowser.calls, [])
        self.assertEqual(self.sentiment_groups, [list(range(16,32))])
        self.assertEqual(len(self.read_csv(output)),32)


class ValidationTests(unittest.TestCase):
    def test_original_cleaning_and_analysis_module_unchanged(self):
        # Compare baseline outside the distributable package when available.
        base=Path(__file__).resolve().parents[2]/'baseline'
        if not base.exists(): self.skipTest('Baseline comparison is part of the build verification')
        self.assertEqual((base/'analysis_logic.py').read_bytes(),Path(br.analysis_logic.__file__).read_bytes())
        self.assertEqual((base/'original_indicbert.py').read_bytes(),(Path(__file__).parent/'original_indicbert.py').read_bytes())
    def test_foreign_output_not_accepted(self):
        for output in ('नमस्ते', 'Hello नमस्ते', '', '123'):
            with self.assertRaises(bp.TranslationFailure): bp.validate_translation('नमस्ते', output)
    def test_google_error_cannot_pollute_cache(self):
        with tempfile.TemporaryDirectory() as root:
            cache=bp.TranslationCache(Path(root)/'cache.sqlite3','auto')
            try:
                with self.assertRaises(bp.TranslationFailure): cache.put('नमस्ते','नमस्ते')
                self.assertIsNone(cache.get('नमस्ते'))
            finally: cache.close()
    def test_unicode_hashtag_bug_stays_fixed(self):
        text='मेरी अपील #परिवर्तन_ही_विकल्प #महाराजगंज_मांगे_परिवर्तन!'
        self.assertEqual(br.analysis_logic.remove_hashtags(text),'मेरी अपील !')
    def test_atomic_save_failure_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'data.csv'
            path.write_text('previous')
            def fail(p):
                p.write_text('partial')
                raise OSError('disk full')
            with self.assertRaises(OSError): bp.atomic_write(path,fail)
            self.assertEqual(path.read_text(),'previous')
    def test_restart_limits_validate(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'in.xlsx'
            path.touch()
            with self.assertRaises(ValueError): bp.RunSettings(path,Path(root),browser_restart_every=0).validate()

if __name__ == '__main__':
    unittest.main(verbosity=2)
