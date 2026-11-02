"""Portable reporting from the current four-decision experiment."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.experiments.protocol_specs import default_spec
from src.experiments.runner import RunService
from src.reporting.bundle import create_report, rebuild_report, verify_report
from test_runner import audio


class SelectedReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        train, test = cls.root/'train', cls.root/'test'
        audio(train)
        audio(test, frequency=190)
        cls.run_path = cls.root/'run'
        service = RunService(cls.run_path)
        with redirect_stdout(io.StringIO()):
            service.prepare(default_spec('champion'), train)
            service.train(train)
            service.evaluate(train, test)
        cls.report = create_report(cls.run_path)

    def test_only_selected_decisions_are_evaluated_and_reported(self):
        value = json.loads((self.run_path/'evaluation.json').read_bytes())
        strategies = {row['strategy'] for row in value['protocols']}
        self.assertEqual(strategies, {'otsu', 'kmeans', 'kmeans2d', 'dynamic', 'kmeans2d_input_active'})
        self.assertEqual(value['native_champion_replays_matched'], 6)
        self.assertIn('Otsu train', (self.report/'README.md').read_text(encoding='utf-8'))
        verify_report(self.report)

    def test_identical_report_reuses_verified_revision(self):
        before = (self.report/'report_manifest.json').read_bytes()
        self.assertEqual(create_report(self.run_path), self.report)
        self.assertEqual((self.report/'report_manifest.json').read_bytes(), before)

    def test_portable_bundle_rebuild_and_tamper_detection(self):
        rebuilt = rebuild_report(self.report, self.root/'rebuilt')
        self.assertEqual((rebuilt/'tables.csv').read_bytes(), (self.report/'tables.csv').read_bytes())
        self.assertEqual(verify_report(rebuilt)['source'], verify_report(self.report)['source'])
        (rebuilt/'tables.csv').write_text('tampered', encoding='utf-8')
        with self.assertRaises(ValueError):
            verify_report(rebuilt)


if __name__ == '__main__':
    unittest.main()
