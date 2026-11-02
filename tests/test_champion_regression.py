"""Original frozen Champion presets must retain their per-file results."""
import json
from pathlib import Path
import unittest

from src.metrics.evaluation import evaluate_result
from src.metrics.aggregation import aggregate
from src.pipelines import build_champion

ROOT = Path(__file__).resolve().parents[1]


class ChampionRegressionTests(unittest.TestCase):
    def test_original_train_and_test_results_are_preserved(self):
        reference = json.loads((ROOT/'tests/fixtures/champion_reference.json').read_bytes())
        for item in reference:
            folder = ROOT/('TinHieuHuanLuyen' if item['dataset'] == 'train' else 'TinHieuKiemThu')
            detector = build_champion(item['method'])
            rows = []
            for expected in item['files']:
                with self.subTest(dataset=item['dataset'], method=item['method'], file=expected['file']):
                    path = folder/expected['file']
                    if path.suffix != '.wav':
                        path = path.with_suffix('.wav')
                    row = evaluate_result(detector.process_file(str(path)), path)
                    rows.append(row)
                    self.assertEqual({key: row[key] for key in expected}, expected)
            self.assertEqual(aggregate(rows), item['metrics'])

    def test_only_champion_processing_plugins_are_exported(self):
        import src.plugins as plugins
        exported = {name for name in vars(plugins) if name.endswith('Plugin')}
        self.assertEqual(exported - {'BasePlugin', 'PreProcessingPlugin', 'DecisionPlugin', 'PostProcessingPlugin'},
                         {'BandpassFilterPlugin', 'CenterClippingPlugin', 'EnergyExtensionPlugin', 'ViterbiTrackingPlugin'})


if __name__ == '__main__':
    unittest.main()
