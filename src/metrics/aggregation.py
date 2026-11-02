"""Named legacy reductions; keep historical rounding boundaries explicit."""
from .numeric import mean_metric, round_metric

FIELDS = ('abs_error_mean', 'mape_f0', 'abs_error_std', 'mape_std',
          'abs_error_num', 'mape_num', 'final_score')
FILE_MACRO = 'file-score-macro-v1'
ROUNDED_MACRO = 'file-score-macro-rounded-2-v1'
ADVANCED_MACRO = 'file-score-macro-count-1-v1'
COMPONENT_MACRO = 'component-score-macro-v1'
PROFILES = (FILE_MACRO, ROUNDED_MACRO, ADVANCED_MACRO, COMPONENT_MACRO)


def aggregate_rows(rows, profile=FILE_MACRO):
    if profile not in PROFILES:
        raise ValueError('Unknown legacy aggregation profile')
    rows = list(rows)
    digits = None if profile in (FILE_MACRO, COMPONENT_MACRO) else 2
    result = {key: mean_metric((row.get(key) for row in rows), digits=digits) for key in FIELDS}
    if profile == ADVANCED_MACRO:
        result['abs_error_num'] = mean_metric((row.get('abs_error_num') for row in rows), digits=1)
    if profile == COMPONENT_MACRO:
        # Benchmark 14 historically averages rounded component MAPEs, instead
        # of the rounded per-file Final Scores used by the experiment runner.
        result['final_score'] = mean_metric([result[key] for key in ('mape_f0', 'mape_std', 'mape_num')], digits=None)
    return result


def aggregate(rows):
    """Compatibility name for the runner's unrounded macro of file metrics."""
    return aggregate_rows(rows, FILE_MACRO)


def table_metrics(rows, profile=ROUNDED_MACRO):
    values = aggregate_rows(rows, profile)
    names = {'diff_f0': 'abs_error_mean', 'mape_f0': 'mape_f0',
             'diff_std': 'abs_error_std', 'mape_std': 'mape_std',
             'diff_num': 'abs_error_num', 'mape_num': 'mape_num', 'final_score': 'final_score'}
    return {output: values[key] for output, key in names.items()}


def suite_metrics(rows):
    """Benchmark 14's existing two-decimal export, with its component score."""
    values = aggregate_rows(rows, COMPONENT_MACRO)
    names = {'average_error_hz': 'abs_error_mean', 'average_std_error': 'abs_error_std',
             'average_num_error': 'abs_error_num', 'mape_f0_pct': 'mape_f0',
             'mape_std_pct': 'mape_std', 'mape_num_pct': 'mape_num',
             'final_composite_score_pct': 'final_score'}
    return {output: round_metric(values[key], 2) for output, key in names.items()}


def sweep_metrics(rows):
    values = aggregate_rows(rows, ROUNDED_MACRO)
    names = {'average_error_hz': 'abs_error_mean', 'average_mape_f0_pct': 'mape_f0',
             'average_std_error': 'abs_error_std', 'average_mape_std_pct': 'mape_std',
             'average_num_error': 'abs_error_num', 'average_mape_num_pct': 'mape_num',
             'final_composite_score_pct': 'final_score'}
    return {output: values[key] for output, key in names.items()}
