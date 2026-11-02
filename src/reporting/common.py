"""Presentation helpers and stable provenance for every report/figure."""
import math


def source_note(data):
    source = data['source']
    return ('Nguồn: run `'+source['run_id']+'`; metric `'+source['metric_version']+'`; '
            '[evaluation](inputs/evaluation.json) SHA `'+source['evaluation_sha256']+'`; '
            '[config](inputs/frozen_config.json) SHA `'+source['config_sha256']+'`; '
            '[model](inputs/frozen_models.json) SHA `'+source['models_sha256']+'`.')


def finite_max(values, default=0.):
    return max((float(value) for value in values if value is not None and math.isfinite(value)), default=default)


def stamp_figure(fig, data):
    source = data['source']
    fig.text(.01, .005, 'run: '+source['run_id']+' | evaluation SHA: '+source['evaluation_sha256']+
             ' | renderer: reporting-v1', fontsize=5)
