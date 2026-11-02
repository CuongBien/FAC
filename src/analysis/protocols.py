"""Decision helpers for scalar/vector thresholds and frozen 2D centroids."""
import numpy as np

def threshold_labels(fd, method, threshold):
    values = fd['periodicity_vals']
    active = fd['is_active']
    voiced = values <= threshold if method == 'amdf' else values >= threshold
    return np.where(active, np.where(voiced, 'v', 'uv'), 'sil').astype(object)

def predict_frozen_cluster(fd, model):
    labels = np.where(fd['is_active'], 'uv', 'sil').astype(object)
    if model['status'] != 'ok':
        return labels
    active = fd['is_active']
    features = np.column_stack([
        (fd['periodicity_vals'][active]-model['p_mean'])/model['p_std'],
        (fd['zcr_vals'][active]-model['z_mean'])/model['z_std'],
    ])
    dv = np.sum((features-np.asarray(model['centroid_v_normalized']))**2, axis=1)
    du = np.sum((features-np.asarray(model['centroid_uv_normalized']))**2, axis=1)
    labels[active] = np.where((dv < du) & ~np.isclose(dv, du, rtol=1e-12, atol=1e-12), 'v', 'uv')
    return labels
