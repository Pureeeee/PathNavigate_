import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from data_processing.feature_utils import extract_coords_from_name

def generate_surprise_heatmap(feature_cache, patch_names, yaad_memory, device="cuda"):

    if not patch_names:
        return {}, {}, np.zeros((1, 1)), (0, 0, 1)

    sorted_patches = sorted(
        patch_names,
        key=lambda n: (extract_coords_from_name(n)[1], extract_coords_from_name(n)[0])
    )

    surprise_map = {}
    coord_surprise_map = {}
    coords_data = []

    for patch_name in sorted_patches:
        feat_np = feature_cache[patch_name]
        feat = torch.tensor(feat_np, dtype=torch.float32).to(device)

        feat = F.normalize(feat, p=2, dim=0)

        surprise = yaad_memory.compute_and_update(feat)

        surprise_map[patch_name] = surprise
        x, y = extract_coords_from_name(patch_name)
        coord_surprise_map[(x, y)] = surprise
        coords_data.append((x, y, surprise))

    heatmap, grid_meta = _build_heatmap(coords_data)

    return surprise_map, coord_surprise_map, heatmap, grid_meta

def _build_heatmap(coords_data):

    if not coords_data:
        return np.zeros((1, 1)), (0, 0, 1)

    xs = [c[0] for c in coords_data]
    ys = [c[1] for c in coords_data]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_sorted = sorted(set(xs))
    if len(x_sorted) > 1:
        diffs = [x_sorted[i + 1] - x_sorted[i] for i in range(len(x_sorted) - 1)]
        step = min(d for d in diffs if d > 0)
    else:
        y_sorted = sorted(set(ys))
        if len(y_sorted) > 1:
            diffs = [y_sorted[i + 1] - y_sorted[i] for i in range(len(y_sorted) - 1)]
            step = min(d for d in diffs if d > 0)
        else:
            step = 1

    H = max(1, (y_max - y_min) // step + 1)
    W = max(1, (x_max - x_min) // step + 1)

    MAX_DIM = 5000
    if H > MAX_DIM or W > MAX_DIM:

        scale = max(H, W) / MAX_DIM
        step = int(step * scale) + 1
        H = max(1, (y_max - y_min) // step + 1)
        W = max(1, (x_max - x_min) // step + 1)

    heatmap = np.zeros((int(H), int(W)), dtype=np.float32)

    for x, y, s in coords_data:
        col = (x - x_min) // step
        row = (y - y_min) // step
        col = min(col, W - 1)
        row = min(row, H - 1)
        heatmap[int(row), int(col)] = max(heatmap[int(row), int(col)], s)

    return heatmap, (x_min, y_min, step)

def nms_jump_targets(heatmap, grid_meta, K=10, nms_window=3, min_surprise_percentile=50,
                     min_distance=0):

    if heatmap.max() == 0:
        return []

    nonzero_vals = heatmap[heatmap > 0]
    if len(nonzero_vals) == 0:
        return []
    min_threshold = np.percentile(nonzero_vals, min_surprise_percentile)

    local_max = maximum_filter(heatmap, size=nms_window)
    is_peak = (heatmap == local_max) & (heatmap >= min_threshold)

    peak_ys, peak_xs = np.where(is_peak)
    if len(peak_ys) == 0:
        return []

    peak_values = heatmap[peak_ys, peak_xs]

    sorted_indices = np.argsort(peak_values)[::-1]

    x_min, y_min, step = grid_meta
    jump_targets = []
    for idx in sorted_indices:
        if len(jump_targets) >= K:
            break
        orig_x = int(peak_xs[idx]) * step + x_min
        orig_y = int(peak_ys[idx]) * step + y_min

        if min_distance > 0 and jump_targets:
            too_close = False
            for tx, ty in jump_targets:
                dist = ((orig_x - tx) ** 2 + (orig_y - ty) ** 2) ** 0.5
                if dist < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

        jump_targets.append((orig_x, orig_y))

    return jump_targets
