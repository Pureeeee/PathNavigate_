import os
import re
import h5py
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image

def find_h5_file_for_wsi(feature_dir: str, short_id: str) -> Optional[str]:

    if not os.path.isdir(feature_dir):
        return None
    for fname in os.listdir(feature_dir):
        if fname.endswith('.h5') and fname.startswith(short_id):
            return os.path.join(feature_dir, fname)
    return None

def get_wsi_full_id_from_h5(feature_dir: str, short_id: str) -> Optional[str]:

    h5_path = find_h5_file_for_wsi(feature_dir, short_id)
    if h5_path is None:
        return None
    return os.path.splitext(os.path.basename(h5_path))[0]

def load_plip_feature_cache(feature_dir: str, wsi_id: str) -> Dict[str, np.ndarray]:

    if not os.path.isdir(feature_dir):
        return {}

    wsi_dir = None
    for d in os.listdir(feature_dir):
        if d.startswith(wsi_id) and os.path.isdir(os.path.join(feature_dir, d)):
            wsi_dir = os.path.join(feature_dir, d)
            break

    if wsi_dir is None:
        return {}

    cache = {}
    for fname in os.listdir(wsi_dir):
        if not fname.endswith('.npy'):
            continue
        name_no_ext = fname.replace('.npy', '')
        parts = name_no_ext.split('_')
        if len(parts) >= 2:
            try:
                x, y = int(parts[0]), int(parts[1])
                feat = np.load(os.path.join(wsi_dir, fname))
                cache[f"patch_{x}_{y}"] = feat
            except (ValueError, Exception):
                continue

    return cache

def load_feature_cache(feature_dir: str, wsi_id: str) -> Dict[str, np.ndarray]:

    h5_path = find_h5_file_for_wsi(feature_dir, wsi_id)
    if h5_path is None:
        return {}

    cache = {}
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        features = f['features'][:]
        for i in range(coords.shape[0]):
            x, y = int(coords[i, 0]), int(coords[i, 1])
            name = f"patch_{x}_{y}"
            cache[name] = features[i]

    return cache

def extract_coords_from_name(patch_name: str) -> Tuple[int, int]:

    base = os.path.basename(patch_name).split('.')[0]

    parts = base.split('_')
    nums_from_end = []
    for p in reversed(parts):
        if p.lstrip('-').isdigit():
            nums_from_end.append(int(p))
        else:
            break
    if len(nums_from_end) >= 2:
        return nums_from_end[1], nums_from_end[0]

    numbers = re.findall(r'-?\d+', base)
    if len(numbers) >= 2:
        return int(numbers[-2]), int(numbers[-1])

    return 0, 0

def get_20x_patches_in_region(
    center_x: int,
    center_y: int,
    feature_cache_20x: Dict[str, np.ndarray],
    radius: int = 4096,
) -> List[str]:

    result = []
    for name in feature_cache_20x.keys():
        x, y = extract_coords_from_name(name)
        if abs(x - center_x) <= radius and abs(y - center_y) <= radius:
            result.append(name)
    return result

_slide_cache = {}

def open_slide(svs_path: str):

    import openslide
    if svs_path not in _slide_cache:
        _slide_cache[svs_path] = openslide.OpenSlide(svs_path)
    return _slide_cache[svs_path]

def close_all_slides():

    for slide in _slide_cache.values():
        slide.close()
    _slide_cache.clear()

def find_svs_path(svs_dir: str, wsi_full_id: str) -> Optional[str]:

    candidate = os.path.join(svs_dir, f"{wsi_full_id}.svs")
    if os.path.exists(candidate):
        return candidate
    return None

def extract_patch_from_svs(
    svs_path: str,
    x: int,
    y: int,
    magnification: int = 20,
    patch_size: int = 512,
    native_mag: float = 40.0,
):

    slide = open_slide(svs_path)

    downsample = native_mag / magnification
    read_size_level0 = int(patch_size * downsample)

    region = slide.read_region((x, y), 0, (read_size_level0, read_size_level0))
    patch = region.convert('RGB').resize((patch_size, patch_size))

    return patch

def coords_to_patch_names(coords_list, all_patch_names):

    coord_to_name = {}
    for name in all_patch_names:
        x, y = extract_coords_from_name(name)
        coord_to_name[(x, y)] = name
    result = []
    for x, y in coords_list:
        if (x, y) in coord_to_name:
            result.append(coord_to_name[(x, y)])
    return result
