import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate, zoom

MODALITY_KEYS = ("t1c", "t1", "t2")


def normalize_foreground(volume):
    volume = volume.astype(np.float32, copy=False)
    foreground = np.abs(volume) > 1e-06
    if not foreground.any():
        foreground = np.ones_like(volume, dtype=bool)
    fg_values = volume[foreground]
    low, high = np.percentile(fg_values, [0.5, 99.5])
    volume = np.clip(volume, low, high)
    fg_values = volume[foreground]
    mean = float(fg_values.mean())
    std = float(fg_values.std())
    if std > 1e-06:
        volume[foreground] = (volume[foreground] - mean) / std
    else:
        volume[foreground] = volume[foreground] - mean
    volume[~foreground] = 0
    return volume.astype(np.float32)


def resize_hw(volume, target_hw, order=1):
    if target_hw is None or tuple(volume.shape[-2:]) == tuple(target_hw):
        return volume.astype(np.float32 if order > 0 else np.uint8)
    zh = target_hw[0] / volume.shape[-2]
    zw = target_hw[1] / volume.shape[-1]
    out = zoom(volume, (1, zh, zw), order=order)
    if order == 0:
        return out.astype(np.uint8)
    return out.astype(np.float32)


def select_fg_group_indices(mask, train=False, total_target_slices=5):
    depth = mask.shape[0]
    nonzero_slice_idx = np.unique(np.nonzero(mask > 0)[0]).tolist()
    nonzero_slice_idx.sort()
    total_nonzero = len(nonzero_slice_idx)
    if total_nonzero >= total_target_slices:
        if train:
            src_start = random.randint(0, total_nonzero - total_target_slices)
        else:
            src_start = (total_nonzero - total_target_slices) // 2
        src_slices = nonzero_slice_idx[src_start:src_start + total_target_slices]
    elif total_nonzero > 0:
        diff = total_target_slices - total_nonzero
        pad_up = [nonzero_slice_idx[0]] * (diff // 2)
        pad_down = [nonzero_slice_idx[-1]] * (diff - diff // 2)
        src_slices = pad_up + nonzero_slice_idx + pad_down
    else:
        center = depth // 2
        src_slices = [int(np.clip(center + offset, 0, depth - 1)) for offset in (-2, -1, 0, 1, 2)]
    src_slices = [int(np.clip(s, 0, depth - 1)) for s in src_slices]
    group1_idx = src_slices[:3]
    group2_idx = src_slices[2:]
    center_idx = src_slices[len(src_slices) // 2]
    return (group1_idx, group2_idx, center_idx, src_slices)


def compute_square_box(mask_2d, box_scale=1.75, min_box_size=48):
    h, w = mask_2d.shape
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        side = max(min_box_size, min(h, w) // 2)
        cy = h / 2.0
        cx = w / 2.0
    else:
        y0 = ys.min()
        y1 = ys.max() + 1
        x0 = xs.min()
        x1 = xs.max() + 1
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        side = max(y1 - y0, x1 - x0, min_box_size)
    side = max(int(math.ceil(side * box_scale)), min_box_size)
    return (cy, cx, side)


def crop_resize_from_box(slice_2d, cy, cx, side, target_hw, order=1):
    h, w = slice_2d.shape
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    y1 = y0 + side
    x1 = x0 + side
    patch = np.zeros((side, side), dtype=slice_2d.dtype)
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, h)
    src_x1 = min(x1, w)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        patch[dst_y0:dst_y1, dst_x0:dst_x1] = slice_2d[src_y0:src_y1, src_x0:src_x1]
    zh = target_hw[0] / patch.shape[0]
    zw = target_hw[1] / patch.shape[1]
    out = zoom(patch, (zh, zw), order=order)
    if order == 0:
        return (out.astype(np.uint8), (y0, x0, y1, x1))
    return (out.astype(np.float32), (y0, x0, y1, x1))


def center_crop_or_pad_last2d(array, target_hw):
    target_h, target_w = target_hw
    src_h, src_w = array.shape[-2:]
    out = np.zeros(array.shape[:-2] + (target_h, target_w), dtype=array.dtype)
    src_h0 = max((src_h - target_h) // 2, 0)
    src_w0 = max((src_w - target_w) // 2, 0)
    dst_h0 = max((target_h - src_h) // 2, 0)
    dst_w0 = max((target_w - src_w) // 2, 0)
    copy_h = min(src_h, target_h)
    copy_w = min(src_w, target_w)
    out[..., dst_h0:dst_h0 + copy_h, dst_w0:dst_w0 + copy_w] = array[..., src_h0:src_h0 + copy_h, src_w0:src_w0 + copy_w]
    return out


def random_rescale_last2d(array, scale):
    zoom_factors = (1,) * (array.ndim - 2) + (scale, scale)
    scaled = zoom(array, zoom_factors, order=1)
    return center_crop_or_pad_last2d(scaled, array.shape[-2:]).astype(np.float32)


def simulate_lowres_last2d(array, scale):
    if scale >= 0.999:
        return array.astype(np.float32)
    zoom_factors = (1,) * (array.ndim - 2) + (scale, scale)
    down = zoom(array, zoom_factors, order=1)
    up_h = array.shape[-2] / down.shape[-2]
    up_w = array.shape[-1] / down.shape[-1]
    up_factors = (1,) * (array.ndim - 2) + (up_h, up_w)
    up = zoom(down, up_factors, order=1)
    return center_crop_or_pad_last2d(up, array.shape[-2:]).astype(np.float32)


def gamma_adjust_last2d(array, gamma, invert=False):
    mins = array.min(axis=(-2, -1), keepdims=True)
    maxs = array.max(axis=(-2, -1), keepdims=True)
    denom = np.maximum(maxs - mins, 1e-06)
    normed = (array - mins) / denom
    if invert:
        normed = 1.0 - normed
    normed = np.clip(normed, 0.0, 1.0) ** gamma
    if invert:
        normed = 1.0 - normed
    return (normed * denom + mins).astype(np.float32)


def crop_modal_slices(modal_arrays, mask_arr, slice_indices, target_hw, box_scale, min_box_size):
    sel_mask = mask_arr[slice_indices]
    union_mask = (sel_mask.max(axis=0) > 0).astype(np.uint8)
    cy, cx, side = compute_square_box(union_mask, box_scale=box_scale, min_box_size=min_box_size)
    cropped = {}
    box = None
    for key in MODALITY_KEYS:
        slices = []
        for slice_idx in slice_indices:
            crop_img, box = crop_resize_from_box(modal_arrays[key][slice_idx], cy, cx, side, target_hw, order=1)
            slices.append(crop_img)
        cropped[key] = np.stack(slices, axis=0).astype(np.float32)
    return (cropped, box, side)


def augment_images(arr):
    if random.random() < 0.5:
        arr = arr[..., ::-1].copy()
    if random.random() < 0.5:
        arr = arr[..., ::-1, :].copy()
    if random.random() < 0.25:
        angle = random.uniform(-15.0, 15.0)
        arr = ndi_rotate(arr, angle=angle, axes=(-2, -1), reshape=False, order=1, mode='nearest')
    if random.random() < 0.25:
        scale = random.uniform(0.8, 1.2)
        arr = random_rescale_last2d(arr, scale)
    if random.random() < 0.1:
        sigma = random.uniform(0.0, 0.03)
        arr = arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    if random.random() < 0.2:
        sigma = random.uniform(0.5, 1.0)
        spatial_sigma = (0,) * (arr.ndim - 2) + (sigma, sigma)
        arr = gaussian_filter(arr, sigma=spatial_sigma).astype(np.float32)
    if random.random() < 0.15:
        arr = arr * np.float32(random.uniform(0.75, 1.25))
    if random.random() < 0.15:
        mean = arr.mean(axis=(-2, -1), keepdims=True)
        arr = (arr - mean) * np.float32(random.uniform(0.75, 1.25)) + mean
    if random.random() < 0.25:
        lowres_scale = random.uniform(0.6, 0.95)
        arr = simulate_lowres_last2d(arr, lowres_scale)
    if random.random() < 0.1:
        arr = gamma_adjust_last2d(arr, gamma=random.uniform(0.7, 1.5), invert=True)
    if random.random() < 0.3:
        arr = gamma_adjust_last2d(arr, gamma=random.uniform(0.7, 1.5), invert=False)
    return arr.astype(np.float32)
