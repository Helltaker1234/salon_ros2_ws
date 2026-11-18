#!/usr/bin/env python3
"""Compare illumination sensitivity of the existing DINOv2 + VLAD pipeline on two ROS 2 bags.

This script intentionally reuses physics_atv_visual_mapping.image_pipeline so the
DINO model, layer, facet, input size, FP16 setting, and VLAD centers are exactly
the same as the mapping pipeline.

Outputs:
  <output>/per_frame.csv
  <output>/summary.csv
  <output>/good/*.png
  <output>/bad/*.png

Each PNG contains:
  RGB | luminance map | min VLAD residual | residual-threshold overlay

The important value is min(VLAD residual over K clusters). With the current
config residual_max=19 and uncertainty_threshold=0.90, the threshold is 17.1.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# If this file is placed at:
#   physics_atv_visual_mapping/scripts/tools/compare_dino_lighting_bags.py
# add the package source root to PYTHONPATH automatically.
THIS_FILE = Path(__file__).resolve()
possible_pkg_root = THIS_FILE.parents[2] if len(THIS_FILE.parents) >= 3 else None
if possible_pkg_root is not None and (possible_pkg_root / 'physics_atv_visual_mapping').is_dir():
    sys.path.insert(0, str(possible_pkg_root))

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline


DEFAULT_CONFIG_REL = Path('config/ros/dino_vlad_rough_vits14_252x448_4bags.yaml')


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compare lighting sensitivity of DINOv2+VLAD on GOOD/BAD ROS2 bags.',
    )
    p.add_argument('--good-bag', required=True, help='Path to the good ROS2 bag directory')
    p.add_argument('--bad-bag', required=True, help='Path to the bad ROS2 bag directory')
    p.add_argument('--output', default='dino_lighting_compare', help='Output directory')

    # If omitted, try to resolve relative to this script's package root.
    p.add_argument('--config', default='', help='DINO/VLAD YAML config path')
    p.add_argument('--centers', default='', help='Override VLAD center directory (contains c_centers.pt)')
    p.add_argument('--topic', default='', help='Override image topic; otherwise use YAML image.image_topic')

    p.add_argument('--interval-sec', type=float, default=0.5, help='Minimum interval between analyzed frames')
    p.add_argument('--max-images', type=int, default=80, help='Max analyzed frames per bag; 0 = unlimited')
    p.add_argument('--good-start-sec', type=float, default=0.0, help='Skip this many seconds from GOOD bag image-topic start')
    p.add_argument('--bad-start-sec', type=float, default=0.0, help='Skip this many seconds from BAD bag image-topic start')
    p.add_argument('--duration-sec', type=float, default=0.0, help='Analyze this many seconds after start; 0 = until end')

    p.add_argument('--residual-max', type=float, default=19.0, help='Same residual_max used by costmap logic')
    p.add_argument('--uncertainty-threshold', type=float, default=0.90, help='Normalized threshold; raw threshold=residual_max*this')

    # Optional normalized ROI in original image coordinates. Useful for isolating
    # the distant road area. x/y are 0..1 fractions of image width/height.
    p.add_argument(
        '--roi', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'),
        help='Optional normalized ROI (0..1), e.g. --roi 0.25 0.20 0.75 0.65',
    )
    p.add_argument('--save-every', type=int, default=1, help='Save panel PNG every N analyzed frames')
    return p.parse_args()


def detect_storage_id(bag_dir: str) -> str:
    metadata_path = Path(bag_dir).expanduser().resolve() / 'metadata.yaml'
    if not metadata_path.is_file():
        raise RuntimeError(f'bag metadata not found: {metadata_path}')
    text = metadata_path.read_text(encoding='utf-8')
    m = re.search(r'^\s*storage_identifier:\s*(\S+)\s*$', text, flags=re.MULTILINE)
    if m is None:
        raise RuntimeError(f'storage_identifier not found in {metadata_path}')
    return m.group(1)


def decode_raw_image(msg):
    """Return BGR uint8 image from sensor_msgs/msg/Image."""
    encoding = msg.encoding.lower()
    info = {
        'bgr8': (3, None),
        'rgb8': (3, cv2.COLOR_RGB2BGR),
        'bgra8': (4, cv2.COLOR_BGRA2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
        'mono8': (1, cv2.COLOR_GRAY2BGR),
        '8uc1': (1, cv2.COLOR_GRAY2BGR),
        '8uc3': (3, None),
        '8uc4': (4, cv2.COLOR_BGRA2BGR),
    }
    if encoding not in info:
        raise RuntimeError(f'unsupported raw image encoding: {msg.encoding}')

    channels, conversion = info[encoding]
    row_bytes = int(msg.step)
    useful = int(msg.width) * channels
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height) * row_bytes
    if raw.size < expected:
        raise RuntimeError(f'truncated image: {raw.size} < {expected}')

    img = raw[:expected].reshape(int(msg.height), row_bytes)
    img = img[:, :useful].reshape(int(msg.height), int(msg.width), channels)
    if conversion is not None:
        img = cv2.cvtColor(img, conversion)
    return img


def decode_image(msg, message_type):
    if message_type == 'sensor_msgs/msg/CompressedImage':
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if message_type == 'sensor_msgs/msg/Image':
        return decode_raw_image(msg)
    raise RuntimeError(f'unsupported image type: {message_type}')


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def masked_values(arr, mask):
    if mask is None:
        return arr.reshape(-1)
    vals = arr[mask]
    return vals.reshape(-1)


def make_roi_mask(h, w, roi):
    if roi is None:
        return None, None
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError('--roi must satisfy 0<=X0<X1<=1 and 0<=Y0<Y1<=1')
    ix0 = int(round(x0 * w))
    ix1 = int(round(x1 * w))
    iy0 = int(round(y0 * h))
    iy1 = int(round(y1 * h))
    ix0 = np.clip(ix0, 0, w - 1)
    ix1 = np.clip(ix1, ix0 + 1, w)
    iy0 = np.clip(iy0, 0, h - 1)
    iy1 = np.clip(iy1, iy0 + 1, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[iy0:iy1, ix0:ix1] = True
    return mask, (ix0, iy0, ix1, iy1)


def to_heatmap(data, vmin, vmax):
    x = np.asarray(data, dtype=np.float32)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    x8 = (x * 255.0).astype(np.uint8)
    return cv2.applyColorMap(x8, cv2.COLORMAP_TURBO)


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_panel(out_path, bgr, luminance_feat, residual_min, high_mask, raw_thr,
               roi_original_px, row):
    h0, w0 = bgr.shape[:2]
    hf, wf = residual_min.shape

    # Same spatial resolution for all four panels.
    panel_w = 640
    panel_h = max(1, int(round(h0 * panel_w / w0)))

    rgb_panel = cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    if roi_original_px is not None:
        x0, y0, x1, y1 = roi_original_px
        sx = panel_w / w0
        sy = panel_h / h0
        cv2.rectangle(
            rgb_panel,
            (int(x0 * sx), int(y0 * sy)),
            (int(x1 * sx), int(y1 * sy)),
            (0, 255, 255), 2,
        )
    rgb_panel = put_label(rgb_panel, 'RGB (yellow rectangle = ROI)')

    lum_hm = to_heatmap(luminance_feat, 0.0, 1.0)
    lum_hm = cv2.resize(lum_hm, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    lum_hm = put_label(lum_hm, 'Luminance (DINO patch grid)')

    # Keep the same residual scale for every frame so GOOD/BAD are comparable.
    res_vmax = max(raw_thr * 1.35, raw_thr + 2.0)
    res_hm = to_heatmap(residual_min, 0.0, res_vmax)
    res_hm = cv2.resize(res_hm, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    res_hm = put_label(res_hm, f'min VLAD residual (threshold={raw_thr:.2f})')

    mask_big = cv2.resize(high_mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = bgr.copy()
    # Red overlay for high residual cells.
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[mask_big] = (0.40 * overlay[mask_big] + 0.60 * red[mask_big]).astype(np.uint8)
    overlay = cv2.resize(overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    if roi_original_px is not None:
        x0, y0, x1, y1 = roi_original_px
        sx = panel_w / w0
        sy = panel_h / h0
        cv2.rectangle(overlay, (int(x0*sx), int(y0*sy)), (int(x1*sx), int(y1*sy)), (0,255,255), 2)
    overlay = put_label(
        overlay,
        'High residual overlay | ratio={:.1f}% | corr(light,res)={:.2f} | corr(edge,res)={:.2f}'.format(
            100.0 * row['high_ratio'], row['corr_luma_res'], row['corr_local_contrast_res']
        )
    )

    panel = np.vstack([np.hstack([rgb_panel, lum_hm]), np.hstack([res_hm, overlay])])
    cv2.imwrite(str(out_path), panel)


def summarize_rows(label, rows):
    if not rows:
        return {}

    def mean_key(k):
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float('nan')

    # Weight patch-level ratios approximately equally per frame. Since DINO grid
    # shape is fixed, this equals the global patch ratio for fixed input size.
    return {
        'label': label,
        'frames': len(rows),
        'mean_luminance': mean_key('mean_luminance'),
        'mean_local_contrast': mean_key('mean_local_contrast'),
        'mean_min_residual': mean_key('mean_min_residual'),
        'mean_p95_residual': mean_key('p95_residual'),
        'mean_high_ratio': mean_key('high_ratio'),
        'mean_corr_luma_res': mean_key('corr_luma_res'),
        'mean_corr_extreme_res': mean_key('corr_extreme_res'),
        'mean_corr_local_contrast_res': mean_key('corr_local_contrast_res'),
        'mean_bright_patch_high_ratio': mean_key('bright_patch_high_ratio'),
        'mean_dark_patch_high_ratio': mean_key('dark_patch_high_ratio'),
    }


def conditional_high_ratio(residual, condition, raw_thr):
    cond = np.asarray(condition, dtype=bool)
    if cond.sum() == 0:
        return float('nan')
    return float(np.mean(residual[cond] >= raw_thr))


def process_bag(label, bag_path, topic, pipeline, intrinsics, out_dir,
                interval_sec, max_images, start_sec, duration_sec,
                residual_max, uncertainty_threshold, roi, save_every):
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_id = detect_storage_id(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(bag_path).expanduser().resolve()), storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        available = '\n'.join(sorted(topic_types))
        raise RuntimeError(f'[{label}] topic not found: {topic}\navailable topics:\n{available}')
    actual_type = topic_types[topic]
    if actual_type not in {'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'}:
        raise RuntimeError(f'[{label}] {topic} has unsupported type {actual_type}')
    msg_type = get_message(actual_type)

    interval_ns = int(interval_sec * 1e9)
    start_ns_rel = int(start_sec * 1e9)
    duration_ns = int(duration_sec * 1e9) if duration_sec > 0 else None

    first_topic_time_ns = None
    last_saved_ns = None
    analyzed = 0
    seen = 0
    rows = []
    raw_thr = residual_max * uncertainty_threshold

    print(f'\n[{label}] bag={bag_path}')
    print(f'[{label}] topic={topic} ({actual_type}) storage={storage_id}')

    while reader.has_next():
        bag_topic, serialized, bag_time_ns = reader.read_next()
        if bag_topic != topic:
            continue
        seen += 1
        if first_topic_time_ns is None:
            first_topic_time_ns = bag_time_ns

        rel_ns = bag_time_ns - first_topic_time_ns
        if rel_ns < start_ns_rel:
            continue
        if duration_ns is not None and rel_ns >= start_ns_rel + duration_ns:
            break
        if last_saved_ns is not None and bag_time_ns - last_saved_ns < interval_ns:
            continue

        msg = deserialize_message(serialized, msg_type)
        bgr = decode_image(msg, actual_type)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2)

        with torch.inference_mode():
            feat, _ = pipeline.run(img_t, intrinsics.unsqueeze(0))

        # Existing DINO+VLAD pipeline output is [B, K, Hf, Wf], where each K
        # value is abs(residual).sum(desc_dim) to one VLAD center.
        if feat.ndim != 4:
            raise RuntimeError(f'Unexpected pipeline output shape: {tuple(feat.shape)}')
        if feat.shape[1] < 2:
            raise RuntimeError(
                f'Expected VLAD residual channels, got shape {tuple(feat.shape)}. '
                'Check that YAML contains both dino and vlad blocks.'
            )

        vlad = feat[0].detach().float().cpu().numpy()  # K,Hf,Wf
        residual_min = np.min(vlad, axis=0)            # Hf,Wf
        hf, wf = residual_min.shape
        high_mask = residual_min >= raw_thr

        # Luminance in RGB [0,1]. Downsample to exactly the DINO/VLAD patch grid.
        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        luminance_feat = cv2.resize(lum, (wf, hf), interpolation=cv2.INTER_AREA)

        # Local illumination discontinuity: useful for sun/shadow boundaries.
        # Sigma is in original-image pixels, then downsampled to patch grid.
        blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=15.0, sigmaY=15.0)
        local_contrast = np.abs(lum - blur)
        local_contrast_feat = cv2.resize(local_contrast, (wf, hf), interpolation=cv2.INTER_AREA)

        roi_mask_feat, _ = make_roi_mask(hf, wf, roi)
        _, roi_px_original = make_roi_mask(bgr.shape[0], bgr.shape[1], roi)

        rv = masked_values(residual_min, roi_mask_feat)
        lv = masked_values(luminance_feat, roi_mask_feat)
        cv = masked_values(local_contrast_feat, roi_mask_feat)
        hv = masked_values(high_mask.astype(np.float32), roi_mask_feat)

        # Illumination extremeness captures both very dark and very bright patches.
        extreme = np.abs(lv - 0.5)

        # Conditions are evaluated inside ROI if ROI is supplied.
        bright = lv >= 0.75
        dark = lv <= 0.25

        row = {
            'label': label,
            'frame_index': analyzed,
            'bag_time_ns': bag_time_ns,
            'rel_time_sec': rel_ns / 1e9,
            'image_width': bgr.shape[1],
            'image_height': bgr.shape[0],
            'feature_width': wf,
            'feature_height': hf,
            'mean_luminance': float(np.mean(lv)),
            'std_luminance': float(np.std(lv)),
            'mean_local_contrast': float(np.mean(cv)),
            'mean_min_residual': float(np.mean(rv)),
            'median_residual': float(np.median(rv)),
            'p95_residual': float(np.percentile(rv, 95)),
            'max_residual': float(np.max(rv)),
            'high_ratio': float(np.mean(hv)),
            'corr_luma_res': safe_corr(lv, rv),
            'corr_extreme_res': safe_corr(extreme, rv),
            'corr_local_contrast_res': safe_corr(cv, rv),
            'bright_patch_ratio': float(np.mean(bright)),
            'dark_patch_ratio': float(np.mean(dark)),
            'bright_patch_high_ratio': conditional_high_ratio(rv, bright, raw_thr),
            'dark_patch_high_ratio': conditional_high_ratio(rv, dark, raw_thr),
        }
        rows.append(row)

        if save_every > 0 and analyzed % save_every == 0:
            panel_name = f'{analyzed:04d}_t{rel_ns/1e9:08.2f}s.png'
            save_panel(
                out_dir / panel_name,
                bgr,
                luminance_feat,
                residual_min,
                high_mask,
                raw_thr,
                roi_px_original,
                row,
            )

        if analyzed % 10 == 0:
            print(
                f'[{label}] frame={analyzed:4d} t={rel_ns/1e9:7.2f}s '
                f'luma={row["mean_luminance"]:.3f} '
                f'res_mean={row["mean_min_residual"]:.2f} '
                f'res_p95={row["p95_residual"]:.2f} '
                f'high={100*row["high_ratio"]:.1f}% '
                f'corr_edge={row["corr_local_contrast_res"]:.2f}'
            )

        analyzed += 1
        last_saved_ns = bag_time_ns
        if max_images > 0 and analyzed >= max_images:
            break

    print(f'[{label}] seen image messages={seen}, analyzed={analyzed}')
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits=3):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(v):
        return 'nan'
    return f'{v:.{digits}f}'


def resolve_config_and_centers(args):
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        # Works when script is placed under package/scripts/tools.
        package_root = possible_pkg_root
        if package_root is None:
            raise RuntimeError('--config is required when package root cannot be inferred')
        config_path = (package_root / DEFAULT_CONFIG_REL).resolve()
    if not config_path.is_file():
        raise RuntimeError(f'config not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.topic:
        config['image']['image_topic'] = args.topic

    # Override center path explicitly if requested.
    vlad_block = next((b for b in config['image_processing'] if b['type'] == 'vlad'), None)
    if vlad_block is None:
        raise RuntimeError('YAML has no vlad block')

    if args.centers:
        vlad_block['args']['cache_dir'] = str(Path(args.centers).expanduser().resolve())
    else:
        configured = Path(vlad_block['args']['cache_dir']).expanduser()
        if not (configured / 'c_centers.pt').is_file():
            # YAML in the provided source points to an installed absolute path.
            # Fall back to the same center folder bundled in source/data.
            candidate = config_path.parents[2] / 'data' / 'dino_clusters' / configured.name
            if (candidate / 'c_centers.pt').is_file():
                print(f'[INFO] VLAD center path from YAML not found: {configured}')
                print(f'[INFO] Falling back to source center path: {candidate}')
                vlad_block['args']['cache_dir'] = str(candidate.resolve())

    center_dir = Path(vlad_block['args']['cache_dir']).expanduser().resolve()
    if not (center_dir / 'c_centers.pt').is_file():
        raise RuntimeError(
            'VLAD center file not found.\n'
            f'  expected: {center_dir / "c_centers.pt"}\n'
            'Pass --centers <directory-containing-c_centers.pt>.'
        )

    return config_path, config, center_dir


def main():
    args = parse_args()
    if args.interval_sec <= 0:
        raise ValueError('--interval-sec must be > 0')
    if args.max_images < 0:
        raise ValueError('--max-images must be >= 0')
    if args.save_every < 0:
        raise ValueError('--save-every must be >= 0')
    if args.residual_max <= 0:
        raise ValueError('--residual-max must be > 0')

    config_path, config, center_dir = resolve_config_and_centers(args)
    topic = config['image']['image_topic']
    device = config.get('device', 'cuda')

    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'YAML device={device}, but CUDA is unavailable. '
            'This repository VLAD loader currently uses .cuda(), so run on the CUDA machine used for mapping.'
        )

    if args.roi is not None:
        # Validate once before expensive model loading.
        make_roi_mask(100, 100, args.roi)

    raw_thr = args.residual_max * args.uncertainty_threshold
    print('=== Configuration ===')
    print(f'config            : {config_path}')
    print(f'image topic       : {topic}')
    print(f'VLAD centers      : {center_dir}')
    print(f'device            : {device}')
    print(f'residual_max      : {args.residual_max}')
    print(f'uncertainty_thr   : {args.uncertainty_threshold}')
    print(f'RAW threshold     : {raw_thr:.3f}')
    print(f'ROI               : {args.roi if args.roi is not None else "FULL IMAGE"}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Exactly the same DINO + VLAD blocks as dino_localmapping.py.
    pipeline = setup_image_pipeline(config)

    if 'rect' in topic:
        K = config['intrinsics']['P']
    else:
        K = config['intrinsics']['K']
    intrinsics = torch.tensor(K, device=device).reshape(3, 3).float()

    out_root = Path(args.output).expanduser().resolve()
    good_out = out_root / 'good'
    bad_out = out_root / 'bad'
    out_root.mkdir(parents=True, exist_ok=True)

    good_rows = process_bag(
        'good', args.good_bag, topic, pipeline, intrinsics, good_out,
        args.interval_sec, args.max_images, args.good_start_sec, args.duration_sec,
        args.residual_max, args.uncertainty_threshold, args.roi, args.save_every,
    )
    bad_rows = process_bag(
        'bad', args.bad_bag, topic, pipeline, intrinsics, bad_out,
        args.interval_sec, args.max_images, args.bad_start_sec, args.duration_sec,
        args.residual_max, args.uncertainty_threshold, args.roi, args.save_every,
    )

    all_rows = good_rows + bad_rows
    write_csv(out_root / 'per_frame.csv', all_rows)

    summaries = [summarize_rows('good', good_rows), summarize_rows('bad', bad_rows)]
    summaries = [s for s in summaries if s]
    write_csv(out_root / 'summary.csv', summaries)

    print('\n=== SUMMARY ===')
    print(f'raw residual threshold = {raw_thr:.3f}')
    for s in summaries:
        print(
            '[{label}] frames={frames} luma={luma} localContrast={lc} '
            'meanRes={mr} p95Res={p95} high={high}% '
            'corr(luma,res)={cl} corr(extreme,res)={ce} corr(edge,res)={cc} '
            'P(high|bright)={hb}% P(high|dark)={hd}%'.format(
                label=s['label'], frames=s['frames'],
                luma=fmt(s['mean_luminance']), lc=fmt(s['mean_local_contrast']),
                mr=fmt(s['mean_min_residual'], 2), p95=fmt(s['mean_p95_residual'], 2),
                high=fmt(100*s['mean_high_ratio'], 1),
                cl=fmt(s['mean_corr_luma_res'], 2), ce=fmt(s['mean_corr_extreme_res'], 2),
                cc=fmt(s['mean_corr_local_contrast_res'], 2),
                hb=fmt(100*s['mean_bright_patch_high_ratio'], 1),
                hd=fmt(100*s['mean_dark_patch_high_ratio'], 1),
            )
        )

    if len(summaries) == 2:
        g, b = summaries
        print('\n=== BAD - GOOD ===')
        print(f'high residual ratio : {100*(b["mean_high_ratio"]-g["mean_high_ratio"]):+.1f} percentage points')
        print(f'mean residual       : {b["mean_min_residual"]-g["mean_min_residual"]:+.2f}')
        print(f'p95 residual        : {b["mean_p95_residual"]-g["mean_p95_residual"]:+.2f}')
        print(f'local contrast      : {b["mean_local_contrast"]-g["mean_local_contrast"]:+.3f}')

    print(f'\nSaved results: {out_root}')
    print(f'  panels     : {good_out} , {bad_out}')
    print(f'  per-frame  : {out_root / "per_frame.csv"}')
    print(f'  summary    : {out_root / "summary.csv"}')
    print('\nInterpretation:')
    print('  1) Check whether red high-residual patches spatially coincide with sun/shadow regions.')
    print('  2) Compare BAD vs GOOD high residual ratio and p95 residual.')
    print('  3) Positive corr(edge,res) and high P(high|bright/dark) support illumination sensitivity.')
    print('  4) This is an image-space test. To prove a 10-20 m distance effect, next add LiDAR projection/distance bins.')


if __name__ == '__main__':
    main()
