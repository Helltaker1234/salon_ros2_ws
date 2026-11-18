#!/usr/bin/env python3
import os
import sys
import csv
import math
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# sensor_msgs PointField datatype constants
PF_INT8 = 1
PF_UINT8 = 2
PF_INT16 = 3
PF_UINT16 = 4
PF_INT32 = 5
PF_UINT32 = 6
PF_FLOAT32 = 7
PF_FLOAT64 = 8

DTYPE_MAP = {
    PF_INT8: np.int8,
    PF_UINT8: np.uint8,
    PF_INT16: np.int16,
    PF_UINT16: np.uint16,
    PF_INT32: np.int32,
    PF_UINT32: np.uint32,
    PF_FLOAT32: np.float32,
    PF_FLOAT64: np.float64,
}

DIST_COLORS = [
    (0.0, 5.0, '#00ff00', '0–5m'),
    (5.0, 10.0, '#ffff00', '5–10m'),
    (10.0, 15.0, '#ff9900', '10–15m'),
    (15.0, 20.0, '#ff0000', '15–20m'),
    (20.0, 1e9, '#ff00ff', '>20m'),
]


def detect_storage_id(bag_path: str) -> str:
    bag = Path(os.path.expanduser(bag_path))
    if not bag.exists():
        raise FileNotFoundError(f'Bag path not found: {bag}')
    if bag.is_file():
        if bag.suffix == '.mcap':
            return 'mcap'
        if bag.suffix == '.db3':
            return 'sqlite3'
    if any(bag.glob('*.mcap')):
        return 'mcap'
    if any(bag.glob('*.db3')):
        return 'sqlite3'
    raise RuntimeError(f'Could not detect storage backend for {bag}')


def open_reader(bag_path: str):
    storage_id = detect_storage_id(bag_path)
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=os.path.expanduser(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topics, storage_id


def get_topic_choice(topics, preferred_filtered='/livox/lidar_filtered', preferred_raw='/livox/lidar', forced=None):
    if forced:
        if forced not in topics:
            raise RuntimeError(f'Requested point topic {forced} not in bag. Available: {sorted(topics.keys())}')
        return forced, topics[forced]
    if preferred_filtered in topics:
        return preferred_filtered, topics[preferred_filtered]
    if preferred_raw in topics:
        return preferred_raw, topics[preferred_raw]
    raise RuntimeError(
        f'Neither {preferred_filtered} nor {preferred_raw} found in bag. '\
        f'Available topics: {sorted(topics.keys())}'
    )


def deserialize_topic(serialized_msg, type_name):
    msg_type = get_message(type_name)
    return deserialize_message(serialized_msg, msg_type)


def pointcloud2_to_xyz(msg):
    field_map = {f.name: f for f in msg.fields}
    needed = ['x', 'y', 'z']
    for n in needed:
        if n not in field_map:
            raise RuntimeError(f'PointCloud2 missing field: {n}')
    names, formats, offsets = [], [], []
    for n in needed:
        f = field_map[n]
        if f.datatype not in DTYPE_MAP:
            raise RuntimeError(f'Unsupported datatype for field {n}: {f.datatype}')
        names.append(n)
        formats.append(DTYPE_MAP[f.datatype])
        offsets.append(f.offset)
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32, copy=False)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def custommsg_to_xyz(msg):
    if not hasattr(msg, 'points'):
        raise RuntimeError('livox CustomMsg does not have points field')
    if len(msg.points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def extract_xyz(serialized_msg, type_name, z_max=None):
    msg = deserialize_topic(serialized_msg, type_name)
    if type_name == 'sensor_msgs/msg/PointCloud2':
        xyz = pointcloud2_to_xyz(msg)
    elif type_name == 'livox_interfaces/msg/CustomMsg':
        xyz = custommsg_to_xyz(msg)
    else:
        raise RuntimeError(f'Unsupported point topic type: {type_name}')
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    return xyz


def read_sampled_clouds(bag_path, point_topic=None, filtered_topic='/livox/lidar_filtered', raw_topic='/livox/lidar',
                        start_sec=0.0, duration_sec=None, interval_sec=2.0, max_frames=60, z_max=None):
    reader, topics, storage_id = open_reader(bag_path)
    chosen_topic, type_name = get_topic_choice(topics, filtered_topic, raw_topic, forced=point_topic)

    filt = rosbag2_py.StorageFilter(topics=[chosen_topic])
    reader.set_filter(filt)

    start_ns = None
    next_target_ns = None
    end_ns = None
    frames = []
    seen = 0

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        if topic != chosen_topic:
            continue
        seen += 1
        if start_ns is None:
            start_ns = t_ns
            next_target_ns = start_ns + int(start_sec * 1e9)
            if duration_sec is not None:
                end_ns = start_ns + int((start_sec + duration_sec) * 1e9)
        if t_ns < next_target_ns:
            continue
        if end_ns is not None and t_ns > end_ns:
            break
        try:
            xyz = extract_xyz(serialized_msg, type_name, z_max=z_max)
        except Exception as e:
            print(f'[WARN] Failed to parse cloud at t={t_ns}: {e}', file=sys.stderr)
            next_target_ns += int(interval_sec * 1e9)
            continue
        frames.append({
            't_sec': (t_ns - start_ns) / 1e9,
            't_ns': t_ns,
            'xyz': xyz,
            'topic': chosen_topic,
            'type_name': type_name,
        })
        next_target_ns += int(interval_sec * 1e9)
        if len(frames) >= max_frames:
            break
    return frames, {'storage_id': storage_id, 'topic': chosen_topic, 'type_name': type_name, 'seen_msgs': seen, 'topics': topics}


def make_occupancy_metrics(xyz, x_max=25.0, y_abs_max=4.0, lane_half_width=0.75, grid_res=0.2, range_bins=None):
    if range_bins is None:
        range_bins = [0, 5, 10, 15, 20, 25]
    if xyz.shape[0] == 0:
        return {
            'n_points_total': 0,
            'n_points_roi': 0,
            'bin_stats': [],
            'grid': np.zeros((max(1, int(np.ceil(2*lane_half_width/grid_res))), max(1, int(np.ceil(x_max/grid_res)))), dtype=np.int32),
        }

    x = xyz[:, 0]
    y = xyz[:, 1]
    roi_mask = (x >= 0.0) & (x < x_max) & (np.abs(y) <= y_abs_max)
    xyz_roi = xyz[roi_mask]
    x_roi = xyz_roi[:, 0] if xyz_roi.shape[0] else np.array([], dtype=np.float32)
    y_roi = xyz_roi[:, 1] if xyz_roi.shape[0] else np.array([], dtype=np.float32)

    # Central corridor occupancy grid for visualization
    nx = max(1, int(np.ceil(x_max / grid_res)))
    ny = max(1, int(np.ceil((2 * lane_half_width) / grid_res)))
    grid = np.zeros((ny, nx), dtype=np.int32)
    center_mask = (x_roi >= 0.0) & (x_roi < x_max) & (np.abs(y_roi) <= lane_half_width)
    if np.any(center_mask):
        gx = np.floor(x_roi[center_mask] / grid_res).astype(int)
        gy = np.floor((y_roi[center_mask] + lane_half_width) / grid_res).astype(int)
        valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        for cx, cy in zip(gx[valid], gy[valid]):
            grid[cy, cx] += 1

    bin_stats = []
    for lo, hi in zip(range_bins[:-1], range_bins[1:]):
        mask = (x >= lo) & (x < hi) & (np.abs(y) <= lane_half_width)
        pts = xyz[mask]
        n_pts = int(pts.shape[0])
        width_cells = max(1, int(np.ceil((hi - lo) / grid_res)))
        height_cells = ny
        total_cells = width_cells * height_cells
        if n_pts > 0:
            cell_x = np.floor((pts[:, 0] - lo) / grid_res).astype(int)
            cell_y = np.floor((pts[:, 1] + lane_half_width) / grid_res).astype(int)
            valid = (cell_x >= 0) & (cell_x < width_cells) & (cell_y >= 0) & (cell_y < height_cells)
            uniq = set(zip(cell_x[valid].tolist(), cell_y[valid].tolist()))
            occ_cells = len(uniq)
        else:
            occ_cells = 0
        occ_ratio = occ_cells / total_cells if total_cells > 0 else 0.0
        bin_stats.append({
            'range_lo': lo,
            'range_hi': hi,
            'n_points': n_pts,
            'occupied_cells': occ_cells,
            'total_cells': total_cells,
            'occupancy_ratio': occ_ratio,
        })

    return {
        'n_points_total': int(xyz.shape[0]),
        'n_points_roi': int(xyz_roi.shape[0]),
        'bin_stats': bin_stats,
        'grid': grid,
    }


def scatter_by_distance(ax, xyz, x_max, y_abs_max):
    x = xyz[:, 0]
    y = xyz[:, 1]
    ranges = np.linalg.norm(xyz, axis=1)
    roi = (x >= 0.0) & (x < x_max) & (np.abs(y) <= y_abs_max)
    x = x[roi]
    y = y[roi]
    ranges = ranges[roi]
    for lo, hi, color, label in DIST_COLORS:
        mask = (ranges >= lo) & (ranges < hi)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=2.0, c=color, label=label, alpha=0.85)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-y_abs_max, y_abs_max)
    ax.set_xlabel('x forward [m]')
    ax.set_ylabel('y left/right [m]')
    ax.set_title('BEV scatter (forward x, lateral y)')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    # central corridor boundary
    return ax


def draw_panel(out_png, tag, frame_idx, t_sec, xyz, metrics, x_max, y_abs_max, lane_half_width, grid_res):
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    scatter_by_distance(ax1, xyz, x_max, y_abs_max)
    rect = Rectangle((0.0, -lane_half_width), x_max, 2 * lane_half_width,
                     fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
    ax1.add_patch(rect)
    for xline in [0, 5, 10, 15, 20, 25]:
        if xline <= x_max:
            ax1.axvline(xline, color='white', linestyle=':', linewidth=0.8, alpha=0.9)
    ax1.legend(loc='upper right', fontsize=8)

    grid = metrics['grid']
    im = ax2.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[0.0, x_max, -lane_half_width, lane_half_width],
        interpolation='nearest'
    )
    ax2.set_xlabel('x forward [m]')
    ax2.set_ylabel('central corridor y [m]')
    ax2.set_title(f'Central corridor occupancy\n(|y| <= {lane_half_width:.2f} m, grid={grid_res:.2f} m)')
    for xline in [0, 5, 10, 15, 20, 25]:
        if xline <= x_max:
            ax2.axvline(xline, color='white', linestyle=':', linewidth=0.8, alpha=0.9)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='points per cell')

    # add summary text
    lines = [
        f'[{tag}] frame={frame_idx:03d}  t={t_sec:7.2f}s',
        f'total points={metrics["n_points_total"]}   ROI points={metrics["n_points_roi"]}',
        'Central corridor occupancy:',
    ]
    for b in metrics['bin_stats']:
        lines.append(
            f'{int(b["range_lo"]):2d}-{int(b["range_hi"]):2d} m : '
            f'pts={b["n_points"]:5d}, occ={b["occupied_cells"]:4d}/{b["total_cells"]:4d} '
            f'({100.0*b["occupancy_ratio"]:5.1f}%)'
        )
    fig.text(0.12, 0.02, '\n'.join(lines), fontsize=10, family='monospace')
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def summarize(tag, rows, range_bins):
    if not rows:
        print(f'[{tag}] No rows')
        return
    print(f'[{tag}] analyzed frames={len(rows)}')
    for lo, hi in zip(range_bins[:-1], range_bins[1:]):
        pts = [r[f'pts_{int(lo)}_{int(hi)}'] for r in rows]
        occ = [r[f'occ_ratio_{int(lo)}_{int(hi)}'] for r in rows]
        print(
            f'  {int(lo):2d}-{int(hi):2d} m : '
            f'mean points={np.mean(pts):8.1f}   mean occupancy={100*np.mean(occ):6.1f}%'
        )


def process_bag(tag, bag_path, args, out_dir, csv_writer, meta_rows):
    frames, info = read_sampled_clouds(
        bag_path,
        point_topic=args.point_topic,
        filtered_topic=args.filtered_topic,
        raw_topic=args.raw_topic,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        interval_sec=args.interval_sec,
        max_frames=args.max_frames,
        z_max=args.z_max,
    )
    print(f'[{tag}] bag={os.path.expanduser(bag_path)}')
    print(f'[{tag}] storage={info["storage_id"]}')
    print(f'[{tag}] point_topic={info["topic"]} ({info["type_name"]})')
    print(f'[{tag}] seen messages={info["seen_msgs"]}, analyzed frames={len(frames)}')

    rows = []
    for idx, fr in enumerate(frames):
        metrics = make_occupancy_metrics(
            fr['xyz'],
            x_max=args.x_max,
            y_abs_max=args.y_abs_max,
            lane_half_width=args.lane_half_width,
            grid_res=args.grid_res,
            range_bins=args.range_bins,
        )
        out_png = os.path.join(out_dir, f'{idx:04d}_t{fr["t_sec"]:07.2f}s.png')
        draw_panel(out_png, tag, idx, fr['t_sec'], fr['xyz'], metrics,
                   args.x_max, args.y_abs_max, args.lane_half_width, args.grid_res)

        row = {
            'tag': tag,
            'frame_idx': idx,
            't_sec': fr['t_sec'],
            'point_topic': info['topic'],
            'point_type': info['type_name'],
            'n_points_total': metrics['n_points_total'],
            'n_points_roi': metrics['n_points_roi'],
        }
        for b in metrics['bin_stats']:
            lo = int(b['range_lo'])
            hi = int(b['range_hi'])
            row[f'pts_{lo}_{hi}'] = b['n_points']
            row[f'occ_cells_{lo}_{hi}'] = b['occupied_cells']
            row[f'occ_total_{lo}_{hi}'] = b['total_cells']
            row[f'occ_ratio_{lo}_{hi}'] = b['occupancy_ratio']
        csv_writer.writerow(row)
        rows.append(row)

        if idx % 10 == 0:
            msg = f'[{tag}] frame={idx:3d} t={fr["t_sec"]:7.2f}s total={metrics["n_points_total"]:6d} roi={metrics["n_points_roi"]:6d}'
            for b in metrics['bin_stats']:
                lo, hi = int(b['range_lo']), int(b['range_hi'])
                msg += f' | {lo:02d}-{hi:02d}:{100*b["occupancy_ratio"]:4.1f}%'
            print(msg)

    meta_rows.append({
        'tag': tag,
        'bag': os.path.expanduser(bag_path),
        'storage': info['storage_id'],
        'point_topic': info['topic'],
        'point_type': info['type_name'],
        'seen_msgs': info['seen_msgs'],
        'analyzed_frames': len(rows),
    })
    summarize(tag, rows, args.range_bins)
    return rows


def main():
    ap = argparse.ArgumentParser(description='Compare GOOD/BAD LiDAR BEV coverage in central road corridor.')
    ap.add_argument('--good-bag', required=True)
    ap.add_argument('--bad-bag', required=True)
    ap.add_argument('--filtered-topic', default='/livox/lidar_filtered', help='Preferred PointCloud2 topic name')
    ap.add_argument('--raw-topic', default='/livox/lidar', help='Fallback raw Livox CustomMsg topic name')
    ap.add_argument('--point-topic', default=None, help='Force this point topic instead of auto-choosing filtered/raw')
    ap.add_argument('--interval-sec', type=float, default=2.0)
    ap.add_argument('--max-frames', type=int, default=60)
    ap.add_argument('--start-sec', type=float, default=0.0)
    ap.add_argument('--duration-sec', type=float, default=None)
    ap.add_argument('--x-max', type=float, default=25.0, help='Forward range shown in BEV [m]')
    ap.add_argument('--y-abs-max', type=float, default=4.0, help='Full BEV lateral extent +-Y [m]')
    ap.add_argument('--lane-half-width', type=float, default=0.75, help='Central corridor half-width [m]')
    ap.add_argument('--grid-res', type=float, default=0.2, help='Occupancy grid resolution [m]')
    ap.add_argument('--range-bins', type=str, default='0,5,10,15,20,25', help='Comma-separated forward bins [m]')
    ap.add_argument('--z-max', type=float, default=None, help='Optional z<=z_max filter after conversion; omit to match your custom_to_pointcloud2 output')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    args.good_bag = os.path.expanduser(args.good_bag)
    args.bad_bag = os.path.expanduser(args.bad_bag)
    args.output = os.path.expanduser(args.output)
    args.range_bins = [float(x.strip()) for x in args.range_bins.split(',') if x.strip()]
    if len(args.range_bins) < 2:
        raise ValueError('--range-bins must contain at least two numbers')

    print('=== BEV LiDAR coverage comparison ===')
    print(f'good bag          : {args.good_bag}')
    print(f'bad bag           : {args.bad_bag}')
    print(f'preferred filtered: {args.filtered_topic}')
    print(f'fallback raw      : {args.raw_topic}')
    print(f'forced point topic: {args.point_topic}')
    print(f'interval_sec      : {args.interval_sec}')
    print(f'max_frames        : {args.max_frames}')
    print(f'start_sec         : {args.start_sec}')
    print(f'duration_sec      : {args.duration_sec}')
    print(f'forward x_max     : {args.x_max}')
    print(f'full lateral      : ±{args.y_abs_max}')
    print(f'central halfwidth : {args.lane_half_width}')
    print(f'grid_res          : {args.grid_res}')
    print(f'range_bins        : {args.range_bins}')
    print(f'z_max filter      : {args.z_max}')
    print('NOTE: If a bag has /livox/lidar_filtered it will use that first. If not, it will read /livox/lidar and convert the Livox CustomMsg internally, which is equivalent in spirit to your livox_custom_to_pointcloud2 alias. No extra z-filter is applied unless you pass --z-max.')

    ensure_dir(args.output)
    good_dir = os.path.join(args.output, 'good')
    bad_dir = os.path.join(args.output, 'bad')
    ensure_dir(good_dir)
    ensure_dir(bad_dir)

    csv_path = os.path.join(args.output, 'per_frame.csv')
    meta_path = os.path.join(args.output, 'summary.csv')

    # prepare CSV header
    fieldnames = ['tag', 'frame_idx', 't_sec', 'point_topic', 'point_type', 'n_points_total', 'n_points_roi']
    for lo, hi in zip(args.range_bins[:-1], args.range_bins[1:]):
        lo_i, hi_i = int(lo), int(hi)
        fieldnames += [
            f'pts_{lo_i}_{hi_i}',
            f'occ_cells_{lo_i}_{hi_i}',
            f'occ_total_{lo_i}_{hi_i}',
            f'occ_ratio_{lo_i}_{hi_i}',
        ]

    meta_rows = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        good_rows = process_bag('good', args.good_bag, args, good_dir, writer, meta_rows)
        bad_rows = process_bag('bad', args.bad_bag, args, bad_dir, writer, meta_rows)

    with open(meta_path, 'w', newline='') as f:
        fieldnames_meta = ['tag', 'bag', 'storage', 'point_topic', 'point_type', 'seen_msgs', 'analyzed_frames']
        # aggregate metrics per bin
        for lo, hi in zip(args.range_bins[:-1], args.range_bins[1:]):
            lo_i, hi_i = int(lo), int(hi)
            fieldnames_meta += [f'mean_pts_{lo_i}_{hi_i}', f'mean_occ_ratio_{lo_i}_{hi_i}']
        writer = csv.DictWriter(f, fieldnames=fieldnames_meta)
        writer.writeheader()
        by_tag = {'good': good_rows, 'bad': bad_rows}
        for m in meta_rows:
            rows = by_tag[m['tag']]
            out = dict(m)
            for lo, hi in zip(args.range_bins[:-1], args.range_bins[1:]):
                lo_i, hi_i = int(lo), int(hi)
                pts = [r[f'pts_{lo_i}_{hi_i}'] for r in rows] if rows else [0]
                occ = [r[f'occ_ratio_{lo_i}_{hi_i}'] for r in rows] if rows else [0.0]
                out[f'mean_pts_{lo_i}_{hi_i}'] = float(np.mean(pts))
                out[f'mean_occ_ratio_{lo_i}_{hi_i}'] = float(np.mean(occ))
            writer.writerow(out)

    def mean_metric(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    print('\n=== BAD - GOOD (mean occupancy ratio, percentage points) ===')
    for lo, hi in zip(args.range_bins[:-1], args.range_bins[1:]):
        lo_i, hi_i = int(lo), int(hi)
        delta = 100.0 * (mean_metric(bad_rows, f'occ_ratio_{lo_i}_{hi_i}') - mean_metric(good_rows, f'occ_ratio_{lo_i}_{hi_i}'))
        dpts = mean_metric(bad_rows, f'pts_{lo_i}_{hi_i}') - mean_metric(good_rows, f'pts_{lo_i}_{hi_i}')
        print(f'  {lo_i:2d}-{hi_i:2d} m : occupancy {delta:+6.1f} pp   mean points {dpts:+8.1f}')

    print(f'\nSaved results: {args.output}')
    print(f'  good PNGs : {good_dir}')
    print(f'  bad PNGs  : {bad_dir}')
    print(f'  per-frame : {csv_path}')
    print(f'  summary   : {meta_path}')
    print('\nHow to interpret:')
    print('  1) Look at the RIGHT panel: central corridor occupancy. White/bright cells = many points; dark gaps = weak or missing coverage.')
    print('  2) If BAD shows much lower occupancy in 10–15 m, 15–20 m, 20–25 m than GOOD, that supports the \"far-road center coverage hole\" hypothesis.')
    print('  3) If occupancy is similar but your costmap still differs, then look next at Fast-LIVO pose / motion distortion / map accumulation instead.')


if __name__ == '__main__':
    main()
