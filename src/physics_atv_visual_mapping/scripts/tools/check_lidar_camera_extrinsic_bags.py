#!/usr/bin/env python3
"""
Offline LiDAR -> RGB projection checker for the user's DINO mapping pipeline.

Purpose
-------
Verify that the LiDAR-camera relative extrinsic and camera intrinsics used by
`dino_localmapping.py` still project LiDAR points onto the correct RGB pixels.
This isolates camera/LiDAR geometry from Fast-LIVO, GP cost, and DINO/VLAD.

It intentionally uses the same convention as dino_localmapping.py:
    P = K @ T_camera_lidar
    uv ~ P @ [x_lidar, y_lidar, z_lidar, 1]^T

Outputs per bag
---------------
  <output>/good/*.png
  <output>/bad/*.png
  <output>/per_frame.csv
  <output>/summary.csv

Each PNG shows the original RGB image and a LiDAR projection overlay. Projected
points are grouped by LiDAR range: 0-5, 5-10, 10-15, 15-20, >20 m.

The script supports both:
  * sensor_msgs/msg/PointCloud2
  * livox_interfaces/msg/CustomMsg
For raw Livox CustomMsg it can apply the same z <= z_max filter as
pcl_z_filter/z_filter_livox_node.py.
"""

import argparse
import csv
import math
import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compare LiDAR-camera projection in GOOD/BAD ROS2 bags.',
    )
    p.add_argument('--good-bag', required=True)
    p.add_argument('--bad-bag', required=True)
    p.add_argument('--output', default='extrinsic_projection_compare')
    p.add_argument('--config', default='', help='Mapping YAML. If omitted, resolve from this workspace source tree.')
    p.add_argument('--image-topic', default='', help='Override RGB image topic')
    p.add_argument('--camera-info-topic', default='', help='Override CameraInfo topic')
    p.add_argument('--point-topic', default='', help='Override point topic. If omitted, use YAML topic, else fallback to /livox/lidar')
    p.add_argument('--interval-sec', type=float, default=2.0, help='Interval between checked pointcloud frames')
    p.add_argument('--max-images', type=int, default=60, help='Max saved overlays per bag; 0 = unlimited')
    p.add_argument('--good-start-sec', type=float, default=0.0)
    p.add_argument('--bad-start-sec', type=float, default=0.0)
    p.add_argument('--duration-sec', type=float, default=0.0, help='0 = until end')
    p.add_argument('--max-image-dt', type=float, default=-1.0, help='Max image-pointcloud stamp difference; <0 uses YAML sync.max_image_dt')
    p.add_argument('--z-max', type=float, default=2.0, help='Apply z <= z_max to raw Livox CustomMsg (same as z_filter_livox_node)')
    p.add_argument('--no-z-filter', action='store_true', help='Do not apply z_max to raw Livox points')
    p.add_argument('--max-range', type=float, default=30.0, help='Do not draw points farther than this range; <=0 disables')
    p.add_argument('--max-draw-points', type=int, default=12000, help='Decimate overlay to at most this many projected points')
    p.add_argument('--point-radius', type=int, default=1)
    p.add_argument('--image-cache', type=int, default=120, help='Number of recent image messages kept for nearest-stamp matching')
    return p.parse_args()


def resolve_config(cli_value: str) -> Path:
    if cli_value:
        path = Path(cli_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # Expected placement:
    # ~/salon_ros2_ws/src/physics_atv_visual_mapping/scripts/tools/this_file.py
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / 'config/ros/dino_vlad_rough_vits14_252x448_4bags.yaml'
        if len(here.parents) >= 3 else Path('/nonexistent'),
        Path.cwd() / 'src/physics_atv_visual_mapping/config/ros/dino_vlad_rough_vits14_252x448_4bags.yaml',
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError('Could not auto-find mapping YAML; pass --config /path/to/yaml')


def detect_storage_id(bag_dir: str) -> str:
    metadata_path = Path(bag_dir).expanduser().resolve() / 'metadata.yaml'
    if not metadata_path.is_file():
        raise RuntimeError(f'bag metadata not found: {metadata_path}')
    text = metadata_path.read_text(encoding='utf-8')
    m = re.search(r'^\s*storage_identifier:\s*(\S+)\s*$', text, flags=re.MULTILINE)
    if m is None:
        raise RuntimeError(f'storage_identifier not found in {metadata_path}')
    return m.group(1)


def stamp_ns(msg) -> int:
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def quaternion_xyzw_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def pose_to_htm(p, q):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_rotation_matrix(q)
    T[:3, 3] = np.asarray(p, dtype=np.float64)
    return T


def decode_raw_image(msg):
    enc = msg.encoding.lower()
    mapping = {
        'bgr8': (3, None),
        'rgb8': (3, cv2.COLOR_RGB2BGR),
        'bgra8': (4, cv2.COLOR_BGRA2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
        'mono8': (1, cv2.COLOR_GRAY2BGR),
        '8uc1': (1, cv2.COLOR_GRAY2BGR),
        '8uc3': (3, None),
        '8uc4': (4, cv2.COLOR_BGRA2BGR),
    }
    if enc not in mapping:
        raise RuntimeError(f'unsupported image encoding: {msg.encoding}')
    channels, conversion = mapping[enc]
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


def decode_image(msg, msg_type: str):
    if msg_type == 'sensor_msgs/msg/Image':
        return decode_raw_image(msg)
    if msg_type == 'sensor_msgs/msg/CompressedImage':
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('cv2.imdecode failed')
        return img
    raise RuntimeError(f'unsupported image type: {msg_type}')


def pointcloud2_xyz(msg):
    # Parse x/y/z by field offsets without sensor_msgs_py dependency.
    fields = {f.name: f for f in msg.fields}
    for name in ('x', 'y', 'z'):
        if name not in fields:
            raise RuntimeError(f'PointCloud2 missing field {name}')
        if int(fields[name].datatype) != 7:  # PointField.FLOAT32
            raise RuntimeError(f'PointCloud2 {name} must be FLOAT32 for this checker')

    n = int(msg.width) * int(msg.height)
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [endian+'f4', endian+'f4', endian+'f4'],
        'offsets': [int(fields['x'].offset), int(fields['y'].offset), int(fields['z'].offset)],
        'itemsize': int(msg.point_step),
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = arr['x']
    xyz[:, 1] = arr['y']
    xyz[:, 2] = arr['z']
    return xyz


def decode_points(msg, msg_type: str, z_max: float, apply_z_filter: bool):
    if msg_type == 'sensor_msgs/msg/PointCloud2':
        xyz = pointcloud2_xyz(msg)
        # Filtered topic already had z_max applied. If the user explicitly points
        # this script at raw PointCloud2, matching z_max is still useful.
        if apply_z_filter:
            xyz = xyz[xyz[:, 2] <= z_max]
        return xyz

    if msg_type == 'livox_interfaces/msg/CustomMsg':
        if len(msg.points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        xyz = np.empty((len(msg.points), 3), dtype=np.float32)
        xyz[:, 0] = [p.x for p in msg.points]
        xyz[:, 1] = [p.y for p in msg.points]
        xyz[:, 2] = [p.z for p in msg.points]
        if apply_z_filter:
            xyz = xyz[xyz[:, 2] <= z_max]
        return xyz

    raise RuntimeError(f'unsupported point type: {msg_type}')


def project_points(xyz, K, T_cam_lidar, width, height, max_range):
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return np.empty((0, 2), np.float32), np.empty(0), np.empty((0,3), np.float32)

    ranges = np.linalg.norm(xyz, axis=1)
    mask = xyz[:, 0] > 0.0  # same front-of-lidar assumption used by current mapping helper
    if max_range > 0:
        mask &= ranges <= max_range
    xyz = xyz[mask]
    ranges = ranges[mask]
    if xyz.size == 0:
        return np.empty((0, 2), np.float32), np.empty(0), np.empty((0,3), np.float32)

    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([xyz.astype(np.float64), ones], axis=1)
    cam = (T_cam_lidar @ homo.T).T[:, :3]

    # Camera optical z must be positive.
    front = cam[:, 2] > 1e-4
    cam = cam[front]
    ranges = ranges[front]
    xyz = xyz[front]
    if cam.size == 0:
        return np.empty((0, 2), np.float32), np.empty(0), np.empty((0,3), np.float32)

    pix_h = (K @ cam.T).T
    uv = pix_h[:, :2] / pix_h[:, [2]]
    in_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[in_img].astype(np.float32), ranges[in_img], xyz[in_img]


def draw_overlay(image, uv, ranges, point_radius=1, max_draw_points=12000):
    out = image.copy()
    if len(uv) == 0:
        cv2.putText(out, 'NO PROJECTED POINTS', (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    if max_draw_points > 0 and len(uv) > max_draw_points:
        # Deterministic even sampling across the full cloud.
        idx = np.linspace(0, len(uv)-1, max_draw_points, dtype=np.int64)
        uv = uv[idx]
        ranges = ranges[idx]

    # BGR distance bins. The exact colors are only labels; geometry is the target.
    bins = [
        (0.0, 5.0,  (0, 255, 0),   '0-5m'),
        (5.0, 10.0, (0, 255, 255), '5-10m'),
        (10.0,15.0, (0, 128, 255), '10-15m'),
        (15.0,20.0, (0, 0, 255),   '15-20m'),
        (20.0,1e9,  (255, 0, 255), '>20m'),
    ]
    for lo, hi, color, _ in bins:
        m = (ranges >= lo) & (ranges < hi)
        pts = np.rint(uv[m]).astype(np.int32)
        for u, v in pts:
            cv2.circle(out, (int(u), int(v)), point_radius, color, -1, lineType=cv2.LINE_AA)

    # Legend
    x0, y0 = 12, 18
    cv2.rectangle(out, (5, 5), (190, 140), (0,0,0), -1)
    cv2.putText(out, 'LiDAR range', (x0, y0+8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    y = y0 + 28
    for _, _, color, label in bins:
        cv2.circle(out, (x0+5, y-4), 5, color, -1)
        cv2.putText(out, label, (x0+18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        y += 20
    return out


def make_panel(image, overlay, title):
    # Side-by-side keeps the raw scene visible while checking point alignment.
    h, w = image.shape[:2]
    left = image.copy()
    right = overlay.copy()
    cv2.rectangle(left, (0,0), (w,30), (0,0,0), -1)
    cv2.rectangle(right, (0,0), (w,30), (0,0,0), -1)
    cv2.putText(left, 'RGB original', (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(right, title, (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    return np.hstack([left, right])


def camera_info_report(cam_info_msg, yaml_K, image_shape=None):
    if cam_info_msg is None:
        return {
            'camera_info_found': False,
            'camera_info_fx': math.nan,
            'camera_info_fy': math.nan,
            'camera_info_cx': math.nan,
            'camera_info_cy': math.nan,
            'camera_info_max_abs_K_diff': math.nan,
            'camera_info_width': math.nan,
            'camera_info_height': math.nan,
            'image_width': image_shape[1] if image_shape is not None else math.nan,
            'image_height': image_shape[0] if image_shape is not None else math.nan,
        }
    Kci = np.asarray(cam_info_msg.k, dtype=np.float64).reshape(3,3)
    return {
        'camera_info_found': True,
        'camera_info_fx': float(Kci[0,0]),
        'camera_info_fy': float(Kci[1,1]),
        'camera_info_cx': float(Kci[0,2]),
        'camera_info_cy': float(Kci[1,2]),
        'camera_info_max_abs_K_diff': float(np.max(np.abs(Kci - yaml_K))),
        'camera_info_width': int(cam_info_msg.width),
        'camera_info_height': int(cam_info_msg.height),
        'image_width': image_shape[1] if image_shape is not None else math.nan,
        'image_height': image_shape[0] if image_shape is not None else math.nan,
    }


def choose_point_topic(topic_types, preferred):
    if preferred and preferred in topic_types:
        return preferred
    if '/livox/lidar_filtered' in topic_types:
        return '/livox/lidar_filtered'
    if '/livox/lidar' in topic_types:
        return '/livox/lidar'
    return None


def process_bag(label, bag_path, cfg, K, T_cam_lidar, out_dir, args, start_sec):
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_id = detect_storage_id(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(bag_path).expanduser().resolve()), storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}

    image_topic = args.image_topic or cfg['image']['image_topic']
    cam_info_topic = args.camera_info_topic or cfg['image'].get('camera_info_topic', '')
    preferred_point_topic = args.point_topic or cfg['pointcloud'].get('topic', '')
    point_topic = choose_point_topic(topic_types, preferred_point_topic)

    if image_topic not in topic_types:
        raise RuntimeError(f'[{label}] image topic not found: {image_topic}')
    if point_topic is None:
        raise RuntimeError(f'[{label}] no usable point topic found. Tried {preferred_point_topic}, /livox/lidar_filtered, /livox/lidar')

    image_type_name = topic_types[image_topic]
    point_type_name = topic_types[point_topic]
    if image_type_name not in {'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'}:
        raise RuntimeError(f'[{label}] unsupported image type: {image_type_name}')
    if point_type_name not in {'sensor_msgs/msg/PointCloud2', 'livox_interfaces/msg/CustomMsg'}:
        raise RuntimeError(f'[{label}] unsupported point type: {point_type_name}')

    image_type = get_message(image_type_name)
    point_type = get_message(point_type_name)
    cam_info_type = get_message(topic_types[cam_info_topic]) if cam_info_topic in topic_types else None

    max_dt = args.max_image_dt if args.max_image_dt >= 0 else float(cfg.get('sync', {}).get('max_image_dt', 0.08))
    max_dt_ns = int(max_dt * 1e9)
    interval_ns = int(max(0.0, args.interval_sec) * 1e9)
    start_ns = int(max(0.0, start_sec) * 1e9)
    duration_ns = int(args.duration_sec * 1e9) if args.duration_sec > 0 else None

    image_cache = deque(maxlen=max(10, args.image_cache))  # (header_stamp_ns, msg)
    pending = deque()  # sampled (header_stamp_ns, msg, rel_sec)
    first_pcl_stamp = None
    last_selected_pcl_stamp = None
    newest_image_stamp = None
    cam_info_msg = None
    rows = []
    saved = 0
    last_image_shape = None

    def finalize_ready(force=False):
        nonlocal saved, last_image_shape
        while pending:
            pcl_stamp, pcl_msg, rel_sec = pending[0]
            if not force:
                if newest_image_stamp is None or newest_image_stamp < pcl_stamp + max_dt_ns:
                    break
            pending.popleft()
            if len(image_cache) == 0:
                continue
            best_stamp, best_img_msg = min(image_cache, key=lambda x: abs(x[0] - pcl_stamp))
            dt_ns = best_stamp - pcl_stamp
            if abs(dt_ns) > max_dt_ns:
                continue

            bgr = decode_image(best_img_msg, image_type_name)
            last_image_shape = bgr.shape[:2]
            xyz = decode_points(
                pcl_msg, point_type_name, args.z_max,
                apply_z_filter=(not args.no_z_filter),
            )
            total_points = int(xyz.shape[0])
            uv, ranges, _ = project_points(xyz, K, T_cam_lidar, bgr.shape[1], bgr.shape[0], args.max_range)
            visible_points = int(len(uv))
            ratio = visible_points / max(total_points, 1)

            overlay = draw_overlay(
                bgr, uv, ranges,
                point_radius=max(1, args.point_radius),
                max_draw_points=max(0, args.max_draw_points),
            )
            title = f'projection | dt={dt_ns/1e6:+.1f} ms | visible={visible_points}/{total_points}'
            panel = make_panel(bgr, overlay, title)
            fname = f'{saved:04d}_t{rel_sec:07.2f}s_dt{dt_ns/1e6:+06.1f}ms.png'
            cv2.imwrite(str(out_dir / fname), panel)

            rows.append({
                'label': label,
                'index': saved,
                'relative_sec': rel_sec,
                'pcl_stamp_ns': pcl_stamp,
                'image_stamp_ns': best_stamp,
                'image_minus_pcl_ms': dt_ns / 1e6,
                'total_points_after_filter': total_points,
                'projected_visible_points': visible_points,
                'projected_visible_ratio': ratio,
                'range_0_5_ratio': float(np.mean((ranges >= 0) & (ranges < 5))) if len(ranges) else 0.0,
                'range_5_10_ratio': float(np.mean((ranges >= 5) & (ranges < 10))) if len(ranges) else 0.0,
                'range_10_15_ratio': float(np.mean((ranges >= 10) & (ranges < 15))) if len(ranges) else 0.0,
                'range_15_20_ratio': float(np.mean((ranges >= 15) & (ranges < 20))) if len(ranges) else 0.0,
                'range_gt20_ratio': float(np.mean(ranges >= 20)) if len(ranges) else 0.0,
                'image_width': int(bgr.shape[1]),
                'image_height': int(bgr.shape[0]),
            })
            if saved % 10 == 0:
                print(
                    f'[{label}] frame={saved:3d} t={rel_sec:7.2f}s '
                    f'dt={dt_ns/1e6:+6.1f}ms points={total_points} visible={visible_points} ({100*ratio:.1f}%)'
                )
            saved += 1
            if args.max_images > 0 and saved >= args.max_images:
                return True
        return False

    print(f'\n[{label}] bag={Path(bag_path).expanduser().resolve()}')
    print(f'[{label}] storage={storage_id}')
    print(f'[{label}] image={image_topic} ({image_type_name})')
    print(f'[{label}] points={point_topic} ({point_type_name})')
    print(f'[{label}] max image dt={max_dt*1000:.1f} ms')

    done = False
    while reader.has_next() and not done:
        topic, serialized, bag_time_ns = reader.read_next()

        if topic == cam_info_topic and cam_info_type is not None and cam_info_msg is None:
            cam_info_msg = deserialize_message(serialized, cam_info_type)
            continue

        if topic == image_topic:
            msg = deserialize_message(serialized, image_type)
            s_ns = stamp_ns(msg)
            image_cache.append((s_ns, msg))
            newest_image_stamp = s_ns
            done = finalize_ready(force=False)
            continue

        if topic != point_topic:
            continue

        msg = deserialize_message(serialized, point_type)
        p_ns = stamp_ns(msg)
        if first_pcl_stamp is None:
            first_pcl_stamp = p_ns
        rel_ns = p_ns - first_pcl_stamp
        if rel_ns < start_ns:
            continue
        if duration_ns is not None and rel_ns >= start_ns + duration_ns:
            break
        if last_selected_pcl_stamp is not None and p_ns - last_selected_pcl_stamp < interval_ns:
            continue
        last_selected_pcl_stamp = p_ns
        pending.append((p_ns, msg, (rel_ns - start_ns) / 1e9))
        done = finalize_ready(force=False)

    if not done:
        finalize_ready(force=True)

    ci = camera_info_report(cam_info_msg, K, last_image_shape)
    print(f'[{label}] saved overlays={len(rows)}')
    if ci['camera_info_found']:
        print(
            f'[{label}] CameraInfo K: fx={ci["camera_info_fx"]:.3f} fy={ci["camera_info_fy"]:.3f} '
            f'cx={ci["camera_info_cx"]:.3f} cy={ci["camera_info_cy"]:.3f}'
        )
        print(f'[{label}] max |CameraInfo.K - YAML.K| = {ci["camera_info_max_abs_K_diff"]:.6f}')
        print(
            f'[{label}] CameraInfo size={ci["camera_info_width"]}x{ci["camera_info_height"]}, '
            f'image size={ci["image_width"]}x{ci["image_height"]}'
        )
    else:
        print(f'[{label}] CameraInfo topic not present/read; only YAML K was used')

    return rows, ci, point_topic


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(label, rows, ci, point_topic):
    if rows:
        dt = np.array([r['image_minus_pcl_ms'] for r in rows], dtype=np.float64)
        vis = np.array([r['projected_visible_ratio'] for r in rows], dtype=np.float64)
        return {
            'label': label,
            'frames': len(rows),
            'point_topic': point_topic,
            'mean_abs_image_pcl_dt_ms': float(np.mean(np.abs(dt))),
            'max_abs_image_pcl_dt_ms': float(np.max(np.abs(dt))),
            'mean_projected_visible_ratio': float(np.mean(vis)),
            **ci,
        }
    return {
        'label': label,
        'frames': 0,
        'point_topic': point_topic,
        'mean_abs_image_pcl_dt_ms': math.nan,
        'max_abs_image_pcl_dt_ms': math.nan,
        'mean_projected_visible_ratio': math.nan,
        **ci,
    }


def main():
    args = parse_args()
    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    K = np.asarray(cfg['intrinsics']['K'], dtype=np.float64).reshape(3,3)
    p = np.asarray(cfg['extrinsics']['p'], dtype=np.float64)
    q = np.asarray(cfg['extrinsics']['q'], dtype=np.float64)
    T_cam_lidar = pose_to_htm(p, q)

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print('=== Geometry used by dino_localmapping ===')
    print(f'config           : {config_path}')
    print(f'YAML K           :\n{K}')
    print(f'extrinsic p      : {p.tolist()}')
    print(f'extrinsic q xyzw : {q.tolist()}')
    print(f'T_camera_lidar   :\n{T_cam_lidar}')
    print('NOTE: moving the camera+LiDAR rigid module together does NOT require changing this relative extrinsic.')

    good_rows, good_ci, good_pt = process_bag(
        'good', args.good_bag, cfg, K, T_cam_lidar, out/'good', args, args.good_start_sec)
    bad_rows, bad_ci, bad_pt = process_bag(
        'bad', args.bad_bag, cfg, K, T_cam_lidar, out/'bad', args, args.bad_start_sec)

    all_rows = good_rows + bad_rows
    write_csv(out/'per_frame.csv', all_rows)
    summaries = [
        summarize('good', good_rows, good_ci, good_pt),
        summarize('bad', bad_rows, bad_ci, bad_pt),
    ]
    write_csv(out/'summary.csv', summaries)

    print('\n=== SUMMARY ===')
    for s in summaries:
        print(
            f'[{s["label"]}] frames={s["frames"]} '
            f'mean|image-pcl dt|={s["mean_abs_image_pcl_dt_ms"]:.1f} ms '
            f'max|dt|={s["max_abs_image_pcl_dt_ms"]:.1f} ms '
            f'visible={100*s["mean_projected_visible_ratio"]:.1f}% '
            f'Kdiff={s["camera_info_max_abs_K_diff"]}'
        )

    print(f'\nSaved: {out}')
    print('  Open several GOOD and BAD PNGs and inspect hard geometric edges:')
    print('  tree trunks, road/grass boundary, poles, rocks, vehicle-visible structures.')
    print('  If 0-5 m aligns but 15-20 m systematically shifts, suspect angular calibration or timing.')
    print('  If all distance bins align in both bags, the LiDAR-camera relative extrinsic is effectively cleared.')


if __name__ == '__main__':
    main()
