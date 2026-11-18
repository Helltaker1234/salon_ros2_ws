#!/usr/bin/env python3
"""
Interactive ROS2 rosbag LiDAR-camera calibration GUI.

Primary use:
  - Fine-tune LiDAR -> camera EXTRINSIC around an existing calibration.
  - Optionally fine-tune camera INTRINSIC K (fx, fy, cx, cy).

The tool mirrors the user's mapping geometry:
    p_cam = T_camera_lidar @ p_lidar
    uv    = K @ p_cam

It does NOT apply lens distortion because the user's current mapping code does not.
It never overwrites the original mapping YAML. Press 's' to save a calibrated copy.

Controls
--------
OpenCV GUI:
  - 'LiDAR-Camera Overlay': compact image-only window
  - 'Calibration Controls': separate trackbar window

Trackbars:
  Frame               select rosbag frame pair
  dTx/dTy/dTz [mm]    translation correction in CAMERA frame
  dRx/dRy/dRz [0.01d] small rotation correction about CAMERA optical axes
  dFx/dFy/dCx/dCy [px] intrinsic correction
  MinRange x10        minimum LiDAR range drawn
  MaxRange x10        maximum LiDAR range drawn
  Draw 1/N            point decimation
  Radius              point radius

Keyboard:
  q / ESC : quit
  s       : save calibration_result.yaml + calibrated full mapping YAML
  r       : reset ALL corrections to zero
  x       : reset only extrinsic corrections
  i       : reset only intrinsic corrections
  e       : toggle Canny edge view behind LiDAR points
  c       : toggle distance colors / single cyan color
  p       : print current K, p, q, and 4x4 T to terminal
  n / b   : next / previous frame

Important convention
--------------------
The correction rotation is LEFT-multiplied:
    R_new = R_corr(camera frame) @ R_base
and corrected translation is:
    t_new = t_base + [dTx,dTy,dTz] in camera axes.
This is convenient for visual fine-tuning of the LiDAR overlay.
"""

import argparse
import copy
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


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Interactive LiDAR-camera calibration from a ROS2 rosbag.',
    )
    p.add_argument('--bag', required=True, help='ROS2 bag directory')
    p.add_argument('--config', default='', help='Mapping YAML containing intrinsics.K and extrinsics.p/q')
    p.add_argument('--image-topic', default='', help='Override RGB image topic')
    p.add_argument('--camera-info-topic', default='', help='Override CameraInfo topic')
    p.add_argument('--point-topic', default='', help='Force point topic, e.g. /livox/lidar')
    p.add_argument('--interval-sec', type=float, default=2.0, help='Sampled pointcloud interval')
    p.add_argument('--max-frames', type=int, default=40, help='Max frame pairs loaded into GUI; 0=unlimited')
    p.add_argument('--start-sec', type=float, default=0.0)
    p.add_argument('--duration-sec', type=float, default=0.0, help='0=until end')
    p.add_argument('--max-dt-ms', type=float, default=20.0, help='Maximum |image - cloud| header stamp difference')
    p.add_argument('--z-max', type=float, default=None, help='Optional z<=value filter. Omit to preserve converter geometry.')
    p.add_argument('--initial-min-range', type=float, default=2.0)
    p.add_argument('--initial-max-range', type=float, default=25.0)
    p.add_argument('--max-draw-points', type=int, default=10000)
    p.add_argument('--output-dir', default='~/lidar_camera_calib_result')
    p.add_argument('--window-scale', type=float, default=1.0, help='Overlay-window scaling relative to the camera image')
    p.add_argument('--trans-range-mm', type=int, default=500, help='Trackbar translation correction range ±mm')
    p.add_argument('--rot-range-deg', type=float, default=15.0, help='Trackbar rotation correction range ±degrees')
    p.add_argument('--intrinsic-range-px', type=int, default=300, help='Trackbar fx/fy/cx/cy correction range ±pixels')
    p.add_argument('--show-overlay-text', action='store_true', help='Reserved compatibility option; V3 keeps calibration overlay text off by default')
    return p.parse_args()


def resolve_config(cli_value: str) -> Path:
    if cli_value:
        p = Path(cli_value).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    here = Path(__file__).resolve()
    candidates = []
    # Expected workspace placement:
    # .../src/physics_atv_visual_mapping/scripts/tools/<this>.py
    try:
        candidates.append(here.parents[2] / 'config/ros/dino_vlad_rough_vits14_252x448_4bags.yaml')
    except Exception:
        pass
    candidates.append(Path.cwd() / 'src/physics_atv_visual_mapping/config/ros/dino_vlad_rough_vits14_252x448_4bags.yaml')
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError('Could not find mapping YAML automatically; pass --config /path/to/yaml')


# ---------- Geometry ----------

def quat_xyzw_to_R(q):
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def R_to_quat_xyzw(R):
    # Stable conversion, returns normalized xyzw.
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    # Keep sign stable-ish: prefer positive w.
    if q[3] < 0:
        q *= -1
    return q


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def correction_R(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = [math.radians(v) for v in (rx_deg, ry_deg, rz_deg)]
    # Apply X then Y then Z in camera axes by matrix composition.
    return Rz(rz) @ Ry(ry) @ Rx(rx)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = np.asarray(t, dtype=np.float64)
    return T


# ---------- ROS bag I/O ----------

def detect_storage_id(bag_dir: str):
    metadata = Path(bag_dir).expanduser().resolve() / 'metadata.yaml'
    if not metadata.is_file():
        raise RuntimeError(f'Bag metadata not found: {metadata}')
    text = metadata.read_text(encoding='utf-8')
    m = re.search(r'^\s*storage_identifier:\s*(\S+)\s*$', text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    # fallback
    if list(metadata.parent.glob('*.mcap')):
        return 'mcap'
    if list(metadata.parent.glob('*.db3')):
        return 'sqlite3'
    raise RuntimeError(f'Could not determine storage backend for {bag_dir}')


def stamp_ns(msg):
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


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
        raise RuntimeError(f'Unsupported image encoding: {msg.encoding}')
    channels, conversion = mapping[enc]
    row_bytes = int(msg.step)
    useful = int(msg.width) * channels
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height) * row_bytes
    if raw.size < expected:
        raise RuntimeError('Truncated sensor_msgs/Image')
    img = raw[:expected].reshape(int(msg.height), row_bytes)
    img = img[:, :useful].reshape(int(msg.height), int(msg.width), channels)
    if conversion is not None:
        img = cv2.cvtColor(img, conversion)
    return img


def decode_image(msg, msg_type):
    if msg_type == 'sensor_msgs/msg/Image':
        return decode_raw_image(msg)
    if msg_type == 'sensor_msgs/msg/CompressedImage':
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('cv2.imdecode failed')
        return img
    raise RuntimeError(f'Unsupported image type: {msg_type}')


def pointcloud2_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    for name in ('x','y','z'):
        if name not in fields:
            raise RuntimeError(f'PointCloud2 missing {name}')
        if int(fields[name].datatype) != 7:  # FLOAT32
            raise RuntimeError(f'PointCloud2 field {name} is not FLOAT32')
    n = int(msg.width) * int(msg.height)
    if n <= 0:
        return np.empty((0,3), dtype=np.float32)
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x','y','z'],
        'formats': [endian+'f4', endian+'f4', endian+'f4'],
        'offsets': [int(fields['x'].offset), int(fields['y'].offset), int(fields['z'].offset)],
        'itemsize': int(msg.point_step),
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32, copy=False)
    return xyz


def decode_points(msg, msg_type, z_max=None):
    if msg_type == 'sensor_msgs/msg/PointCloud2':
        xyz = pointcloud2_xyz(msg)
    elif msg_type == 'livox_interfaces/msg/CustomMsg':
        if len(msg.points) == 0:
            return np.empty((0,3), dtype=np.float32)
        xyz = np.empty((len(msg.points),3), dtype=np.float32)
        xyz[:,0] = [p.x for p in msg.points]
        xyz[:,1] = [p.y for p in msg.points]
        xyz[:,2] = [p.z for p in msg.points]
    else:
        raise RuntimeError(f'Unsupported point type: {msg_type}')
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if z_max is not None:
        xyz = xyz[xyz[:,2] <= z_max]
    return xyz


def choose_point_topic(topic_types, forced):
    if forced:
        if forced not in topic_types:
            raise RuntimeError(f'Forced point topic not found: {forced}')
        return forced
    if '/livox/lidar_filtered' in topic_types:
        return '/livox/lidar_filtered'
    if '/livox/lidar' in topic_types:
        return '/livox/lidar'
    raise RuntimeError('Neither /livox/lidar_filtered nor /livox/lidar found')


def load_frame_pairs(args, cfg):
    bag = str(Path(args.bag).expanduser().resolve())
    storage_id = detect_storage_id(bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}

    image_topic = args.image_topic or cfg['image']['image_topic']
    camera_info_topic = args.camera_info_topic or cfg['image'].get('camera_info_topic', '')
    point_topic = choose_point_topic(topic_types, args.point_topic)

    if image_topic not in topic_types:
        raise RuntimeError(f'Image topic not found: {image_topic}')
    image_type_name = topic_types[image_topic]
    point_type_name = topic_types[point_topic]
    image_type = get_message(image_type_name)
    point_type = get_message(point_type_name)
    cam_info_type = get_message(topic_types[camera_info_topic]) if camera_info_topic in topic_types else None

    max_dt_ns = int(args.max_dt_ms * 1e6)
    interval_ns = int(max(0.0, args.interval_sec) * 1e9)
    start_ns = int(max(0.0, args.start_sec) * 1e9)
    duration_ns = int(args.duration_sec * 1e9) if args.duration_sec > 0 else None

    image_cache = deque(maxlen=180)
    pending = deque()
    newest_image_stamp = None
    first_pcl_stamp = None
    last_selected = None
    cam_info_msg = None
    pairs = []

    def finalize_ready(force=False):
        while pending:
            pcl_stamp, pcl_msg, rel_sec = pending[0]
            if not force and (newest_image_stamp is None or newest_image_stamp < pcl_stamp + max_dt_ns):
                break
            pending.popleft()
            if not image_cache:
                continue
            img_stamp, img_msg = min(image_cache, key=lambda x: abs(x[0] - pcl_stamp))
            dt_ns = img_stamp - pcl_stamp
            if abs(dt_ns) > max_dt_ns:
                continue
            image = decode_image(img_msg, image_type_name)
            xyz = decode_points(pcl_msg, point_type_name, z_max=args.z_max)
            pairs.append({
                'image': image,
                'xyz': xyz,
                'dt_ms': dt_ns / 1e6,
                'rel_sec': rel_sec,
            })
            if args.max_frames > 0 and len(pairs) >= args.max_frames:
                return True
        return False

    print('=== Loading calibration frames ===')
    print(f'bag          : {bag}')
    print(f'storage      : {storage_id}')
    print(f'image topic  : {image_topic} ({image_type_name})')
    print(f'point topic  : {point_topic} ({point_type_name})')
    print(f'max |dt|     : {args.max_dt_ms:.1f} ms')
    print(f'interval     : {args.interval_sec:.2f} s')
    print(f'z filter     : {args.z_max}')

    done = False
    while reader.has_next() and not done:
        topic, serialized, _ = reader.read_next()

        if topic == camera_info_topic and cam_info_type is not None and cam_info_msg is None:
            cam_info_msg = deserialize_message(serialized, cam_info_type)
            continue

        if topic == image_topic:
            msg = deserialize_message(serialized, image_type)
            s = stamp_ns(msg)
            image_cache.append((s, msg))
            newest_image_stamp = s
            done = finalize_ready(False)
            continue

        if topic != point_topic:
            continue
        msg = deserialize_message(serialized, point_type)
        p = stamp_ns(msg)
        if first_pcl_stamp is None:
            first_pcl_stamp = p
        rel_ns = p - first_pcl_stamp
        if rel_ns < start_ns:
            continue
        if duration_ns is not None and rel_ns >= start_ns + duration_ns:
            break
        if last_selected is not None and p - last_selected < interval_ns:
            continue
        last_selected = p
        pending.append((p, msg, (rel_ns - start_ns)/1e9))
        done = finalize_ready(False)

    if not done:
        finalize_ready(True)

    if not pairs:
        raise RuntimeError('No synchronized image/cloud pairs were loaded. Increase --max-dt-ms or check topics.')

    print(f'Loaded pairs : {len(pairs)}')
    dts = np.array([x['dt_ms'] for x in pairs])
    print(f'dt mean|.|   : {np.mean(np.abs(dts)):.2f} ms, max|.|={np.max(np.abs(dts)):.2f} ms')
    if cam_info_msg is not None:
        Kci = np.asarray(cam_info_msg.k, dtype=np.float64).reshape(3,3)
        print('CameraInfo K :')
        print(Kci)
        print(f'CameraInfo size: {cam_info_msg.width}x{cam_info_msg.height}')
    else:
        Kci = None
        print('CameraInfo   : not found/read')
    return pairs, Kci, point_topic


# ---------- Projection / drawing ----------

def project_points(xyz, K, T, width, height, min_range, max_range):
    if xyz.size == 0:
        return np.empty((0,2), dtype=np.float32), np.empty(0, dtype=np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    m = np.isfinite(xyz).all(axis=1) & (rng >= min_range) & (rng <= max_range)
    xyz = xyz[m]
    rng = rng[m]
    if xyz.size == 0:
        return np.empty((0,2), dtype=np.float32), np.empty(0, dtype=np.float32)

    homo = np.c_[xyz.astype(np.float64), np.ones(len(xyz))]
    cam = (T @ homo.T).T[:, :3]
    m = cam[:,2] > 1e-4
    cam = cam[m]
    rng = rng[m]
    if cam.size == 0:
        return np.empty((0,2), dtype=np.float32), np.empty(0, dtype=np.float32)

    ph = (K @ cam.T).T
    uv = ph[:,:2] / ph[:,[2]]
    m = (uv[:,0] >= 0) & (uv[:,0] < width) & (uv[:,1] >= 0) & (uv[:,1] < height)
    return uv[m].astype(np.float32), rng[m].astype(np.float32)


def edge_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    bg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return bg


def draw_points(base, uv, ranges, draw_every, radius, distance_colors=True, max_draw_points=10000):
    out = base.copy()
    if len(uv) == 0:
        return out
    idx = np.arange(0, len(uv), max(1, int(draw_every)))
    if max_draw_points > 0 and len(idx) > max_draw_points:
        idx = idx[np.linspace(0, len(idx)-1, max_draw_points, dtype=np.int64)]
    uv = uv[idx]
    ranges = ranges[idx]

    if not distance_colors:
        pts = np.rint(uv).astype(np.int32)
        for u,v in pts:
            cv2.circle(out, (int(u),int(v)), radius, (255,255,0), -1, cv2.LINE_AA)
        return out

    bins = [
        (0,5,(0,255,0)),
        (5,10,(0,255,255)),
        (10,15,(0,128,255)),
        (15,20,(0,0,255)),
        (20,1e9,(255,0,255)),
    ]
    for lo,hi,color in bins:
        m = (ranges >= lo) & (ranges < hi)
        pts = np.rint(uv[m]).astype(np.int32)
        for u,v in pts:
            cv2.circle(out, (int(u),int(v)), radius, color, -1, cv2.LINE_AA)
    return out


# ---------- GUI ----------

def main():
    args = parse_args()
    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    K_base = np.asarray(cfg['intrinsics']['K'], dtype=np.float64).reshape(3,3)
    p_base = np.asarray(cfg['extrinsics']['p'], dtype=np.float64)
    q_base = np.asarray(cfg['extrinsics']['q'], dtype=np.float64)
    R_base = quat_xyzw_to_R(q_base)

    pairs, K_camera_info, point_topic = load_frame_pairs(args, cfg)

    print('\n=== Base calibration ===')
    print(f'config : {config_path}')
    print('K base :\n', K_base)
    print('p base :', p_base.tolist())
    print('q base :', q_base.tolist(), '(xyzw)')
    if K_camera_info is not None:
        print(f'max |CameraInfo K - YAML K| = {np.max(np.abs(K_camera_info-K_base)):.6f}')

    view_window = 'LiDAR-Camera Overlay'
    control_window = 'Calibration Controls'

    # Keep the image visible in its own compact window. Trackbars live in a
    # separate window so their vertical stack does not consume the image area.
    cv2.namedWindow(view_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)

    first_h, first_w = pairs[0]['image'].shape[:2]
    cv2.resizeWindow(
        view_window,
        max(480, int(first_w * args.window_scale)),
        max(270, int(first_h * args.window_scale)),
    )
    # This window only contains controls. It can be moved aside independently.
    cv2.resizeWindow(control_window, 620, 720)

    # Put them side-by-side on a typical desktop. Window managers may override.
    try:
        cv2.moveWindow(view_window, 20, 40)
        cv2.moveWindow(control_window, max(520, int(first_w * args.window_scale) + 50), 40)
    except Exception:
        pass

    window = control_window

    # Trackbar centers / scales.
    tr_mm = int(args.trans_range_mm)
    tr_center = tr_mm
    rot_units = int(round(args.rot_range_deg * 100.0)) # 0.01 degree units
    rot_center = rot_units
    in_px = int(args.intrinsic_range_px)
    in_center = in_px

    def noop(_):
        pass

    cv2.createTrackbar('Frame', window, 0, max(0,len(pairs)-1), noop)
    cv2.createTrackbar('dTx mm', window, tr_center, 2*tr_mm, noop)
    cv2.createTrackbar('dTy mm', window, tr_center, 2*tr_mm, noop)
    cv2.createTrackbar('dTz mm', window, tr_center, 2*tr_mm, noop)
    cv2.createTrackbar('dRx 0.01deg', window, rot_center, 2*rot_units, noop)
    cv2.createTrackbar('dRy 0.01deg', window, rot_center, 2*rot_units, noop)
    cv2.createTrackbar('dRz 0.01deg', window, rot_center, 2*rot_units, noop)
    cv2.createTrackbar('dFx px', window, in_center, 2*in_px, noop)
    cv2.createTrackbar('dFy px', window, in_center, 2*in_px, noop)
    cv2.createTrackbar('dCx px', window, in_center, 2*in_px, noop)
    cv2.createTrackbar('dCy px', window, in_center, 2*in_px, noop)
    cv2.createTrackbar('MinRange x10', window, int(round(args.initial_min_range*10)), 300, noop)
    cv2.createTrackbar('MaxRange x10', window, int(round(args.initial_max_range*10)), 400, noop)
    cv2.createTrackbar('Draw 1/N', window, 2, 20, noop)
    cv2.createTrackbar('Radius', window, 1, 4, noop)

    edge_mode = False
    distance_colors = True

    def reset_extrinsic():
        for name, val in [('dTx mm',tr_center),('dTy mm',tr_center),('dTz mm',tr_center),
                          ('dRx 0.01deg',rot_center),('dRy 0.01deg',rot_center),('dRz 0.01deg',rot_center)]:
            cv2.setTrackbarPos(name, window, val)

    def reset_intrinsic():
        for name in ('dFx px','dFy px','dCx px','dCy px'):
            cv2.setTrackbarPos(name, window, in_center)

    def current_values():
        fi = cv2.getTrackbarPos('Frame', window)
        dtx = (cv2.getTrackbarPos('dTx mm', window) - tr_center) / 1000.0
        dty = (cv2.getTrackbarPos('dTy mm', window) - tr_center) / 1000.0
        dtz = (cv2.getTrackbarPos('dTz mm', window) - tr_center) / 1000.0
        drx = (cv2.getTrackbarPos('dRx 0.01deg', window) - rot_center) / 100.0
        dry = (cv2.getTrackbarPos('dRy 0.01deg', window) - rot_center) / 100.0
        drz = (cv2.getTrackbarPos('dRz 0.01deg', window) - rot_center) / 100.0
        dfx = cv2.getTrackbarPos('dFx px', window) - in_center
        dfy = cv2.getTrackbarPos('dFy px', window) - in_center
        dcx = cv2.getTrackbarPos('dCx px', window) - in_center
        dcy = cv2.getTrackbarPos('dCy px', window) - in_center
        minr = cv2.getTrackbarPos('MinRange x10', window)/10.0
        maxr = cv2.getTrackbarPos('MaxRange x10', window)/10.0
        if maxr <= minr + 0.1:
            maxr = minr + 0.1
        draw_every = max(1, cv2.getTrackbarPos('Draw 1/N', window))
        radius = max(1, cv2.getTrackbarPos('Radius', window))

        K = K_base.copy()
        K[0,0] += dfx
        K[1,1] += dfy
        K[0,2] += dcx
        K[1,2] += dcy

        Rc = correction_R(drx,dry,drz)
        R = Rc @ R_base
        t = p_base + np.array([dtx,dty,dtz], dtype=np.float64)
        T = make_T(R,t)
        q = R_to_quat_xyzw(R)
        return {
            'frame': fi,
            'dtrans': (dtx,dty,dtz),
            'drot_deg': (drx,dry,drz),
            'dintr': (dfx,dfy,dcx,dcy),
            'minr': minr,
            'maxr': maxr,
            'draw_every': draw_every,
            'radius': radius,
            'K': K,
            'R': R,
            't': t,
            'T': T,
            'q': q,
        }

    def print_current(v):
        print('\n=== Current calibration ===')
        print('delta translation [m] :', [round(x,6) for x in v['dtrans']])
        print('delta rotation [deg]  :', [round(x,4) for x in v['drot_deg']], '(camera-axis correction)')
        print('delta intrinsic [px]  :', v['dintr'])
        print('K:')
        print(v['K'])
        print('p:', [float(x) for x in v['t']])
        print('q xyzw:', [float(x) for x in v['q']])
        print('T_camera_lidar:')
        print(v['T'])

    def save_current(v):
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        result = {
            'source_bag': str(Path(args.bag).expanduser().resolve()),
            'source_config': str(config_path),
            'point_topic': point_topic,
            'convention': 'LiDAR to camera; R_new = R_corr(camera axes) @ R_base; t_new=t_base+delta_camera',
            'intrinsics': {
                'K': [float(x) for x in v['K'].reshape(-1)],
            },
            'extrinsics': {
                'p': [float(x) for x in v['t']],
                'q': [float(x) for x in v['q']],
            },
            'corrections': {
                'translation_m': [float(x) for x in v['dtrans']],
                'rotation_camera_deg': [float(x) for x in v['drot_deg']],
                'intrinsic_px_fx_fy_cx_cy': [float(x) for x in v['dintr']],
            },
        }
        result_path = out_dir / 'calibration_result.yaml'
        result_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding='utf-8')

        cfg_new = copy.deepcopy(cfg)
        cfg_new['intrinsics']['K'] = [float(x) for x in v['K'].reshape(-1)]
        cfg_new['extrinsics']['p'] = [float(x) for x in v['t']]
        cfg_new['extrinsics']['q'] = [float(x) for x in v['q']]
        full_path = out_dir / 'dino_vlad_rough_calibrated.yaml'
        full_path.write_text(yaml.safe_dump(cfg_new, sort_keys=False), encoding='utf-8')

        # Save current screenshot too.
        pair = pairs[v['frame']]
        uv, ranges = project_points(pair['xyz'], v['K'], v['T'], pair['image'].shape[1], pair['image'].shape[0], v['minr'], v['maxr'])
        vis = draw_points(pair['image'], uv, ranges, v['draw_every'], v['radius'], True, args.max_draw_points)
        cv2.imwrite(str(out_dir / 'calibration_preview.png'), vis)

        print(f'\nSaved calibration:\n  {result_path}\n  {full_path}\n  {out_dir / "calibration_preview.png"}')
        print_current(v)

    print('\nGUI ready.')
    print(f'Default tuning ranges: translation ±{args.trans_range_mm} mm, rotation ±{args.rot_range_deg:.1f} deg, intrinsic ±{args.intrinsic_range_px} px.')
    print('The overlay and controls are now separate windows; resize/move the overlay freely without losing the image.')
    print('V4: the overlay is completely clean: no text, no dark/translucent header. Only the image and projected LiDAR points are shown.')
    print('Recommended: first leave fx/fy/cx/cy at 0 because your previous CameraInfo K matched YAML exactly.')
    print('Use clear tree trunks / poles / road edges at several distances, not only one frame.')
    print('Keys: s save | r reset all | x reset extrinsic | i reset intrinsic | e edge view | c color mode | p print | n/b frame | q quit')

    while True:
        v = current_values()
        pair = pairs[v['frame']]
        img = pair['image']
        base = edge_background(img) if edge_mode else img
        uv, ranges = project_points(pair['xyz'], v['K'], v['T'], img.shape[1], img.shape[0], v['minr'], v['maxr'])
        view = draw_points(base, uv, ranges, v['draw_every'], v['radius'], distance_colors, args.max_draw_points)

        # Clean calibration view: RGB/edge background + projected LiDAR points only.
        cv2.imshow(view_window, view)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            save_current(v)
        elif key == ord('r'):
            reset_extrinsic(); reset_intrinsic()
        elif key == ord('x'):
            reset_extrinsic()
        elif key == ord('i'):
            reset_intrinsic()
        elif key == ord('e'):
            edge_mode = not edge_mode
        elif key == ord('c'):
            distance_colors = not distance_colors
        elif key == ord('p'):
            print_current(v)
        elif key == ord('n'):
            cv2.setTrackbarPos('Frame', window, min(len(pairs)-1, v['frame']+1))
        elif key == ord('b'):
            cv2.setTrackbarPos('Frame', window, max(0, v['frame']-1))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()