#!/usr/bin/env python3
import yaml
import json
import copy
import warnings
from array import array
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy,
)
import numpy as np

import torch
import torch.nn.functional as F
import cv_bridge
import os
import time
from torch_scatter import scatter as ts_scatter

import distinctipy as COLORS

from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32, Header
from sensor_msgs.msg import PointField
from std_srvs.srv import Trigger

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.processing_schedule import CloudSchedule
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import (
    get_pixel_from_3D_source, get_points_and_pixels_in_frame,
    xyz_array_to_point_cloud_msg,
)
from physics_atv_visual_mapping.localmapping.localmapping import localmap_from_pointcloud, aggregate_localmaps, aligned_map_metadata
from physics_atv_visual_mapping.utils import (
    pcl_msg_to_xyz,
    pose_to_htm,
    quaternion_xyzw_to_rotation_matrix,
    transform_xyz,
)


class DinoMappingNode(Node):
    def __init__(self, config):
        super().__init__('visual_mapping')

        self.localmap = None
        self.pcl_msg = None
        self.odom_msg = None
        self.img_msg = None
        self.odom_frame = None

        self.device = config['device']
        self.base_metadata = config['localmapping']['metadata']
        self.localmap_ema = config['localmapping']['ema']
        self.layer_key = config['localmapping'].get('layer_key', None)
        self.layer_keys = self.make_layer_keys(config['localmapping']['layer_keys']) \
            if 'layer_keys' in config['localmapping'] else None
        self.last_update_time = 0.
        self.compute_slope = config.get('slope', {}).get('enable', False)

        perf_cfg = config.get('PERFORMANCE', {})
        self.use_amp = bool(perf_cfg.get('use_amp', False))
        self.publish_dino_pcl = bool(
            perf_cfg.get('publish_dino_pcl', False))
        self.publish_dino_image = bool(
            perf_cfg.get('publish_dino_image', False))
        self.publish_rgb_viz = bool(
            perf_cfg.get('publish_rgb_viz', False))
        self.log_frame_timing = bool(
            perf_cfg.get('log_frame_timing', False))
        self.pcl_viz_stride = max(
            1, int(perf_cfg.get('pcl_viz_stride', 4)))

        # Runtime profiler. CUDA synchronization gives accurate per-stage GPU
        # timings, but should only be enabled for short diagnostic runs because
        # it prevents asynchronous GPU work from overlapping with CPU work.
        profile_cfg = config.get('PROFILE', {})
        self.profile_enable = bool(profile_cfg.get('enable', False))
        self.profile_period = max(1, int(profile_cfg.get('period', 10)))
        self.profile_warn_ms = float(profile_cfg.get('warn_ms', 200.0))
        self.profile_cuda_sync = bool(profile_cfg.get('cuda_sync', False))
        self.profile_topk = max(1, int(profile_cfg.get('topk', 10)))
        self.profile_ema_alpha = min(
            1.0, max(0.0, float(profile_cfg.get('ema_alpha', 0.10))))
        self.profile_hz_window = max(
            2, int(profile_cfg.get('hz_window', 30)))
        self.profile_seq = 0
        self.profile_output_count = 0
        self.profile_ema = {}
        self.profile_callback_times = deque(maxlen=self.profile_hz_window)
        self.profile_output_times = deque(maxlen=self.profile_hz_window)

        # The cached DINOv2 code emits three UserWarnings when the optional
        # xFormers package is absent. Inference remains valid through its
        # PyTorch scaled-dot-product-attention fallback, so suppress only those
        # known import-time messages and leave every other warning visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'xFormers is (?:not available|disabled|available).*',
                category=UserWarning,
            )
            self.image_pipeline = setup_image_pipeline(config)

        self.get_logger().info(
            'DINO native FP16 is {}'.format(
                'enabled' if self.use_amp else 'disabled'))
        if self.profile_enable:
            self.get_logger().info(
                'DINO profiler enabled: period={}, cuda_sync={}, hz_window={}'.format(
                    self.profile_period,
                    self.profile_cuda_sync,
                    self.profile_hz_window,
                ))

        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config['image']['image_compressed']
        self.pcl_frame_remap = config['pointcloud'].get('frame', None)

        if 'rect' in config['image']['image_topic']:
            self.intrinsics = torch.tensor(config['intrinsics']['P'], device=config['device']).reshape(3, 3).float()
        else:
            self.intrinsics = torch.tensor(config['intrinsics']['K'], device=config['device']).reshape(3, 3).float()

        self.extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q']),
        ], axis=-1))
        # Calibration is camera <- lidar; Super-LIO deskewed points are in
        # IMU/body coordinates. Compose lidar <- body explicitly, never relabel
        # IMU coordinates as raw LiDAR coordinates for camera projection.
        input_to_calibration = config['extrinsics'].get('input_to_calibration')
        if input_to_calibration is not None:
            T_calibration_input = pose_to_htm(np.concatenate([
                np.array(input_to_calibration['p']),
                np.array(input_to_calibration['q']),
            ]))
            self.extrinsics = self.extrinsics @ T_calibration_input
        self.extrinsics_device = self.extrinsics.to(self.device)
        self.projection_matrix = None
        self.projection_signature = None

        # Local-map metadata is consumed through scalar .item() calls. Keeping
        # it on CPU avoids repeated GPU allocations and synchronization without
        # changing the float32 values used by rasterization.
        self.metadata_length_x = torch.tensor(
            self.base_metadata['length_x'], dtype=torch.float32)
        self.metadata_length_y = torch.tensor(
            self.base_metadata['length_y'], dtype=torch.float32)
        self.metadata_resolution = torch.tensor(
            self.base_metadata['resolution'], dtype=torch.float32)

        # Inputs wake a mutually exclusive processing guard. Keep just one
        # pending cloud; sensor timestamps bound inference rate independently
        # of callback/timer phase. Old clouds are never queued for later work.
        sync_cfg = config.get('sync', {})
        self.process_hz = max(0.1, float(sync_cfg.get('process_hz', 5.0)))
        self.cache_size = max(2, int(sync_cfg.get('cache_size', 10)))
        self.max_image_dt = float(sync_cfg.get('max_image_dt', 0.08))
        self.max_odom_dt = float(sync_cfg.get('max_odom_dt', 0.12))

        self.latest_pcl = None
        self.image_cache = deque(maxlen=self.cache_size)
        self.odom_cache = deque(maxlen=self.cache_size)
        self.last_processed_pcl_stamp_ns = None
        self.cloud_schedule = CloudSchedule(self.process_hz)
        self.sync_wait_s = max(0., float(sync_cfg.get('wait_for_pair_s', .05)))
        self.pending_since = None
        self.processing = False
        self.scheduling_stats = Counter()
        self.processing_times_ms = deque(maxlen=2048)
        self.pair_deltas_ms = deque(maxlen=2048)
        # Default callback group is mutually exclusive, including this guard.
        self.process_guard = self.create_guard_condition(self._process_latest_inputs)

        image_msg_type = CompressedImage if self.compressed_img else Image

        self.input_reliability = str(sync_cfg.get('input_reliability', 'best_effort')).lower()
        policies = {'best_effort': ReliabilityPolicy.BEST_EFFORT,
                    'reliable': ReliabilityPolicy.RELIABLE}
        if self.input_reliability not in policies:
            raise ValueError('sync.input_reliability must be best_effort or reliable')
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=policies[self.input_reliability],
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pcl_sub = self.create_subscription(
            PointCloud2,
            config['pointcloud']['topic'],
            self._pcl_callback,
            sensor_qos,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            config['odometry']['topic'],
            self._odom_callback,
            sensor_qos,
        )
        # A frame can arrive while the single inference callback is running.
        # Retain a small bounded camera history so the closest frame is not
        # overwritten at DDS depth=1 before Python can cache it.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.cache_size,
            reliability=policies[self.input_reliability],
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_sub = self.create_subscription(
            image_msg_type,
            config['image']['image_topic'],
            self._image_callback,
            image_qos,
        )

        # Lightweight retry only: resolves a bounded sync wait if a stream
        # stops. The guard/rate gate, not this timer, schedules inference.
        self.process_timer = self.create_timer(
            min(.02, 1.0 / self.process_hz),
            self._request_processing,
        )

        self.pcl_pub = self.create_publisher(PointCloud2, '/dino_pcl', 1)
        self.gridmap_pub = self.create_publisher(GridMap, '/dino_gridmap', 5)
        self.image_pub = self.create_publisher(Image, '/dino_image', 1)

        self.viz = config.get('viz', False)
        self.stats_service = self.create_service(
            Trigger, '~/processing_stats', self._processing_status)

    def _processing_status(self, request, response):
        # Read-only diagnostics; no reset, synchronization, or GPU work.
        samples = sorted(self.processing_times_ms)
        response.success = True
        response.message = json.dumps({
            'counts': dict(self.scheduling_stats),
            'sample_wall_s': time.monotonic(),
            'input_reliability': self.input_reliability,
            'processing_ms': {
                'n': len(samples),
                'mean': sum(samples) / len(samples) if samples else None,
                'p95': samples[int((len(samples) - 1) * .95)] if samples else None,
                'max': samples[-1] if samples else None,
            },
            'max_odom_error_ms': max((abs(v[0]) for v in self.pair_deltas_ms), default=0.),
            'max_image_error_ms': max((abs(v[1]) for v in self.pair_deltas_ms), default=0.),
            'pair_error_abs_p95_ms': {
                name: (float(np.percentile([abs(v[i]) for v in self.pair_deltas_ms], 95))
                       if self.pair_deltas_ms else None)
                for i, name in enumerate(('odom', 'image'))
            },
            'grid_origin_alignment': 'world_lattice',
            'last_processed_cloud_ns': self.last_processed_pcl_stamp_ns,
        })
        return response

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk['n']):
                out.append('{}_{}'.format(lk['key'], i))
        return out

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _stamp_to_sec(self, stamp):
        return self._stamp_to_ns(stamp) * 1e-9

    def _pcl_callback(self, msg):
        # Single latest slot bounds memory/latency even under overload.
        stamp = self._stamp_to_ns(msg.header.stamp)
        previous = self.latest_pcl
        if previous is None or self._stamp_to_ns(previous.header.stamp) != stamp:
            if (previous is not None and
                    self._stamp_to_ns(previous.header.stamp) != self.last_processed_pcl_stamp_ns):
                self.scheduling_stats['pending_replaced'] += 1
            self.pending_since = time.monotonic()
        self.latest_pcl = msg
        self.scheduling_stats['cloud_received'] += 1
        self._request_processing()

    def _odom_callback(self, msg):
        self.scheduling_stats['odom_received'] += 1
        self.odom_cache.append(msg)
        self._request_processing()

    def _image_callback(self, msg):
        self.scheduling_stats['image_received'] += 1
        self.image_cache.append(msg)
        self._request_processing()

    def _request_processing(self):
        if self.processing or self.latest_pcl is None:
            return
        stamp = self._stamp_to_ns(self.latest_pcl.header.stamp)
        if self.cloud_schedule.ready(stamp):
            self.process_guard.trigger()

    def _nearest_message(self, messages, target_stamp_ns):
        if not messages:
            return None, float('inf')

        nearest = min(
            messages,
            key=lambda msg: abs(
                self._stamp_to_ns(msg.header.stamp) - target_stamp_ns),
        )
        dt = (
            self._stamp_to_ns(nearest.header.stamp) - target_stamp_ns
        ) * 1e-9
        return nearest, dt

    def _process_latest_inputs(self):
        if self.processing:
            return
        pcl_msg = self.latest_pcl
        if pcl_msg is None:
            return

        pcl_stamp_ns = self._stamp_to_ns(pcl_msg.header.stamp)
        if not self.cloud_schedule.ready(pcl_stamp_ns):
            return

        odom_msg, odom_dt = self._nearest_message(
            self.odom_cache, pcl_stamp_ns)
        img_msg, image_dt = self._nearest_message(
            self.image_cache, pcl_stamp_ns)

        if odom_msg is None or img_msg is None:
            self.scheduling_stats['waiting_for_inputs'] += 1
            return
        if abs(odom_dt) > self.max_odom_dt or abs(image_dt) > self.max_image_dt:
            self.scheduling_stats['waiting_for_alignment'] += 1
            return

        # Allow the matching odometry and the next camera frame to arrive.
        # This avoids choosing a stale pair just because the cloud callback
        # ran first. The original maximum timestamp errors remain enforced.
        pair_complete = (
            self._stamp_to_ns(self.odom_cache[-1].header.stamp) >= pcl_stamp_ns and
            self._stamp_to_ns(self.image_cache[-1].header.stamp) >= pcl_stamp_ns)
        waiting_s = time.monotonic() - self.pending_since if self.pending_since is not None else 0.
        if not pair_complete and waiting_s < self.sync_wait_s:
            self.scheduling_stats['waiting_for_watermark'] += 1
            return

        self.cloud_schedule.mark(pcl_stamp_ns)
        self.last_processed_pcl_stamp_ns = pcl_stamp_ns
        self.pcl_msg = pcl_msg
        if (self.pcl_frame_remap is not None and
                self.pcl_msg.header.frame_id != self.pcl_frame_remap):
            self.get_logger().error(
                'cloud frame {} differs from configured {}; a coordinate transform is required'.format(
                    self.pcl_msg.header.frame_id, self.pcl_frame_remap),
                throttle_duration_sec=1.0)
            return

        self.odom_msg = odom_msg
        self.img_msg = img_msg

        if self.odom_frame is None:
            self.odom_frame = self.odom_msg.header.frame_id

        self.scheduling_stats['processed'] += 1
        self.pair_deltas_ms.append((odom_dt * 1000., image_dt * 1000.))
        self.processing = True
        started = time.perf_counter()
        try:
            self.spin_once()
        finally:
            self.processing_times_ms.append((time.perf_counter() - started) * 1000.)
            self.processing = False

    def _profile_sync_cuda(self):
        """Wait for this process's CUDA work when accurate GPU timing is requested."""
        if (
                self.profile_enable
                and self.profile_cuda_sync
                and torch.cuda.is_available()):
            torch.cuda.synchronize()

    @staticmethod
    def _profile_rolling_hz(timestamps):
        """Return the rate over a small rolling wall-clock window."""
        if len(timestamps) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(timestamps) - 1) / elapsed

    def _profile_start(self):
        """Start profiling one synchronized input callback."""
        if not self.profile_enable:
            return None

        self._profile_sync_cuda()
        now = time.perf_counter()
        self.profile_callback_times.append(now)

        cuda_alloc = 0
        cuda_reserved = 0
        if torch.cuda.is_available():
            # Make max_memory_* below represent this callback rather than the
            # highest allocation observed since process startup.
            torch.cuda.reset_peak_memory_stats()
            cuda_alloc = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()

        return {
            't0': now,
            'last': now,
            'cpu0': time.process_time(),
            'items': [],
            'cuda_alloc0': cuda_alloc,
            'cuda_reserved0': cuda_reserved,
        }

    def _profile_mark(self, prof, name, extra=''):
        """Record end-to-end wall time since the previous profiler mark."""
        if prof is None:
            return

        self._profile_sync_cuda()
        now = time.perf_counter()
        dt_ms = (now - prof['last']) * 1000.0
        prof['last'] = now
        prof['items'].append((name, dt_ms, extra))

        old = self.profile_ema.get(name, dt_ms)
        alpha = self.profile_ema_alpha
        self.profile_ema[name] = (1.0 - alpha) * old + alpha * dt_ms

    def _profile_report(self, prof, total_extra='', output_published=False):
        """Print a compact timing, effective-rate, and CUDA-memory report."""
        if prof is None:
            return

        self._profile_sync_cuda()
        now = time.perf_counter()
        total_ms = (now - prof['t0']) * 1000.0
        cpu_ms = (time.process_time() - prof['cpu0']) * 1000.0
        self.profile_seq += 1

        if output_published:
            self.profile_output_count += 1
            self.profile_output_times.append(now)

        total_key = (
            '__total_output__' if output_published else '__total_skipped__')
        old_total = self.profile_ema.get(total_key, total_ms)
        alpha = self.profile_ema_alpha
        total_ema = (1.0 - alpha) * old_total + alpha * total_ms
        self.profile_ema[total_key] = total_ema

        period_counter = (
            self.profile_output_count if output_published else self.profile_seq)
        should_print = (
            (period_counter % self.profile_period) == 0
            or total_ms >= self.profile_warn_ms
        )
        if not should_print:
            return

        callback_hz = self._profile_rolling_hz(self.profile_callback_times)
        output_hz = self._profile_rolling_hz(self.profile_output_times)
        capacity_hz = 1000.0 / max(total_ema, 1e-9)
        mode = 'accurate-gpu-sync' if self.profile_cuda_sync else 'low-overhead-async'

        lines = [
            (
                '[PROFILE dino_localmapping #{:d}] total={:.2f} ms '
                '(ema={:.2f} ms, capacity={:.2f} Hz){}'
            ).format(
                self.profile_seq,
                total_ms,
                total_ema,
                capacity_hz,
                '  SLOW' if total_ms >= self.profile_warn_ms else '',
            ),
            (
                '  rate: callback={:.2f} Hz, output={:.2f} Hz '
                '(published={}) | mode={}'
            ).format(
                callback_hz,
                output_hz,
                self.profile_output_count,
                mode,
            ),
            '  CPU process: {:.2f} ms ({:.1f}% core-equivalent)'.format(
                cpu_ms,
                cpu_ms / max(total_ms, 1e-9) * 100.0,
            ),
        ]

        if total_extra:
            lines.append('  meta: {}'.format(total_extra))

        sorted_items = sorted(
            prof['items'], key=lambda item: item[1], reverse=True)
        for name, dt_ms, extra in sorted_items[:self.profile_topk]:
            pct = dt_ms / max(total_ms, 1e-9) * 100.0
            ema = self.profile_ema.get(name, dt_ms)
            suffix = ' | {}'.format(extra) if extra else ''
            lines.append(
                '  {:>7.2f} ms ({:>5.1f}%)  ema={:>7.2f}  {}{}'.format(
                    dt_ms, pct, ema, name, suffix))

        accounted_ms = sum(item[1] for item in prof['items'])
        unaccounted_ms = max(0.0, total_ms - accounted_ms)
        if unaccounted_ms >= 0.05:
            lines.append(
                '  {:>7.2f} ms ({:>5.1f}%)  profiler/report gap'.format(
                    unaccounted_ms,
                    unaccounted_ms / max(total_ms, 1e-9) * 100.0,
                ))

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            lines.append(
                (
                    '  CUDA mem: live alloc={:.1f}MB ({:+.1f}), '
                    'reserved={:.1f}MB ({:+.1f}); '
                    'frame peak alloc={:.1f}MB, reserved={:.1f}MB'
                ).format(
                    alloc / 1024**2,
                    (alloc - prof['cuda_alloc0']) / 1024**2,
                    reserved / 1024**2,
                    (reserved - prof['cuda_reserved0']) / 1024**2,
                    peak_alloc / 1024**2,
                    peak_reserved / 1024**2,
                )
            )

        self.get_logger().info('\n'.join(lines))

    def _odom_pose_to_htm(self):
        """
        Build T_odom_pcl directly from nav_msgs/Odometry.pose.pose.

        This assumes the standard Odometry convention:
          - odom_msg.header.frame_id is the parent/odom frame
          - odom_msg.child_frame_id is the child/base or lidar pointcloud frame
          - odom_msg.pose.pose is the pose of child_frame_id expressed in header.frame_id

        The caller already checks that odom_msg.child_frame_id == pcl_msg.header.frame_id,
        so this transform maps pointcloud-frame points into odom_frame.
        """
        pose = self.odom_msg.pose.pose
        p = pose.position
        q = pose.orientation

        R_np = quaternion_xyzw_to_rotation_matrix(
            [q.x, q.y, q.z, q.w]).astype(np.float32)

        T = torch.eye(4, dtype=torch.float32, device=self.device)
        T[:3, :3] = torch.as_tensor(R_np, dtype=torch.float32, device=self.device)
        T[:3, 3] = torch.tensor([p.x, p.y, p.z], dtype=torch.float32, device=self.device)
        return T

    def preprocess_inputs(self, prof=None):
        if self.pcl_msg is None:
            self.get_logger().warning('no pcl msg received', throttle_duration_sec=1.0)
            return None

        pcl_time = self._stamp_to_sec(self.pcl_msg.header.stamp)
        if abs(pcl_time - self.last_update_time) < 1e-3:
            self.get_logger().warning('pcl time problem', throttle_duration_sec=1.0)
            return None

        if self.odom_msg is None:
            self.get_logger().warning('no odom msg received', throttle_duration_sec=1.0)
            return None

        if self.img_msg is None:
            self.get_logger().warning('no img msg received', throttle_duration_sec=1.0)
            return None

        if self.odom_msg.child_frame_id != self.pcl_msg.header.frame_id:
            self.get_logger().warning(
                'need pcls in child frame of odom (got {}, expected {})'.format(
                    self.pcl_msg.header.frame_id, self.odom_msg.child_frame_id),
                throttle_duration_sec=1.0)
            return None

        if self.odom_msg.header.frame_id != self.odom_frame:
            self.get_logger().warning(
                'odom frame changed from {} to {}'.format(
                    self.odom_frame, self.odom_msg.header.frame_id),
                throttle_duration_sec=1.0)
            self.odom_frame = self.odom_msg.header.frame_id

        # Use Odometry.pose.pose directly instead of TF lookup.
        # Because odom_msg.child_frame_id == pcl_msg.header.frame_id,
        # odom_msg.pose.pose gives T_odom_pcl under the standard Odometry convention.
        pcl_htm = self._odom_pose_to_htm()
        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)
        pcl_odom = transform_xyz(pcl, pcl_htm)

        _metadata = {
            'origin': torch.tensor([
                self.odom_msg.pose.pose.position.x - 0.5 * self.base_metadata['length_x'],
                self.odom_msg.pose.pose.position.y - 0.5 * self.base_metadata['length_y'],
            ], dtype=torch.float32),
            'length_x': self.metadata_length_x,
            'length_y': self.metadata_length_y,
            'resolution': self.metadata_resolution,
        }
        self._profile_mark(
            prof,
            'pcl_decode+odom_transform',
            'points={}'.format(pcl.shape[0]),
        )

        self.last_update_time = pcl_time

        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8') / 255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8').astype(np.float32) / 255.

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
        self._profile_mark(
            prof,
            'image_decode+tensor',
            'input={}x{}'.format(img.shape[-2], img.shape[-1]),
        )

        profile_callback = None
        if prof is not None:
            def profile_callback(block, output):
                block_name = block.__class__.__name__
                stage_name = {
                    'Dinov2Block': 'dino_inference',
                    'VLADBlock': 'vlad_residual_projection',
                }.get(block_name, 'image_block:{}'.format(block_name))
                self._profile_mark(
                    prof,
                    stage_name,
                    'output={}'.format(tuple(output.shape)),
                )

        with torch.inference_mode():
            dino_img, dino_intrinsics = self.image_pipeline.run(
                img,
                self.intrinsics.unsqueeze(0),
                profile_callback=profile_callback,
            )

        # Keep the downstream VLAD, projection, and local-map path in FP32.
        dino_img = dino_img.float()
        dino_intrinsics = dino_intrinsics.float()
        dino_img = dino_img[0].permute(1, 2, 0)
        dino_intrinsics = dino_intrinsics[0]
        self._profile_mark(
            prof,
            'image_pipeline_finalize',
            'feature={}x{}x{}, tokens={}'.format(
                dino_img.shape[0],
                dino_img.shape[1],
                dino_img.shape[2],
                dino_img.shape[0] * dino_img.shape[1],
            ),
        )

        projection_signature = (
            int(img.shape[-2]),
            int(img.shape[-1]),
            int(dino_img.shape[0]),
            int(dino_img.shape[1]),
        )
        if self.projection_signature != projection_signature:
            intrinsics_4x4 = torch.eye(
                4,
                dtype=dino_intrinsics.dtype,
                device=dino_intrinsics.device,
            )
            intrinsics_4x4[:3, :3] = dino_intrinsics
            self.projection_matrix = (
                intrinsics_4x4 @ self.extrinsics_device)[:-1, :]
            self.projection_signature = projection_signature
        P = self.projection_matrix

        pixel_coordinates, camera_depth = get_pixel_from_3D_source(
            pcl[:, :3], P, return_depth=True)
        pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3],
            pixel_coordinates,
            dino_img.shape[0],
            dino_img.shape[1],
            return_lidar_points=False,
            camera_depth=camera_depth,
        )

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        visible_xyz = pcl_odom[ind_in_frame, :3]
        dino_pcl_parts = [visible_xyz, dino_features]

        if self.compute_slope:
            # Append z (odom frame) as an extra feature channel to be EMA-accumulated.
            # Slope is derived later from the stable accumulated elevation map.
            dino_pcl_parts.append(visible_xyz[:, [2]])
        dino_pcl = torch.cat(dino_pcl_parts, dim=-1)
        self._profile_mark(
            prof,
            'project+gather_features',
            'visible_points={}'.format(dino_pcl.shape[0]),
        )

        return {
            'metadata': _metadata,
            'dino_image': dino_img,
            'dino_pcl': dino_pcl,
        }

    def update_localmap(self, pcl, metadata):
        pcl_pos = pcl[:, :3]
        pcl_data = pcl[:, 3:]
        metadata = aligned_map_metadata(metadata)
        localmap, known_mask, metadata_out = localmap_from_pointcloud(pcl_pos, pcl_data, metadata)
        localmap_update = {'data': localmap, 'known': known_mask, 'metadata': metadata_out}

        if self.localmap is None:
            return localmap_update
        return aggregate_localmaps(localmap_update, self.localmap, ema=self.localmap_ema)

    def compute_slope_from_elevation(self, localmap):
        """
        Compute slope_longitudinal and slope_lateral from the EMA-accumulated elevation map.
        Uses finite differences on the stable elevation grid rather than noisy single-scan normals.

        Expects localmap['data'][..., -1] to be the EMA-accumulated z (elevation) channel.
        Returns a new localmap with two extra channels: slope_longitudinal, slope_lateral (radians).
        """
        data  = localmap['data']    # [nx, ny, n_channels]
        known = localmap['known']   # [nx, ny]
        metadata = localmap['metadata']

        res = metadata['resolution'].item()

        # Mark unknown cells as NaN so gradients don't bleed across map boundaries
        elev = data[..., -1].clone()  # [nx, ny]
        elev[~known] = float('nan')

        # Robot heading from odom quaternion
        q = self.odom_msg.pose.pose.orientation
        R_mat = quaternion_xyzw_to_rotation_matrix(
            [q.x, q.y, q.z, q.w])
        forward = torch.from_numpy(R_mat[:, 0].copy()).float().to(data.device)
        lateral = torch.from_numpy(R_mat[:, 1].copy()).float().to(data.device)
        forward[2] = 0.
        lateral[2] = 0.
        norm_f = forward.norm()
        norm_l = lateral.norm()

        if norm_f < 1e-6 or norm_l < 1e-6:
            slope_lon = torch.zeros_like(elev)
            slope_lat = torch.zeros_like(elev)
        else:
            forward = forward / norm_f
            lateral = lateral / norm_l

            # Central finite differences: dz/dx (grid axis-0) and dz/dy (grid axis-1)
            dz_dx = torch.zeros_like(elev)
            dz_dx[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2.0 * res)
            dz_dx[0, :]    = (elev[1, :]  - elev[0, :])   / res
            dz_dx[-1, :]   = (elev[-1, :] - elev[-2, :])  / res

            dz_dy = torch.zeros_like(elev)
            dz_dy[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2.0 * res)
            dz_dy[:, 0]    = (elev[:, 1]  - elev[:, 0])   / res
            dz_dy[:, -1]   = (elev[:, -1] - elev[:, -2])  / res

            # NaN from unknown cells → treat as flat (0 gradient) at boundaries
            dz_dx = torch.nan_to_num(dz_dx, nan=0.0)
            dz_dy = torch.nan_to_num(dz_dy, nan=0.0)

            # Project gradient onto robot heading directions (rise / run)
            fx, fy = forward[0], forward[1]
            lx, ly = lateral[0], lateral[1]
            slope_lon = torch.atan(dz_dx * fx + dz_dy * fy).clamp(-0.5, 0.5)
            slope_lat = torch.atan(dz_dx * lx + dz_dy * ly).clamp(-0.5, 0.5)

            slope_lon[~known] = 0.
            slope_lat[~known] = 0.

        new_data = torch.cat([
            data,
            slope_lon.unsqueeze(-1),
            slope_lat.unsqueeze(-1),
        ], dim=-1)

        return {
            'data':     new_data,
            'known':    known,
            'metadata': metadata,
        }

    def np_to_float32_array(self, x: np.ndarray) -> array:
        """
        Convert contiguous float32 numpy array to ROS-compatible array('f')
        without creating a Python float list.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = array('f')
        # array.frombytes accepts any bytes-like object. A byte-cast memoryview
        # avoids allocating a second full bytes copy with ndarray.tobytes().
        out.frombytes(memoryview(x).cast('B'))
        return out

    def make_float32_multiarray(self, mat: np.ndarray) -> Float32MultiArray:
        """
        mat should already be in the final memory order expected by GridMap.
        """
        mat = np.ascontiguousarray(mat, dtype=np.float32)

        msg = Float32MultiArray()

        rows, cols = mat.shape

        # GridMap convention is usually column_index / row_index.
        # If your current consumer depends on the old stride values,
        # keep the old layout temporarily. But this layout is more standard.
        msg.layout.dim.append(
            MultiArrayDimension(
                label='column_index',
                size=cols,
                stride=rows * cols,
            )
        )
        msg.layout.dim.append(
            MultiArrayDimension(
                label='row_index',
                size=rows,
                stride=rows,
            )
        )

        msg.data = self.np_to_float32_array(mat)
        return msg

    def make_gridmap_msg(self, localmap, publish_elevation=True, publish_rgb_viz=True):
        t1 = time.time()

        gridmap_msg = GridMap()

        # CPU transfer
        gridmap_data = localmap['data'].detach().cpu().numpy()
        if gridmap_data.dtype != np.float32:
            gridmap_data = gridmap_data.astype(np.float32, copy=False)

        H, W, C = gridmap_data.shape

        gridmap_msg.header.stamp = self.pcl_msg.header.stamp
        gridmap_msg.header.frame_id = self.odom_frame

        if self.layer_keys is None:
            if self.compute_slope:
                # channels: VLAD×8, z_elevation, slope_longitudinal, slope_lateral
                n_vlad = C - 3
                gridmap_msg.layers = [f'{self.layer_key}_{i}' for i in range(n_vlad)]
                gridmap_msg.layers += [
                    'z_elevation',
                    'slope_longitudinal',
                    'slope_lateral',
                ]
            else:
                gridmap_msg.layers = [f'{self.layer_key}_{i}' for i in range(C)]
        else:
            # deepcopy is usually unnecessary unless self.layer_keys is mutated elsewhere.
            gridmap_msg.layers = list(self.layer_keys)

        info = gridmap_msg.info
        metadata = localmap['metadata']

        info.resolution = float(metadata['resolution'].item())
        info.length_x = float(metadata['length_x'].item())
        info.length_y = float(metadata['length_y'].item())

        origin = metadata['origin']
        info.pose.position.x = float(origin[0].item()) + 0.5 * info.length_x
        info.pose.position.y = float(origin[1].item()) + 0.5 * info.length_y
        info.pose.position.z = self.odom_msg.pose.pose.position.z
        info.pose.orientation.w = 1.0

        t2 = time.time()

        # 기존 코드의 orientation 유지:
        # layer_data[::-1, ::-1].T
        #
        # 기존에는 layer마다 reverse/transpose/flatten을 반복했지만,
        # 여기서는 C개 layer 전체를 한 번에 변환합니다.
        layers_out = np.ascontiguousarray(
            gridmap_data[::-1, ::-1, :].transpose(2, 1, 0),
            dtype=np.float32,
        )

        for i in range(C):
            gridmap_msg.data.append(self.make_float32_multiarray(layers_out[i]))

        t3 = time.time()

        if publish_elevation:
            gridmap_msg.layers.append('elevation')

            elevation_z = self.odom_msg.pose.pose.position.z - 1.73
            # The layer is spatially constant, so constructing it directly in
            # final (transposed) shape is identical to reverse+transpose.
            elevation_out = np.full(
                (W, H), elevation_z, dtype=np.float32)

            gridmap_msg.data.append(self.make_float32_multiarray(elevation_out))

        t4 = time.time()

        if publish_rgb_viz:
            gridmap_msg.layers.append('rgb_viz')

            rgb = gridmap_data[..., :3]

            vmin = rgb.reshape(-1, 3).min(axis=0).reshape(1, 1, 3)
            vmax = rgb.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)

            denom = np.maximum(vmax - vmin, 1e-6)

            rgb_u32 = np.clip((rgb - vmin) / denom * 255.0, 0, 255).astype(np.uint32)

            packed_rgb = (
                (rgb_u32[..., 0] << 16)
                | (rgb_u32[..., 1] << 8)
                | rgb_u32[..., 2]
            )

            # grid_map RViz color layer convention:
            # packed RGB bits are reinterpreted as float32.
            rgb_float = packed_rgb.view(np.float32)

            rgb_out = np.ascontiguousarray(rgb_float[::-1, ::-1].T)

            gridmap_msg.data.append(self.make_float32_multiarray(rgb_out))

        t5 = time.time()

        if self.log_frame_timing:
            self.get_logger().info(
                'Timing in making grid map: \n'
                '\t msg: {:.6f}s\n'
                '\t vlad: {:.6f}s\n'
                '\t elevation: {:.6f}s\n'
                '\t rgb: {:.6f}s'.format(
                    t2 - t1,
                    t3 - t2,
                    t4 - t3,
                    t5 - t4,
                )
            )

        return gridmap_msg

    def make_pcl_msg(self, pcl):
        if pcl.shape[0] == 0:                                             
            return xyz_array_to_point_cloud_msg(                          
                points=np.zeros((0, 3), dtype=np.float32),                
                frame=self.odom_frame,                                    
                timestamp=self.pcl_msg.header.stamp,                      
                rgb_values=np.zeros((0, 3), dtype=np.uint8),              
            )
        pcl_pos = pcl[:, :3].cpu().numpy()
        pcl_cs = pcl[:, [5, 4, 3]]
        vmin = pcl_cs.min(dim=0)[0].view(1, 3)
        vmax = pcl_cs.max(dim=0)[0].view(1, 3)
        denom = (vmax - vmin).clamp(min=1e-6)                             
        pcl_cs = ((pcl_cs - vmin) / denom).cpu().numpy()    

        return xyz_array_to_point_cloud_msg(
            points=pcl_pos,
            frame=self.odom_frame,
            timestamp=self.pcl_msg.header.stamp,
            rgb_values=(pcl_cs * 255.).astype(np.uint8),
        )
    

    def publish_messages(self, res, publish_map=None, prof=None):
        if publish_map is None:
            publish_map = self.localmap
        gridmap_msg = self.make_gridmap_msg(
            publish_map,
            publish_rgb_viz=self.publish_rgb_viz,
        )
        self._profile_mark(
            prof,
            'serialize_gridmap',
            'layers={}'.format(len(gridmap_msg.layers)),
        )
        self.gridmap_pub.publish(gridmap_msg)
        self.scheduling_stats['published'] += 1
        self._profile_mark(prof, 'publish_gridmap')

        if (
                self.publish_dino_pcl
                and self.pcl_pub.get_subscription_count() > 0):
            pcl_for_viz = res['dino_pcl'][::self.pcl_viz_stride]
            pcl_msg = self.make_pcl_msg(pcl_for_viz)
            self.pcl_pub.publish(pcl_msg)
            self._profile_mark(
                prof,
                'dino_pcl_viz+publish',
                'points={}'.format(pcl_for_viz.shape[0]),
            )

        if (
                self.publish_dino_image
                and self.image_pub.get_subscription_count() > 0):
            vmin = self.localmap['data'][..., :3].view(
                -1, 3).min(dim=0)[0].view(1, 1, 3)
            vmax = self.localmap['data'][..., :3].view(
                -1, 3).max(dim=0)[0].view(1, 1, 3)
            denom = (vmax - vmin).clamp(min=1e-6)
            viz_img = (
                (res['dino_image'][..., :3] - vmin) / denom
            ).clip(0., 1.).cpu().numpy() * 255

            img_msg = self.bridge.cv2_to_imgmsg(
                viz_img.astype(np.uint8), 'rgb8')
            img_msg.header.stamp = self.img_msg.header.stamp
            self.image_pub.publish(img_msg)
            self._profile_mark(prof, 'dino_image_viz+publish')

    def spin_once(self):
        prof = self._profile_start()
        if self.log_frame_timing:
            self.get_logger().info('processing synchronized inputs...')

        t1 = time.time()
        res = self.preprocess_inputs(prof)
        t2 = time.time()

        if res:
            if self.log_frame_timing:
                self.get_logger().info('updating localmap...')
            t3 = time.time()
            self.localmap = self.update_localmap(pcl=res['dino_pcl'], metadata=res['metadata'])
            self._profile_mark(
                prof,
                'localmap_update',
                'grid={}'.format(tuple(self.localmap['data'].shape)),
            )
            t4 = time.time()
            publish_map = self.localmap
            if self.compute_slope and self.localmap is not None:
                publish_map = self.compute_slope_from_elevation(self.localmap)
                self._profile_mark(prof, 'slope_compute')
            self.publish_messages(res, publish_map, prof)
            t5 = time.time()
            if self.log_frame_timing:
                self.get_logger().info(
                    'Timing:\n\tpreproc: {:.6f}s\n\tlocalmap: {:.6f}s\n\tserialize: {:.6f}s'.format(
                        t2 - t1, t4 - t3, t5 - t4))
            self._profile_report(
                prof,
                'visible_points={}, map={}'.format(
                    res['dino_pcl'].shape[0],
                    tuple(publish_map['data'].shape),
                ),
                output_published=True,
            )

        else:
            if self.log_frame_timing:
                self.get_logger().info('skipped because no result')
            self._profile_report(
                prof, 'output=skipped', output_published=False)

def main(args=None):
    rclpy.init(args=args)

    node_handle = rclpy.create_node('_config_reader')
    node_handle.declare_parameter('config_fp', '')
    node_handle.declare_parameter('input_reliability', '')
    config_fp = node_handle.get_parameter('config_fp').value
    reliability_override = node_handle.get_parameter('input_reliability').value
    node_handle.destroy_node()

    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)
    if reliability_override:
        config.setdefault('sync', {})['input_reliability'] = reliability_override

    node = DinoMappingNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
