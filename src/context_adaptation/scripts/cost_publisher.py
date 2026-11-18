#!/usr/bin/env python3

import os

import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

from scipy.signal import welch
from scipy.integrate import simpson

from ament_index_python.packages import get_package_share_directory


class Buffer:
    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = int(buffer_size)

        if not padded:
            self._data = []
        else:
            assert pad_val is not None
            self._data = [pad_val] * self.buffer_size

        self.data = np.array(self._data, dtype=np.float64)

    def insert(self, data_point):
        self._data.append(float(data_point))

        if len(self._data) > self.buffer_size:
            self._data = self._data[-self.buffer_size:]

        self.data = np.array(self._data, dtype=np.float64)


def bandpower(data, sf, band, welch_window_sec=None, relative=False):
    """
    Welch PSD를 계산한 뒤 지정 주파수 band의 PSD를 적분해 Band Power 계산.
    """
    data = np.asarray(data, dtype=np.float64)
    low, high = np.asarray(band, dtype=np.float64)

    if welch_window_sec is None:
        nperseg = None
    else:
        nperseg = int(round(welch_window_sec * sf))
        nperseg = min(nperseg, len(data))

    freqs, psd = welch(
        data,
        fs=sf,
        window='hann',
        nperseg=nperseg,
        noverlap=(nperseg // 2) if nperseg is not None else None,
        detrend='constant',
        scaling='density',
    )

    idx_band = np.logical_and(
        freqs >= low,
        freqs <= high,
    )

    if np.count_nonzero(idx_band) < 2:
        return 0.0

    bp = simpson(
        psd[idx_band],
        x=freqs[idx_band],
    )

    if relative:
        total_power = simpson(
            psd,
            x=freqs,
        )

        if total_power > 0.0:
            bp /= total_power

    return float(bp)


def normalize_cost(value, stats):
    """
    min/max를 기준으로 0~1 Cost 계산.
    """
    min_value = float(stats['min'])
    max_value = float(stats['max'])

    denom = max_value - min_value

    if denom <= 0.0:
        return 0.0

    cost = (
        float(value) - min_value
    ) / denom

    return float(
        np.clip(
            cost,
            0.0,
            1.0,
        )
    )


def legacy_cost_function(
    data,
    sensor_freq,
    cost_name,
    cost_stats,
    freq_range=None,
    num_bins=None,
):
    """
    기존 코드의 cost_function 동작을 최대한 그대로 유지.
    """
    if 'bins' in cost_name:
        assert num_bins is not None

        freq_width = (
            sensor_freq // 2
        ) / num_bins

        bps = [
            bandpower(
                data,
                sensor_freq,
                [
                    i * freq_width + 1,
                    (i + 1) * freq_width + 1,
                ],
                welch_window_sec=None,
            )
            for i in range(num_bins)
        ]

        cost = np.mean(bps)

    elif 'band' in cost_name:
        assert freq_range is not None

        cost = bandpower(
            data,
            sensor_freq,
            freq_range,
            welch_window_sec=None,
        )

    else:
        raise NotImplementedError(
            'cost_name needs to include bins or band'
        )

    return normalize_cost(
        cost,
        cost_stats,
    )


class TraversabilityCostNode(Node):

    def __init__(self):
        super().__init__(
            'traversability_cost_publisher'
        )

        # =============================================================
        # ROS parameters
        # =============================================================

        self.declare_parameter(
            'cost_stats_dir',
            'wanda_cost_statistics.yaml',
        )

        self.declare_parameter(
            'imu_topic',
            '/imu/data',
        )

        # True  = 기존 알고리즘
        # False = 새 분석 기반 알고리즘
        self.declare_parameter(
            'use_legacy_mode',
            False,
        )

        # 새 알고리즘에서 Z / Roll 결합 weight
        self.declare_parameter(
            'optimized_z_weight',
            0.09,
        )

        self.declare_parameter(
            'optimized_roll_weight',
            0.91,
        )

        cost_stats_file = self.get_parameter(
            'cost_stats_dir'
        ).value

        imu_topic = self.get_parameter(
            'imu_topic'
        ).value

        self.use_legacy_mode = bool(
            self.get_parameter(
                'use_legacy_mode'
            ).value
        )

        self.optimized_z_weight = float(
            self.get_parameter(
                'optimized_z_weight'
            ).value
        )

        self.optimized_roll_weight = float(
            self.get_parameter(
                'optimized_roll_weight'
            ).value
        )

        weight_sum = (
            self.optimized_z_weight
            + self.optimized_roll_weight
        )

        if weight_sum <= 0.0:
            self.optimized_z_weight = 0.5
            self.optimized_roll_weight = 0.5
        else:
            self.optimized_z_weight /= weight_sum
            self.optimized_roll_weight /= weight_sum

        if not os.path.isabs(
            cost_stats_file
        ):
            assets_dir = os.path.join(
                get_package_share_directory(
                    'context_adaptation'
                ),
                'assets',
                'cost_configs',
            )

            cost_stats_file = os.path.join(
                assets_dir,
                cost_stats_file,
            )

        # =============================================================
        # Publisher / Subscriber
        # =============================================================

        self.cost_publisher = (
            self.create_publisher(
                Float32,
                '/traversability_cost',
                10,
            )
        )

        # 디버깅용 개별 Cost
        self.z_cost_publisher = (
            self.create_publisher(
                Float32,
                '/traversability_cost/z',
                10,
            )
        )

        self.roll_cost_publisher = (
            self.create_publisher(
                Float32,
                '/traversability_cost/roll',
                10,
            )
        )

        # 디버깅용 실제 Band Power
        self.z_bp_publisher = (
            self.create_publisher(
                Float32,
                '/traversability_cost/z_bandpower',
                10,
            )
        )

        self.roll_bp_publisher = (
            self.create_publisher(
                Float32,
                '/traversability_cost/roll_bandpower',
                10,
            )
        )

        self.create_subscription(
            Imu,
            imu_topic,
            self.handle_imu,
            1,
        )

        # =============================================================
        # IMU configuration
        # =============================================================

        self.imu_freq = 100.0

        # =============================================================
        # YAML
        # =============================================================

        with open(
            cost_stats_file,
            'r',
            encoding='utf-8',
        ) as f:
            self.all_costs_stats = (
                yaml.safe_load(f)
            )

        # =============================================================
        # LEGACY MODE
        # =============================================================

        if self.use_legacy_mode:

            # 기존 코드 그대로 1초
            self.buffer_size = int(
                self.imu_freq
            )

            self.buffer_z = Buffer(
                self.buffer_size,
                padded=True,
                pad_val=9.81,
            )

            self.buffer_roll = Buffer(
                self.buffer_size,
                padded=True,
                pad_val=0.0,
            )

            self.legacy_cost_name = (
                'freq_band_1_30'
            )

            self.legacy_stats = (
                self.all_costs_stats[
                    self.legacy_cost_name
                ]
            )

            self.legacy_min_freq = 3.0
            self.legacy_max_freq = 20.0

            self.get_logger().info(
                'Traversability mode: LEGACY'
            )

            self.get_logger().info(
                'Legacy buffer: 1.0 sec'
            )

            self.get_logger().info(
                'Legacy band: Z/Roll = 3.0~20.0 Hz'
            )

            self.get_logger().info(
                'Legacy fusion: C = Cz + 100 * Croll'
            )

        # =============================================================
        # OPTIMIZED MODE
        # =============================================================

        else:

            optimized_cfg = (
                self.all_costs_stats[
                    'optimized'
                ]
            )

            analysis_cfg = (
                optimized_cfg.get(
                    'analysis',
                    {}
                )
            )

            self.optimized_buffer_sec = float(
                analysis_cfg.get(
                    'buffer_sec',
                    4.0,
                )
            )

            self.optimized_welch_window_sec = float(
                analysis_cfg.get(
                    'welch_window_sec',
                    2.0,
                )
            )

            self.buffer_size = int(
                round(
                    self.imu_freq
                    * self.optimized_buffer_sec
                )
            )

            self.buffer_z = Buffer(
                self.buffer_size,
                padded=True,
                pad_val=9.81,
            )

            self.buffer_roll = Buffer(
                self.buffer_size,
                padded=True,
                pad_val=0.0,
            )

            # Z optimized parameters
            z_cfg = optimized_cfg[
                'accel_z'
            ]

            self.z_band = [
                float(
                    z_cfg[
                        'min_freq'
                    ]
                ),
                float(
                    z_cfg[
                        'max_freq'
                    ]
                ),
            ]

            self.z_stats = {
                'min': float(
                    z_cfg['min']
                ),
                'max': float(
                    z_cfg['max']
                ),
            }

            # Roll optimized parameters
            roll_cfg = optimized_cfg[
                'roll_gx'
            ]

            self.roll_band = [
                float(
                    roll_cfg[
                        'min_freq'
                    ]
                ),
                float(
                    roll_cfg[
                        'max_freq'
                    ]
                ),
            ]

            self.roll_stats = {
                'min': float(
                    roll_cfg['min']
                ),
                'max': float(
                    roll_cfg['max']
                ),
            }

            self.get_logger().info(
                'Traversability mode: OPTIMIZED'
            )

            self.get_logger().info(
                f'Optimized buffer: '
                f'{self.optimized_buffer_sec:.1f} sec'
            )

            self.get_logger().info(
                f'Welch window: '
                f'{self.optimized_welch_window_sec:.1f} sec'
            )

            self.get_logger().info(
                f'Accel Z band: '
                f'{self.z_band[0]:.1f}~'
                f'{self.z_band[1]:.1f} Hz'
            )

            self.get_logger().info(
                f'Roll band: '
                f'{self.roll_band[0]:.1f}~'
                f'{self.roll_band[1]:.1f} Hz'
            )

            self.get_logger().info(
                'Optimized fusion: '
                f'C = '
                f'{self.optimized_z_weight:.3f} * Cz + '
                f'{self.optimized_roll_weight:.3f} * Croll'
            )

        self.get_logger().info(
            'Initialized traversability_cost_publisher node'
        )

    # =================================================================
    # Legacy
    # =================================================================

    def calculate_legacy_cost(self):

        cost_z = legacy_cost_function(
            self.buffer_z.data,
            self.imu_freq,
            self.legacy_cost_name,
            self.legacy_stats,
            freq_range=[
                self.legacy_min_freq,
                self.legacy_max_freq,
            ],
        )

        cost_roll = legacy_cost_function(
            self.buffer_roll.data,
            self.imu_freq,
            self.legacy_cost_name,
            self.legacy_stats,
            freq_range=[
                self.legacy_min_freq,
                self.legacy_max_freq,
            ],
        )

        # 원래 코드 그대로 유지
        total_cost = (
            cost_z
            + cost_roll * 100.0
        )

        return (
            float(total_cost),
            float(cost_z),
            float(cost_roll),
            0.0,
            0.0,
        )

    # =================================================================
    # Optimized
    # =================================================================

    def calculate_optimized_cost(self):

        # -------------------------------------------------------------
        # Accel Z
        # -------------------------------------------------------------

        z_bp = bandpower(
            self.buffer_z.data,
            self.imu_freq,
            self.z_band,
            welch_window_sec=(
                self.optimized_welch_window_sec
            ),
        )

        cost_z = normalize_cost(
            z_bp,
            self.z_stats,
        )

        # -------------------------------------------------------------
        # Roll
        # -------------------------------------------------------------

        roll_bp = bandpower(
            self.buffer_roll.data,
            self.imu_freq,
            self.roll_band,
            welch_window_sec=(
                self.optimized_welch_window_sec
            ),
        )

        cost_roll = normalize_cost(
            roll_bp,
            self.roll_stats,
        )

        # -------------------------------------------------------------
        # Weighted fusion
        # -------------------------------------------------------------

        total_cost = (
            self.optimized_z_weight
            * cost_z
            +
            self.optimized_roll_weight
            * cost_roll
        )

        # 새 mode에서는 최종 Cost도 0~1 유지
        total_cost = float(
            np.clip(
                total_cost,
                0.0,
                1.0,
            )
        )

        return (
            total_cost,
            float(cost_z),
            float(cost_roll),
            float(z_bp),
            float(roll_bp),
        )

    # =================================================================
    # IMU callback
    # =================================================================

    def handle_imu(self, msg):

        self.buffer_z.insert(
            msg.linear_acceleration.z
        )

        self.buffer_roll.insert(
            msg.angular_velocity.x
        )

        if self.use_legacy_mode:
            (
                total_cost,
                cost_z,
                cost_roll,
                z_bp,
                roll_bp,
            ) = self.calculate_legacy_cost()

        else:
            (
                total_cost,
                cost_z,
                cost_roll,
                z_bp,
                roll_bp,
            ) = self.calculate_optimized_cost()

        # Main Cost
        msg_total = Float32()
        msg_total.data = float(
            total_cost
        )
        self.cost_publisher.publish(
            msg_total
        )

        # Debug: individual costs
        msg_z = Float32()
        msg_z.data = float(
            cost_z
        )
        self.z_cost_publisher.publish(
            msg_z
        )

        msg_roll = Float32()
        msg_roll.data = float(
            cost_roll
        )
        self.roll_cost_publisher.publish(
            msg_roll
        )

        # Debug: band power only in optimized mode
        if not self.use_legacy_mode:
            msg_z_bp = Float32()
            msg_z_bp.data = float(
                z_bp
            )
            self.z_bp_publisher.publish(
                msg_z_bp
            )

            msg_roll_bp = Float32()
            msg_roll_bp.data = float(
                roll_bp
            )
            self.roll_bp_publisher.publish(
                msg_roll_bp
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = TraversabilityCostNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()