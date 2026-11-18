#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
import numpy as np
import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simpson
import os
import yaml
from ament_index_python.packages import get_package_share_directory


class Buffer:
    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = buffer_size
        if not padded:
            self._data = []
        else:
            assert pad_val is not None
            self._data = [pad_val] * buffer_size
        self.data = np.array(self._data)

    def insert(self, data_point):
        self._data.append(data_point)
        if len(self._data) > self.buffer_size:
            self._data = self._data[1:]
        self.data = np.array(self._data)


def bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band
    nperseg = window_sec * sf if window_sec is not None else None
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simpson(psd[idx_band], dx=freq_res)
    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp


def cost_function(data, sensor_freq, cost_name, cost_stats, freq_range=None, num_bins=None):
    if 'bins' in cost_name:
        assert num_bins is not None
        freq_width = (sensor_freq // 2) / num_bins
        bps = [bandpower(data, sensor_freq,
                         [i * freq_width + 1, (i + 1) * freq_width + 1]) for i in range(num_bins)]
        cost = np.mean(bps)
    elif 'band' in cost_name:
        assert freq_range is not None
        cost = bandpower(data, sensor_freq, freq_range)
    else:
        raise NotImplementedError('cost_name needs to include bins or band')

    cost = (cost - cost_stats['min']) / (cost_stats['max'] - cost_stats['min'])
    return max(min(cost, 1), 0)


class TraversabilityCostNode(Node):
    def __init__(self):
        super().__init__('traversability_cost_publisher')

        self.declare_parameter('cost_stats_dir', 'wanda_cost_statistics.yaml')
        self.declare_parameter('imu_topic', '/imu/data')
        cost_stats_file = self.get_parameter('cost_stats_dir').value
        imu_topic = self.get_parameter('imu_topic').value
        if not os.path.isabs(cost_stats_file):
            assets_dir = os.path.join(get_package_share_directory('context_adaptation'), 'assets', 'cost_configs')
            cost_stats_file = os.path.join(assets_dir, cost_stats_file)

        self.cost_publisher = self.create_publisher(Float32, '/traversability_cost', 10)
        self.create_subscription(Imu, imu_topic, self.handle_imu, 1)

        self.imu_freq = 100
        self.buffer_size = self.imu_freq
        self.buffer = Buffer(self.buffer_size, padded=True, pad_val=9.81)
        self.bufferX = Buffer(self.buffer_size, padded=True, pad_val=0.0)

        with open(cost_stats_file, 'r') as f:
            self.all_costs_stats = yaml.safe_load(f)

        self.cost_name = 'freq_band_1_30'
        self.cost_stats = self.all_costs_stats[self.cost_name]
        self.min_freq = 3
        self.max_freq = 20

        self.get_logger().info('Initialized traversability_cost_publisher node')

    def handle_imu(self, msg):
        self.buffer.insert(msg.linear_acceleration.z)
        self.bufferX.insert(msg.angular_velocity.x)
        cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats,
                             freq_range=[self.min_freq, self.max_freq])
        costX = cost_function(self.bufferX.data, self.imu_freq, self.cost_name, self.cost_stats,
                              freq_range=[self.min_freq, self.max_freq])
        cost = cost + costX * 100
        cost_msg = Float32()
        cost_msg.data = float(cost)
        self.cost_publisher.publish(cost_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityCostNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
