#!/usr/bin/env python3
"""Inspect VLAD value distributions from grid_map_msgs/GridMap messages."""

import argparse

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/dino_gridmap')
    parser.add_argument('--messages', type=int, default=10)
    return parser.parse_args()


class VlalGridMapInspector(Node):
    def __init__(self, topic, message_count):
        super().__init__('vlad_gridmap_inspector')
        self.message_count = message_count
        self.received = 0
        self.samples = []
        self.subscription = self.create_subscription(
            GridMap,
            topic,
            self.callback,
            10,
        )
        self.get_logger().info(
            f'Collecting {message_count} messages from {topic}')

    def callback(self, msg):
        layer_indices = [
            index for index, name in enumerate(msg.layers)
            if name.startswith('VLAD_')
        ]
        if len(layer_indices) != 8:
            self.get_logger().error(
                f'Expected 8 VLAD layers, got {len(layer_indices)}: '
                f'{list(msg.layers)}')
            rclpy.shutdown()
            return

        layer_values = []
        for index in layer_indices:
            values = np.asarray(msg.data[index].data, dtype=np.float32)
            layer_values.append(values)

        vlad = np.stack(layer_values, axis=0)
        finite = np.isfinite(vlad).all(axis=0)
        known = np.any(np.abs(vlad) > 1.0e-8, axis=0)
        valid = finite & known
        if np.any(valid):
            self.samples.append(vlad[:, valid])

        self.received += 1
        self.get_logger().info(
            f'message={self.received}/{self.message_count} '
            f'valid_cells={int(valid.sum())}/{valid.size}')
        if self.received >= self.message_count:
            self.report()
            rclpy.shutdown()

    def report(self):
        if not self.samples:
            print('No finite, nonzero VLAD cells were found.')
            return

        values = np.concatenate(self.samples, axis=1)
        percentiles = [1, 5, 50, 90, 95, 99]

        print('\n=== ViT-S VLAD statistics ===')
        print(f'valid_samples={values.shape[1]}')
        for channel in range(values.shape[0]):
            stats = np.percentile(values[channel], percentiles)
            formatted = ' '.join(
                f'p{p}={value:.6f}'
                for p, value in zip(percentiles, stats)
            )
            print(
                f'VLAD_{channel}: min={values[channel].min():.6f} '
                f'{formatted} max={values[channel].max():.6f}')

        all_values = values.reshape(-1)
        pixel_min = values.min(axis=0)
        all_stats = np.percentile(all_values, percentiles)
        min_stats = np.percentile(pixel_min, percentiles)
        print(
            'all_channels: '
            + ' '.join(
                f'p{p}={value:.6f}'
                for p, value in zip(percentiles, all_stats)
            )
        )
        print(
            'pixel_cluster_min: '
            + ' '.join(
                f'p{p}={value:.6f}'
                for p, value in zip(percentiles, min_stats)
            )
        )
        print(
            'initial_residual_max_candidate='
            f'{np.percentile(all_values, 99):.6f}')


def main():
    args = parse_args()
    if args.messages < 1:
        raise ValueError('--messages must be positive')

    rclpy.init()
    node = VlalGridMapInspector(args.topic, args.messages)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
