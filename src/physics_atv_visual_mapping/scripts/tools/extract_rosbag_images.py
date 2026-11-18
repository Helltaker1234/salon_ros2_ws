#!/usr/bin/env python3
"""Extract time-spaced JPEG images from a ROS 2 bag image topic."""

import argparse
from pathlib import Path
import re

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, help='ROS 2 bag directory')
    parser.add_argument(
        '--topic',
        default='/rough/D435_1/color/image_raw/compressed',
        help='sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage topic',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--interval-sec',
        type=float,
        default=0.5,
        help='Minimum bag-time interval between saved images',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Maximum images to save; 0 means unlimited',
    )
    return parser.parse_args()


def detect_storage_id(bag_dir):
    metadata_path = Path(bag_dir).expanduser().resolve() / 'metadata.yaml'
    if not metadata_path.is_file():
        raise RuntimeError(f'bag metadata not found: {metadata_path}')
    match = re.search(
        r'^\s*storage_identifier:\s*(\S+)\s*$',
        metadata_path.read_text(encoding='utf-8'),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(
            f'storage_identifier not found in {metadata_path}')
    return match.group(1)


def decode_raw_image(msg):
    encoding = msg.encoding.lower()
    encoding_info = {
        'bgr8': (3, None),
        'rgb8': (3, cv2.COLOR_RGB2BGR),
        'bgra8': (4, cv2.COLOR_BGRA2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
        'mono8': (1, cv2.COLOR_GRAY2BGR),
        '8uc1': (1, cv2.COLOR_GRAY2BGR),
        '8uc3': (3, None),
        '8uc4': (4, cv2.COLOR_BGRA2BGR),
    }
    if encoding not in encoding_info:
        raise RuntimeError(f'unsupported raw image encoding: {msg.encoding}')

    channels, color_conversion = encoding_info[encoding]
    row_bytes = int(msg.step)
    useful_row_bytes = int(msg.width) * channels
    if row_bytes < useful_row_bytes:
        raise RuntimeError(
            f'invalid image step: step={row_bytes}, '
            f'width={msg.width}, encoding={msg.encoding}')

    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    expected_bytes = int(msg.height) * row_bytes
    if encoded.size < expected_bytes:
        raise RuntimeError(
            f'truncated image: bytes={encoded.size}, expected={expected_bytes}')

    image = encoded[:expected_bytes].reshape(int(msg.height), row_bytes)
    image = image[:, :useful_row_bytes].reshape(
        int(msg.height), int(msg.width), channels)
    if color_conversion is not None:
        image = cv2.cvtColor(image, color_conversion)
    return image


def decode_image(msg, message_type):
    if message_type == 'sensor_msgs/msg/CompressedImage':
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if message_type == 'sensor_msgs/msg/Image':
        return decode_raw_image(msg)
    raise RuntimeError(
        f'unsupported topic type: {message_type}; expected '
        'sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage')


def main():
    args = parse_args()
    if args.interval_sec <= 0:
        raise ValueError('--interval-sec must be positive')
    if args.max_images < 0:
        raise ValueError('--max-images cannot be negative')

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    storage_id = detect_storage_id(args.bag)
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''),
    )

    topic_types = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }
    if args.topic not in topic_types:
        available = '\n'.join(sorted(topic_types))
        raise RuntimeError(
            f'topic not found: {args.topic}\navailable topics:\n{available}')

    actual_type = topic_types[args.topic]
    supported_types = {
        'sensor_msgs/msg/Image',
        'sensor_msgs/msg/CompressedImage',
    }
    if actual_type not in supported_types:
        raise RuntimeError(
            f'{args.topic} has unsupported type {actual_type}; expected one of '
            f'{sorted(supported_types)}')

    msg_type = get_message(actual_type)
    interval_ns = int(args.interval_sec * 1_000_000_000)
    last_saved_ns = None
    saved = 0
    seen = 0
    decode_failures = 0

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic != args.topic:
            continue

        seen += 1
        if (
                last_saved_ns is not None
                and bag_time_ns - last_saved_ns < interval_ns):
            continue

        msg = deserialize_message(serialized, msg_type)
        try:
            image = decode_image(msg, actual_type)
        except RuntimeError as exc:
            decode_failures += 1
            if decode_failures <= 5:
                print(f'decode failure: {exc}')
            continue
        if image is None:
            decode_failures += 1
            continue

        filename = output_dir / f'{saved:06d}_{bag_time_ns}.jpg'
        if not cv2.imwrite(
                str(filename),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f'failed to write {filename}')

        last_saved_ns = bag_time_ns
        saved += 1
        if args.max_images and saved >= args.max_images:
            break

    print(f'topic={args.topic}')
    print(f'topic_type={actual_type}')
    print(f'storage_id={storage_id}')
    print(f'seen={seen}')
    print(f'saved={saved}')
    print(f'decode_failures={decode_failures}')
    print(f'output_dir={output_dir}')


if __name__ == '__main__':
    main()
