#!/usr/bin/env python3
"""Build VLAD cluster centers from an image directory using DINOv2."""

import argparse
import json
import math
from pathlib import Path
import random

import cv2
import fast_pytorch_kmeans as fpk
import numpy as np
import torch
import torch.nn.functional as F

from physics_atv_visual_mapping.image_processing.anyloc_utils import (
    DinoV2ExtractFeatures,
)


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='dinov2_vits14')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument(
        '--facet',
        choices=['query', 'key', 'value', 'token'],
        default='value',
    )
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--height', type=int, default=252)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--max-descriptors', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def list_images(image_dir):
    return sorted(
        path for path in image_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb_tensor(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f'failed to decode image: {path}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)


def main():
    args = parse_args()
    if args.width % 14 or args.height % 14:
        raise ValueError('--width and --height must be multiples of 14')
    if args.clusters < 2:
        raise ValueError('--clusters must be at least 2')
    if args.max_descriptors < args.clusters:
        raise ValueError('--max-descriptors must be >= --clusters')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f'no images found in {image_dir}')

    extractor = DinoV2ExtractFeatures(
        str(Path.home() / '.cache' / 'torch' / 'hub'),
        dino_model=args.model,
        layer=args.layer,
        input_size=(args.width, args.height),
        facet=args.facet,
        norm_descs=True,
        device=args.device,
        use_fp16=False,
    )

    descriptors = []
    total_descriptors = 0
    per_image_limit = max(
        args.clusters,
        math.ceil(args.max_descriptors / len(images)),
    )
    decode_failures = 0
    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            try:
                image = load_rgb_tensor(path)
            except RuntimeError as exc:
                print(exc)
                decode_failures += 1
                continue

            features = extractor(image)
            flat = features[0].permute(1, 2, 0).reshape(
                -1, features.shape[1])
            total_descriptors += int(flat.shape[0])

            # Bound peak host memory before concatenation. Sampling a similar
            # number from every image also prevents long or repetitive bags
            # from dominating the K-means input.
            if flat.shape[0] > per_image_limit:
                indices = torch.randperm(
                    flat.shape[0],
                    device=flat.device,
                )[:per_image_limit]
                flat = flat[indices]
            descriptors.append(flat.detach().cpu())

            if index == 1 or index % 25 == 0 or index == len(images):
                print(
                    f'extracted={index}/{len(images)} '
                    f'feature_shape={tuple(features.shape)}')

    if not descriptors:
        raise RuntimeError('no descriptors were extracted')

    all_descriptors = torch.cat(descriptors, dim=0).float()
    sampled_descriptors = int(all_descriptors.shape[0])
    if sampled_descriptors > args.max_descriptors:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(
            sampled_descriptors,
            generator=generator,
        )[:args.max_descriptors]
        all_descriptors = all_descriptors[indices]

    all_descriptors = F.normalize(all_descriptors, dim=1)
    used_descriptors = int(all_descriptors.shape[0])
    train_descriptors = all_descriptors.to(args.device)

    print(
        f'kmeans_input={tuple(train_descriptors.shape)} '
        f'clusters={args.clusters} mode=cosine')
    kmeans = fpk.KMeans(
        n_clusters=args.clusters,
        mode='cosine',
        verbose=1,
    )
    kmeans.fit(train_descriptors)
    centers = kmeans.centroids.detach().float().cpu()

    expected_dim = 384 if args.model == 'dinov2_vits14' else None
    if centers.shape[0] != args.clusters:
        raise RuntimeError(f'unexpected center shape: {tuple(centers.shape)}')
    if expected_dim is not None and centers.shape[1] != expected_dim:
        raise RuntimeError(f'unexpected center shape: {tuple(centers.shape)}')
    if not torch.isfinite(centers).all():
        raise RuntimeError('cluster centers contain NaN or Inf')

    output_dir.mkdir(parents=True, exist_ok=True)
    center_path = output_dir / 'c_centers.pt'
    torch.save(centers, center_path)

    metadata = {
        'model': args.model,
        'layer': args.layer,
        'facet': args.facet,
        'input_width': args.width,
        'input_height': args.height,
        'feature_grid_width': args.width // 14,
        'feature_grid_height': args.height // 14,
        'clusters': args.clusters,
        'descriptor_dim': int(centers.shape[1]),
        'images_found': len(images),
        'decode_failures': decode_failures,
        'descriptors_extracted': total_descriptors,
        'descriptors_sampled_before_final_limit': sampled_descriptors,
        'descriptors_used': used_descriptors,
        'seed': args.seed,
        'source_image_dir': str(image_dir),
    }
    metadata_path = output_dir / 'center_metadata.json'
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8',
    )

    print(f'center_shape={tuple(centers.shape)}')
    print(f'center_dtype={centers.dtype}')
    print(f'center_path={center_path}')
    print(f'metadata_path={metadata_path}')


if __name__ == '__main__':
    main()
