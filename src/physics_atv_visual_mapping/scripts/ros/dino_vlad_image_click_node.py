#!/usr/bin/env python3

import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

import torch
import cv_bridge

from sensor_msgs.msg import Image, CompressedImage

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline


class DinoVladImageClickNode(Node):
    def __init__(self, config):
        super().__init__('dino_vlad_image_click_node')

        self.config = config
        self.device = config['device']

        self.bridge = cv_bridge.CvBridge()
        self.image_pipeline = setup_image_pipeline(config)

        self.compressed_img = config['image']['image_compressed']
        self.image_topic = config['image']['image_topic']

        if 'rect' in self.image_topic:
            self.intrinsics = torch.tensor(
                config['intrinsics']['P'],
                device=self.device
            ).reshape(3, 3).float()
        else:
            self.intrinsics = torch.tensor(
                config['intrinsics']['K'],
                device=self.device
            ).reshape(3, 3).float()

        click_cfg = config.get('click_vlad', {})
        dino_cfg = next(
            block['args']
            for block in config['image_processing']
            if block['type'] == 'dino'
        )
        vlad_cfg = next(
            block['args']
            for block in config['image_processing']
            if block['type'] == 'vlad'
        )

        self.window_name = 'DINO_VLAD_IMAGE_CLICK'

        # feature map에서 평균낼 반경
        self.click_radius_feat = int(click_cfg.get('radius_feat', 1))

        # 원본 이미지를 너무 크게/작게 보일 때 조절
        self.display_scale = float(click_cfg.get('display_scale', 1.0))

        self.residual_max = float(click_cfg.get('residual_max', 80.0))

        self.latest_dino_img_np = None      # [H_feat, W_feat, C]
        self.latest_rgb_uint8 = None        # [H_img, W_img, 3]
        self.latest_display_shape = None    # [H_display, W_display]

        msg_type = CompressedImage if self.compressed_img else Image

        self.create_subscription(
            msg_type,
            self.image_topic,
            self.handle_image,
            1
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.handle_click)

        self.get_logger().info('DINO VLAD image click node started.')
        self.get_logger().info(f'image_topic       : {self.image_topic}')
        self.get_logger().info(f'compressed        : {self.compressed_img}')
        self.get_logger().info(f'radius_feat       : {self.click_radius_feat}')
        self.get_logger().info(f'display_scale     : {self.display_scale}')
        self.get_logger().info(f'residual_max      : {self.residual_max}')
        self.get_logger().info(
            'DINO               : {} layer={} facet={} input={}'.format(
                dino_cfg['dino_type'],
                dino_cfg['dino_layer'],
                dino_cfg.get('desc_facet', 'token'),
                dino_cfg['image_insize'],
            )
        )
        self.get_logger().info(
            'VLAD               : clusters={} center={}'.format(
                vlad_cfg['n_clusters'],
                vlad_cfg['cache_dir'],
            )
        )

        print('\n[INFO] OpenCV 창에서 마우스 왼쪽 클릭하면 VLAD feature가 출력됩니다.\n', flush=True)

    def handle_image(self, msg):
        if self.compressed_img:
            rgb = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding='rgb8'
            )
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='rgb8'
            ).astype(np.float32) / 255.0

        # 원본 RGB 저장
        self.latest_rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)

        img_t = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2)

        with torch.no_grad():
            dino_img, _ = self.image_pipeline.run(
                img_t,
                self.intrinsics.unsqueeze(0)
            )

        # dino_img: [1, C, H_feat, W_feat] -> [H_feat, W_feat, C]
        dino_img = dino_img[0].permute(1, 2, 0)
        self.latest_dino_img_np = dino_img.detach().cpu().numpy()

        H_img, W_img = self.latest_rgb_uint8.shape[:2]
        H_feat, W_feat, C = self.latest_dino_img_np.shape

        # 사람이 보기 좋은 원본 이미지 기반 display
        if self.display_scale != 1.0:
            disp_w = int(W_img * self.display_scale)
            disp_h = int(H_img * self.display_scale)
            display_img = cv2.resize(
                self.latest_rgb_uint8,
                (disp_w, disp_h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            display_img = self.latest_rgb_uint8.copy()
            disp_h, disp_w = H_img, W_img

        self.latest_display_shape = (disp_h, disp_w)

        # 화면에 상태 표시
        text1 = f'Image: {W_img}x{H_img} | Feature: {W_feat}x{H_feat}x{C}'
        text2 = 'Left click object/terrain to print VLAD feature'

        cv2.putText(
            display_img,
            text1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            display_img,
            text2,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            self.window_name,
            cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        )

        # mouse callback 이벤트 처리
        cv2.waitKey(20)

    def handle_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f'\n[CLICK] display pixel x={x}, y={y}', flush=True)

        if self.latest_dino_img_np is None or self.latest_rgb_uint8 is None:
            print('No DINO/VLAD feature map received yet.', flush=True)
            return

        H_img, W_img = self.latest_rgb_uint8.shape[:2]
        H_feat, W_feat, C = self.latest_dino_img_np.shape

        if self.latest_display_shape is None:
            print('No display image shape available.', flush=True)
            return

        H_disp, W_disp = self.latest_display_shape

        # display 좌표 -> 원본 이미지 좌표
        img_x = int(x / W_disp * W_img)
        img_y = int(y / H_disp * H_img)

        img_x = np.clip(img_x, 0, W_img - 1)
        img_y = np.clip(img_y, 0, H_img - 1)

        # 원본 이미지 좌표 -> feature map 좌표
        feat_x = int(img_x / W_img * W_feat)
        feat_y = int(img_y / H_img * H_feat)

        feat_x = np.clip(feat_x, 0, W_feat - 1)
        feat_y = np.clip(feat_y, 0, H_feat - 1)

        r = self.click_radius_feat

        x0 = max(feat_x - r, 0)
        x1 = min(feat_x + r + 1, W_feat)
        y0 = max(feat_y - r, 0)
        y1 = min(feat_y + r + 1, H_feat)

        patch = self.latest_dino_img_np[y0:y1, x0:x1, :]
        patch_flat = patch.reshape(-1, C)

        valid_mask = np.all(np.isfinite(patch_flat), axis=1)
        valid_patch = patch_flat[valid_mask]

        if valid_patch.shape[0] == 0:
            print('Clicked patch has no valid feature.', flush=True)
            return

        feature = np.mean(valid_patch, axis=0)

        print('\n========== Clicked Image DINO/VLAD Feature ==========', flush=True)
        print(f'display pixel : x={x}, y={y}', flush=True)
        print(f'image pixel   : x={img_x}, y={img_y}', flush=True)
        print(f'feature pixel : x={feat_x}, y={feat_y}', flush=True)
        print(f'image size    : {W_img} x {H_img}', flush=True)
        print(f'feature size  : {W_feat} x {H_feat}', flush=True)
        print(f'feature dim   : {C}', flush=True)
        print(f'patch radius  : {r} feature cells', flush=True)
        print(f'valid cells   : {valid_patch.shape[0]}', flush=True)

        print('\nFeature values:', flush=True)
        print(feature.tolist(), flush=True)

        if C == 8:
            vlad8 = feature
            print('\nVLAD 8 values:', flush=True)
            print(vlad8.tolist(), flush=True)

            print(
                f'\nNormalized VLAD 8 values (/ {self.residual_max:.6f}):',
                flush=True,
            )
            print((vlad8 / self.residual_max).tolist(), flush=True)

            print('\nCopy-paste for one-shot code:', flush=True)
            print('torch.Tensor([', flush=True)
            for v in vlad8:
                print(f'    {v:.6f},', flush=True)
            print(']).cuda() / self.residual_max', flush=True)

        elif C > 8:
            vlad8 = feature[:8]
            print('\nWARNING: feature dim is larger than 8.', flush=True)
            print('First 8 values are printed below.', flush=True)

            print('\nFirst 8 values:', flush=True)
            print(vlad8.tolist(), flush=True)

            print('\nCopy-paste for one-shot code:', flush=True)
            print('torch.Tensor([', flush=True)
            for v in vlad8:
                print(f'    {v:.6f},', flush=True)
            print(']).cuda() / self.residual_max', flush=True)

        else:
            print('\nWARNING: feature dimension is smaller than 8.', flush=True)

        print('=====================================================\n', flush=True)


def main(args=None):
    rclpy.init(args=args)

    param_node = rclpy.create_node('_dino_vlad_click_config_reader')
    param_node.declare_parameter('config_fp', '')
    config_fp = param_node.get_parameter('config_fp').value
    param_node.destroy_node()

    if config_fp == '':
        raise RuntimeError('config_fp parameter is empty.')

    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)

    node = DinoVladImageClickNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
