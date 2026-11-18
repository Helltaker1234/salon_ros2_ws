#!/usr/bin/env python3
import yaml
import copy
import rclpy
from rclpy.node import Node
import numpy as np

import torch
import torch.nn.functional as F
import cv_bridge
import os
import time
from scipy.spatial.transform import Rotation as ScipyRotation
from torch_scatter import scatter as ts_scatter

import distinctipy as COLORS

from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32, Header
from sensor_msgs.msg import PointField

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import (
    get_intrinsics, get_extrinsics, obtain_projection_matrix,
    get_pixel_from_3D_source, get_points_and_pixels_in_frame,
    xyz_array_to_point_cloud_msg,
)
from physics_atv_visual_mapping.localmapping.localmapping import localmap_from_pointcloud, aggregate_localmaps
from physics_atv_visual_mapping.utils import pcl_msg_to_xyz, pose_to_htm, transform_points


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

        self.image_pipeline = setup_image_pipeline(config)


        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config['image']['image_compressed']
        if self.compressed_img:
            self.image_sub = self.create_subscription(
                CompressedImage, config['image']['image_topic'], self.handle_img, 1)
        else:
            self.image_sub = self.create_subscription(
                Image, config['image']['image_topic'], self.handle_img, 1)

        if 'rect' in config['image']['image_topic']:
            self.intrinsics = torch.tensor(config['intrinsics']['P'], device=config['device']).reshape(3, 3).float()
        else:
            self.intrinsics = torch.tensor(config['intrinsics']['K'], device=config['device']).reshape(3, 3).float()

        self.extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q']),
        ], axis=-1))

        self.pcl_sub = self.create_subscription(
            PointCloud2, config['pointcloud']['topic'], self.handle_pointcloud, 1)
        self.odom_sub = self.create_subscription(
            Odometry, config['odometry']['topic'], self.handle_odom, 10)

        self.pcl_frame_remap = config['pointcloud'].get('frame', None)

        self.pcl_pub = self.create_publisher(PointCloud2, '/dino_pcl', 1)
        self.gridmap_pub = self.create_publisher(GridMap, '/dino_gridmap', 1)
        self.image_pub = self.create_publisher(Image, '/dino_image', 1)

        self.viz = config.get('viz', False)
        self.timer = self.create_timer(0.1, self.spin_once)

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk['n']):
                out.append('{}_{}'.format(lk['key'], i))
        return out

    def handle_pointcloud(self, msg):
        self.pcl_msg = msg
        if self.pcl_frame_remap is not None:
            self.pcl_msg.header.frame_id = self.pcl_frame_remap

    def handle_odom(self, msg):
        if self.odom_frame is None:
            self.odom_frame = msg.header.frame_id
        self.odom_msg = msg

    def handle_img(self, msg):
        self.img_msg = msg

    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

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

        R_np = ScipyRotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)

        T = torch.eye(4, dtype=torch.float32, device=self.device)
        T[:3, :3] = torch.as_tensor(R_np, dtype=torch.float32, device=self.device)
        T[:3, 3] = torch.tensor([p.x, p.y, p.z], dtype=torch.float32, device=self.device)
        return T

    def preprocess_inputs(self):
        if self.pcl_msg is None:
            self.get_logger().warning('no pcl msg received', throttle_duration_sec=0.01)
            return None

        pcl_time = self._stamp_to_sec(self.pcl_msg.header.stamp)
        if abs(pcl_time - self.last_update_time) < 1e-3:
            self.get_logger().warning('pcl time problem', throttle_duration_sec=0.01)
            return None

        if self.odom_msg is None:
            self.get_logger().warning('no odom msg received', throttle_duration_sec=0.01)
            return None

        if self.img_msg is None:
            self.get_logger().warning('no img msg received', throttle_duration_sec=0.01)
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
        pcl_odom = transform_points(pcl.clone(), pcl_htm)

        _metadata = {
            'origin': torch.tensor([
                self.odom_msg.pose.pose.position.x - 0.5 * self.base_metadata['length_x'],
                self.odom_msg.pose.pose.position.y - 0.5 * self.base_metadata['length_y'],
            ]).float().to(self.device),
            'length_x': torch.tensor(self.base_metadata['length_x']).float().to(self.device),
            'length_y': torch.tensor(self.base_metadata['length_y']).float().to(self.device),
            'resolution': torch.tensor(self.base_metadata['resolution']).float().to(self.device),
        }

        self.last_update_time = pcl_time

        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8') / 255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8').astype(np.float32) / 255.

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)

        dino_img, dino_intrinsics = self.image_pipeline.run(img, self.intrinsics.unsqueeze(0))
        dino_img = dino_img[0].permute(1, 2, 0)
        dino_intrinsics = dino_intrinsics[0]

        I = get_intrinsics(dino_intrinsics).to(self.device)
        E = get_extrinsics(self.extrinsics).to(self.device)
        P = obtain_projection_matrix(I, E)

        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        lidar_points_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3], pixel_coordinates, dino_img.shape[0], dino_img.shape[1])

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl_odom[ind_in_frame][:, :3], dino_features], dim=-1)

        if self.compute_slope:
            # Append z (odom frame) as an extra feature channel to be EMA-accumulated.
            # Slope is derived later from the stable accumulated elevation map.
            z_in_frame = pcl_odom[ind_in_frame][:, [2]]
            dino_pcl = torch.cat([dino_pcl, z_in_frame], dim=-1)

        return {
            'pcl': pcl_odom,
            'metadata': _metadata,
            'image': img,
            'dino_image': dino_img,
            'dino_pcl': dino_pcl,
            'pixel_projection': pixel_coordinates[ind_in_frame],
        }

    def update_localmap(self, pcl, metadata):
        pcl_pos = pcl[:, :3]
        pcl_data = pcl[:, 3:]
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
        R_mat = ScipyRotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
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

    def make_gridmap_msg(self, localmap):
        gridmap_msg = GridMap()
        gridmap_data = localmap['data'].cpu().numpy()

        gridmap_msg.header.stamp = self.img_msg.header.stamp
        gridmap_msg.header.frame_id = self.odom_frame

        if self.layer_keys is None:
            if self.compute_slope:
                # channels: VLAD×8, z_elevation, slope_longitudinal, slope_lateral
                n_vlad = gridmap_data.shape[-1] - 3
                gridmap_msg.layers = ['{}_{}'.format(self.layer_key, i) for i in range(n_vlad)]
                gridmap_msg.layers += ['z_elevation', 'slope_longitudinal', 'slope_lateral']
            else:
                gridmap_msg.layers = ['{}_{}'.format(self.layer_key, i)
                                      for i in range(gridmap_data.shape[-1])]
        else:
            gridmap_msg.layers = copy.deepcopy(self.layer_keys)

        gridmap_msg.info.resolution = localmap['metadata']['resolution'].item()
        gridmap_msg.info.length_x = localmap['metadata']['length_x'].item()
        gridmap_msg.info.length_y = localmap['metadata']['length_y'].item()
        gridmap_msg.info.pose.position.x = localmap['metadata']['origin'][0].item() + 0.5 * gridmap_msg.info.length_x
        gridmap_msg.info.pose.position.y = localmap['metadata']['origin'][1].item() + 0.5 * gridmap_msg.info.length_y
        gridmap_msg.info.pose.position.z = self.odom_msg.pose.pose.position.z
        gridmap_msg.info.pose.orientation.w = 1.

        for i in range(gridmap_data.shape[-1]):
            layer_data = gridmap_data[..., i]
            gridmap_layer_msg = Float32MultiArray()
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(label='row_index', size=layer_data.shape[0],
                                    stride=layer_data.shape[0] * layer_data.shape[1]))
            gridmap_layer_msg.data = layer_data[::-1, ::-1].T.flatten().tolist()
            gridmap_msg.data.append(gridmap_layer_msg)

        # dummy elevation layer
        gridmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(gridmap_data[..., 0]) + self.odom_msg.pose.pose.position.z - 1.73
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(label='row_index', size=layer_data.shape[0],
                                stride=layer_data.shape[0] * layer_data.shape[1]))
        gridmap_layer_msg.data = layer_data.flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        # rgb viz layer
        gridmap_rgb = gridmap_data[..., :3]
        vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1, 1, 3)
        vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)
        gridmap_cs = ((gridmap_rgb - vmin) / (vmax - vmin) * 255.).astype(np.int32)
        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        gridmap_msg.layers.append('rgb_viz')
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(label='row_index', size=layer_data.shape[0],
                                stride=layer_data.shape[0] * layer_data.shape[1]))
        gridmap_layer_msg.data = gridmap_color.T[::-1, ::-1].flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    # def make_pcl_msg(self, pcl):
    #     pcl_pos = pcl[:, :3].cpu().numpy()
    #     pcl_cs = pcl[:, [5, 4, 3]]
    #     vmin = pcl_cs.min(dim=0)[0].view(1, 3)
    #     vmax = pcl_cs.max(dim=0)[0].view(1, 3)
    #     pcl_cs = ((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()

    #     return xyz_array_to_point_cloud_msg(
    #         points=pcl_pos,
    #         frame=self.odom_frame,
    #         timestamp=self.pcl_msg.header.stamp,
    #         rgb_values=(pcl_cs * 255.).astype(np.uint8),
    #     )

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
    

    def publish_messages(self, res, publish_map=None):
        if publish_map is None:
            publish_map = self.localmap
        gridmap_msg = self.make_gridmap_msg(publish_map)
        self.gridmap_pub.publish(gridmap_msg)

        pcl_msg = self.make_pcl_msg(res['dino_pcl'])
        self.pcl_pub.publish(pcl_msg)

        vmin = self.localmap['data'][..., :3].view(-1, 3).min(dim=0)[0].view(1, 1, 3)
        vmax = self.localmap['data'][..., :3].view(-1, 3).max(dim=0)[0].view(1, 1, 3)
        viz_img = ((res['dino_image'][..., :3] - vmin) / (vmax - vmin)).clip(0., 1.).cpu().numpy() * 255

        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), 'rgb8')
        img_msg.header.stamp = pcl_msg.header.stamp
        self.image_pub.publish(img_msg)

    def spin_once(self):
        self.get_logger().info('spinning...')

        t1 = time.time()
        res = self.preprocess_inputs()
        t2 = time.time()

        if res:
            self.get_logger().info('updating localmap...')
            t3 = time.time()
            self.localmap = self.update_localmap(pcl=res['dino_pcl'], metadata=res['metadata'])
            t4 = time.time()
            publish_map = self.localmap
            if self.compute_slope and self.localmap is not None:
                publish_map = self.compute_slope_from_elevation(self.localmap)
            self.publish_messages(res, publish_map)
            t5 = time.time()
            self.get_logger().info(
                'Timing:\n\tpreproc: {:.6f}s\n\tlocalmap: {:.6f}s\n\tserialize: {:.6f}s'.format(
                    t2 - t1, t4 - t3, t5 - t4))

        else:
            self.get_logger().info('skipped because no result')

def main(args=None):
    rclpy.init(args=args)

    node_handle = rclpy.create_node('_config_reader')
    node_handle.declare_parameter('config_fp', '')
    config_fp = node_handle.get_parameter('config_fp').value
    node_handle.destroy_node()

    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)

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
