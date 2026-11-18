#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
from cv_bridge import CvBridge
import os
import yaml
import math
import time

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, Float32, Float32MultiArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap

import cv2
import matplotlib.pyplot as plt
from scipy import signal, stats
import pickle

from context_adaptation.context_adaptation_common.utils import pose_msg_to_se3

import torch
import torch.nn.functional as F
from context_adaptation.context_adaptation_common.gridmap import GridMapConvert

import gpytorch
gpytorch.settings.debug(False)

import matplotlib
CMAP = matplotlib.cm.get_cmap('magma')
SMAP = matplotlib.cm.get_cmap('jet')

from ament_index_python.packages import get_package_share_directory


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.covar_module.base_kernel.lengthscale = lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ContextClustererNode(Node):
    def __init__(self, cost_topic, odom_topic, gridmap_topic, costmap_topic, vel_pub_topic, config):
        super().__init__('context_publisher')

        self._lock = Lock()

        assets_dir = os.path.join(get_package_share_directory('context_adaptation'), 'assets') + '/'

        self.odom_msg = None
        self.hz_counter = 0
        self.image = None
        self.new_msg = False
        self.cost = 0.0
        self.speed_mismatch = 0.0
        self.terrain_mismatch = 0.0
        self.pose_se3 = None

        self.costmap_pub = self.create_publisher(GridMap, costmap_topic, 2)
        self.speedmap_pub = self.create_publisher(GridMap, '/shortrange_speedmap', 2)
        self.max_vel_pub = self.create_publisher(Float32, vel_pub_topic, 2)
        self.cvar_pub = self.create_publisher(Float32, '/hdif_speedmap_cvar', 2)

        height_diff = -1 * config['ROBOT']['frame_height']
        front_x = config['ROBOT']['front_x']
        back_x = config['ROBOT']['back_x']
        width = config['ROBOT']['width']

        def make_tf(x, y, z):
            t = np.identity(4)
            t[0:3, -1] = np.array([x, y, z])
            return t

        self.tire_transforms = [
            make_tf(front_x, width, height_diff),
            make_tf(front_x, -width, height_diff),
            make_tf(back_x, width, height_diff),
            make_tf(back_x, -width, height_diff),
        ]

        self.channels = []
        self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])

        self.VLAD_CLUSTS = config['VLAD']['n_clusters']
        self.gridmap_size = None
        self.residual_max = config['BEHAVIOR']['residual_max']
        self.uc_thresh = config['BEHAVIOR']['uncertainty_threshold']
        self.cost_lengthscale = config['BEHAVIOR']['cost_lengthscale']
        self.speed_lengthscale = config['BEHAVIOR']['speed_lengthscale']
        self.train_kernel = config['BEHAVIOR']['train_kernel']

        buffer_len = config['BEHAVIOR']['buffer_size']
        self.buffer_update_freq = config['BEHAVIOR']['buffer_update_freq']
        self.train_in_buffer = torch.zeros((buffer_len, self.VLAD_CLUSTS + 2)).cuda()
        self.train_label_buffer = torch.zeros((buffer_len, 1)).cuda()
        self.train_buffer_classes = np.zeros((buffer_len))
        self.toi = np.zeros((buffer_len))
        self.buffer_idx = 0
        self.buffer_full = False
        self.in_std = 0.0
        self.in_mean = 0.0

        self.cost_offset = .0
        self.velocity = 0.0
        self.fake_velocity = 0.0

        avoid_class = 3
        num_insert = 8
        spread = 1.2
        avoid_data = torch.ones(num_insert, self.VLAD_CLUSTS + 2).cuda() * .9
        avoid_classes = np.zeros((num_insert)) + avoid_class
        avoid_labels = torch.ones((num_insert, 1)).cuda()
        avoid_data[:, -1] = 1.0
        for i in range(num_insert):
            avoid_data[i, -2] = i * spread
            avoid_data[i, :self.VLAD_CLUSTS] = torch.Tensor([
                19.777752, 23.18438, 21.908875, 18.12688,
                25.28553, 23.31651, 21.984331, 24.273615,
            ]).cuda() / self.residual_max

        self.train_in_buffer = torch.vstack((avoid_data, self.train_in_buffer))
        self.train_label_buffer = torch.vstack((avoid_labels, self.train_label_buffer))
        self.train_buffer_classes = np.concatenate((avoid_classes, self.train_buffer_classes))
        self.num_insert = num_insert
        self.buffer_idx += self.num_insert

        self.gp = None
        self.likelihood = None
        self.gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'gp_params_speed_input'))

        self.speed_gp = None
        self.speed_likelihood = None
        self.speed_gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'speed_gp_params_new'))

        self.speed_gp_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        self.max_roughness = config['BEHAVIOR']['max_cost']
        self.max_velocity = config['BEHAVIOR']['max_velocity']
        self.velocity_margin = config['BEHAVIOR']['velocity_margin']
        self.cvar_alpha = .0
        self.rough_history = 0.2
        self.vel_history = 0.0
        self.costmap_cvar = config['BEHAVIOR']['costmap_cvar']
        self.unknown_cost = config['BEHAVIOR']['unknown_cost']
        self.unknown_speed = config['BEHAVIOR']['unknown_speed']
        self.maxpool_speedmap = config['BEHAVIOR']['maxpool_speedmap']

        self.create_subscription(GridMap, gridmap_topic, self.handle_map, 5)
        self.create_subscription(Float32, cost_topic, self.handle_cost, 1)
        self.create_subscription(Odometry, odom_topic, self.handle_odom, 1)
        self.create_subscription(Float32, '/hdif_max_roughness', self.handle_max_roughness, 1)
        self.create_subscription(Float32, '/hdif_cvar', self.handle_cvar, 1)

        self.timer = self.create_timer(1.0 / 10.0, self.run_map)
        self.get_logger().info('DONE WITH INIT')

    def handle_odom(self, msg):
        vel = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self.velocity = vel
        self.odom_msg = msg
        self.pose_se3 = pose_msg_to_se3(msg)
        self.vel_history = self.vel_history + (self.velocity - self.vel_history) * .1

    def handle_cost(self, msg):
        self.cost = msg.data - self.cost_offset
        self.rough_history = self.rough_history + (msg.data - self.rough_history) * .2

    def handle_max_roughness(self, msg):
        self.max_roughness = msg.data
        self.fake_velocity = msg.data

    def handle_cvar(self, msg):
        self.cvar_alpha = msg.data

    def handle_map(self, msg):
        with self._lock:
            self.dino_map = msg
            self.new_msg = True
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'VLAD' in layer:
                        self.channels.append(layer)
                self.grid_map_cvt.channels = self.channels

    def update_train_buffer(self, input, label, class_id):
        self.train_in_buffer[self.buffer_idx] = input
        self.train_label_buffer[self.buffer_idx] = label
        self.train_buffer_classes[self.buffer_idx] = class_id
        self.buffer_idx = min(self.buffer_idx + 1, self.train_in_buffer.shape[0])
        if self.buffer_idx == self.train_in_buffer.shape[0]:
            self.buffer_full = True

    def insert_train_buffer(self, input, label, class_id, idx):
        self.train_in_buffer[idx] = input
        self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))
        for i, transform in enumerate(self.tire_transforms):
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]
        return tire_points

    def run_map(self):
        if self.pose_se3 is None:
          print('waiting for odom...')
          return

        now = time.perf_counter()
        print('----')
        if self.hz_counter == 5000000:
            self.hz_counter = 0
        self.hz_counter += 1

        if not self.new_msg:
            print('no new map')
            return

        with self._lock:
            if self.dino_map is not None:
                info = self.dino_map.info
                nx = int(info.length_x / info.resolution)
                ny = int(info.length_y / info.resolution)
                self.grid_map_cvt.size = [nx, ny]
                self.gridmap_size = nx
                gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map)
                msg_header = self.dino_map.header
                self.new_msg = False
            else:
                gridmap = None

        if gridmap is None:
            print('NO MAP')
            return

        da = gridmap['data'].argmin(axis=0)
        unc_map = gridmap['data'].min(axis=0)
        unc_map /= self.residual_max
        unc_map[unc_map < self.uc_thresh] = 0

        e_kernel = np.ones((2, 2), np.float32)
        unc_map = cv2.erode(unc_map, e_kernel, iterations=1)
        unc_map = cv2.dilate(unc_map, e_kernel, iterations=1)

        P = gridmap['metadata']['origin']
        res = gridmap['metadata']['resolution']
        tp = self.estimate_tire_points()
        tx = np.floor((tp[:, 0] - P[0]) / res)
        ty = np.floor((tp[:, 1] - P[1]) / res)
        xloc, yloc = ty.astype(int), tx.astype(int)

        if np.any(xloc > self.gridmap_size) or np.any(yloc > self.gridmap_size):
            print('WARNING: OUT OF BOUNDS! Not computing costmap')
            return

        classes = da[xloc, yloc]
        spot = stats.mode(classes)[0]

        input_t = torch.from_numpy(gridmap['data'] / self.residual_max)
        zero_mask = (input_t == 0).all(dim=0).cpu().numpy()
        known_mask = ~zero_mask

        if (self.velocity > .3) and self.hz_counter % self.buffer_update_freq == 0:
            input_sample = input_t[:, xloc, yloc].mean(dim=1)
            input_sample = torch.cat((input_sample,
                                      torch.Tensor([self.velocity]),
                                      torch.Tensor([self.cost]))).cuda()
            if torch.count_nonzero(input_sample[:-2]) == 0:
                print('IN UNKNOWN SPACE')
            else:
                if self.buffer_full:
                    most_class = stats.mode(self.train_buffer_classes)[0]
                    vel_hist = torch.histc(
                        self.train_in_buffer[self.train_buffer_classes == most_class, -2],
                        bins=10, min=0, max=10)
                    most_vel = torch.argmax(vel_hist)
                    edges = np.arange(0, 11)
                    sel_min = edges[most_vel]
                    sel_max = edges[most_vel + 1]
                    train_vels = self.train_in_buffer[:, -2].cpu().numpy()
                    insert_idx = np.random.choice(np.where(
                        (self.train_buffer_classes == most_class) &
                        (train_vels > sel_min) & (train_vels < sel_max))[0])
                    if insert_idx > self.num_insert:
                        self.insert_train_buffer(input_sample, self.cost, spot, insert_idx)
                else:
                    self.update_train_buffer(input_sample, self.cost, spot)

        costmap = np.zeros(self.gridmap_size * self.gridmap_size)
        speedmap = np.zeros(self.gridmap_size * self.gridmap_size) + 4.5

        if self.hz_counter % 10 == 0 or self.gp is None:
            print('****************************************************')
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            self.speed_likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            if not self.buffer_full:
                train_in = self.train_in_buffer[:self.buffer_idx]
                train_label = self.train_label_buffer[:self.buffer_idx, 0]
            else:
                train_in = self.train_in_buffer
                train_label = self.train_label_buffer[:, 0]

            self.in_mean = torch.mean(train_in, dim=0)
            self.in_std = torch.std(train_in, dim=0).clamp(min=1e-6)
            train_in_norm = (train_in - self.in_mean) / self.in_std
            self.gp = ExactGPModel(train_in_norm[:, :-1], train_in_norm[:, -1],
                                   self.likelihood, self.cost_lengthscale).cuda()
            self.speed_gp = ExactGPModel(train_in_norm[:, self.speed_gp_indices], train_in_norm[:, -2],
                                         self.speed_likelihood, self.speed_lengthscale).cuda()
            if self.gp_params is not None:
                self.gp.load_state_dict(self.gp_params)
            if self.speed_gp_params is not None:
                self.speed_gp.load_state_dict(self.speed_gp_params)

        elif self.train_kernel:
            if not self.buffer_full:
                train_in = self.train_in_buffer[:self.buffer_idx]
            else:
                train_in = self.train_in_buffer
            if train_in.shape[0] < 13:
                return
            optimizer = torch.optim.SGD(self.speed_gp.parameters(), lr=0.01)
            with gpytorch.settings.debug(False):
                self.speed_gp.train()
                self.speed_likelihood.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.speed_likelihood, self.speed_gp)
                optimizer.zero_grad()
                output = self.speed_gp(train_in[:, self.speed_gp_indices])
                loss = -mll(output, train_in[:, -2])
                print('LOSS - ', loss.item())
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                self.speed_gp_params = self.speed_gp.state_dict()

        with torch.no_grad():
            if self.buffer_idx < 5 + self.num_insert:
                costmap = np.zeros((self.gridmap_size, self.gridmap_size))
                speedmap = np.zeros((self.gridmap_size, self.gridmap_size)) + 4.5
                costmap_var = np.zeros_like(costmap)
                speedmap_var = np.zeros_like(speedmap)
            else:
                self.gp.eval()
                self.speed_gp.eval()
                self.likelihood.eval()
                self.speed_likelihood.eval()
                input_t = input_t.permute(1, 2, 0).view(-1, self.VLAD_CLUSTS)
                zero_mask_f = zero_mask.flatten()
                keep_mask = ~zero_mask_f
                input_t = input_t[keep_mask]

                vel_append = torch.ones(input_t.shape[0], 1) * self.velocity
                rough_append = torch.ones(input_t.shape[0], 1) * self.max_roughness
                input_t = torch.hstack((input_t, vel_append, rough_append)).cuda()
                input_t = (input_t - self.in_mean) / self.in_std

                observed_pred = self.likelihood(self.gp(input_t[:, :-1]))
                cost_pred = observed_pred.mean
                cost_var = observed_pred.variance

                observed_pred = self.speed_likelihood(self.speed_gp(input_t[:, self.speed_gp_indices]))
                speed_pred = observed_pred.mean
                speed_var = observed_pred.variance

                phi = stats.norm.pdf(stats.norm.ppf(self.cvar_alpha))
                cvar = speed_pred + (speed_var * phi) / (1.0 - self.cvar_alpha)
                speed_pred = (cvar * self.in_std[-2]) + self.in_mean[-2]

                cmap_cvar = self.costmap_cvar
                phi = stats.norm.pdf(stats.norm.ppf(cmap_cvar))
                cvar = cost_pred + (cost_var * phi) / (1.0 - cmap_cvar)
                cost_pred = (cvar * self.in_std[-1]) + self.in_mean[-1]

                costmap[keep_mask] = cost_pred.cpu().numpy()
                speedmap[keep_mask] = speed_pred.cpu().numpy()
                costmap = costmap.reshape(self.gridmap_size, self.gridmap_size)
                speedmap = speedmap.reshape(self.gridmap_size, self.gridmap_size)

                if self.maxpool_speedmap:
                    speedmap_t = torch.from_numpy(speedmap).cuda()
                    ksize = 3
                    padding = (ksize - 1) // 2
                    m = torch.nn.MaxPool2d(ksize, stride=1, padding=padding)
                    speedmap = m(speedmap_t.unsqueeze(0))[0].cpu().numpy()

                if ((self.rough_history < self.max_roughness) and
                        np.abs(self.vel_history - speedmap[xloc[0], yloc[0]]) < self.velocity_margin):
                    self.cvar_alpha += 0.005
                elif self.rough_history > self.max_roughness:
                    self.cvar_alpha -= 0.02
                self.cvar_alpha = np.clip(self.cvar_alpha, 0, .99)

                if (np.abs(self.vel_history - speedmap[xloc[0], yloc[0]]) > .4 and
                        speedmap[xloc[0], yloc[0]] < 2.8):
                    self.cvar_alpha += .01

            if self.buffer_idx < 30 + self.num_insert:
                speedmap = np.zeros((self.gridmap_size, self.gridmap_size)) + 4.5
                self.cvar_alpha = 0.0

        costmap /= 2.0
        ids = np.where(unc_map != 0)
        costmap[ids] = unc_map[ids]
        costmap[~np.isfinite(costmap)] = 0.0
        costmap[zero_mask] = self.unknown_cost
        speedmap[zero_mask] = self.unknown_speed

        print('Speedmap Min/Max', speedmap.min(), speedmap.max(), speedmap[xloc[0], yloc[0]])
        speedmap = np.clip(speedmap, 0, self.max_velocity)

        costmap_msg = self.costmap_to_gridmap(costmap, info, header=msg_header)
        self.costmap_pub.publish(costmap_msg)

        speedmap_msg = self.costmap_to_gridmap(speedmap, info, header=msg_header, costmap_layer='speedmap',
                                               norm_factor=self.max_velocity)
        self.speedmap_pub.publish(speedmap_msg)

        cvar_msg = Float32()
        cvar_msg.data = self.cvar_alpha
        self.cvar_pub.publish(cvar_msg)

        print(time.perf_counter() - now, 'total time', self.buffer_idx, 'buffer_idx')

    def costmap_to_gridmap(self, costmap, info, header=None, costmap_layer='costmap', variance=None, norm_factor=None):
        costmap_msg = GridMap()
        costmap_msg.info = info
        if header is not None:
            costmap_msg.header = header
        costmap_msg.layers = [costmap_layer]

        def make_layer(data):
            layer_msg = Float32MultiArray()
            layer_msg.layout.dim.append(
                MultiArrayDimension(label='column_index', size=data.shape[0], stride=data.shape[0]))
            layer_msg.layout.dim.append(
                MultiArrayDimension(label='row_index', size=data.shape[0],
                                    stride=data.shape[0] * data.shape[1]))
            layer_msg.data = data[::-1, ::-1].flatten().tolist()
            return layer_msg

        costmap_msg.data.append(make_layer(costmap))

        costmap_msg.layers.append('elevation')
        if variance is not None:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 - variance * 5
        else:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73
        costmap_msg.data.append(make_layer(layer_data))

        if norm_factor is None:
            vcostmap = np.clip(costmap * 2., 0, 1)
        else:
            vcostmap = np.clip(costmap, 0, norm_factor) / norm_factor

        if costmap_layer == 'costmap':
            gridmap_cs = (CMAP(vcostmap) * 255).astype(np.int32)
        else:
            gridmap_cs = (SMAP(vcostmap) * 255).astype(np.int32)

        gridmap_color = (gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2])
        gridmap_color = gridmap_color.view(dtype=np.float32)

        costmap_msg.layers.append('rgb_viz')
        rgb_layer_msg = Float32MultiArray()
        rgb_layer_msg.layout.dim.append(
            MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
        rgb_layer_msg.layout.dim.append(
            MultiArrayDimension(label='row_index', size=layer_data.shape[0],
                                stride=layer_data.shape[0] * layer_data.shape[1]))
        rgb_layer_msg.data = gridmap_color[::-1, ::-1].flatten().tolist()
        costmap_msg.data.append(rgb_layer_msg)

        return costmap_msg


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('_param_reader')
    node.declare_parameter('cost_topic', '')
    node.declare_parameter('odom_topic', '')
    node.declare_parameter('gridmap_topic', '')
    node.declare_parameter('costmap_topic', '')
    node.declare_parameter('vel_pub_topic', '')
    node.declare_parameter('config_file', '')

    cost_topic = node.get_parameter('cost_topic').value
    odom_topic = node.get_parameter('odom_topic').value
    gridmap_topic = node.get_parameter('gridmap_topic').value
    costmap_topic = node.get_parameter('costmap_topic').value
    vel_pub_topic = node.get_parameter('vel_pub_topic').value
    config_file = node.get_parameter('config_file').value
    node.destroy_node()

    config_dict = yaml.safe_load(open(config_file, 'r'))

    context_node = ContextClustererNode(
        cost_topic, odom_topic, gridmap_topic, costmap_topic, vel_pub_topic, config_dict)

    try:
        rclpy.spin(context_node)
    except KeyboardInterrupt:
        pass
    finally:
        context_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
