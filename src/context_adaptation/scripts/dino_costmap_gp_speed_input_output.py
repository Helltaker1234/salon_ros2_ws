#!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# import numpy as np
# from threading import Lock
# from cv_bridge import CvBridge
# import os
# import yaml
# import math
# import time

# from sensor_msgs.msg import CompressedImage, Image
# from std_msgs.msg import Int32, Float32, Float32MultiArray
# from std_msgs.msg import Float32MultiArray, MultiArrayDimension
# from grid_map_msgs.msg import GridMap

# import cv2
# import matplotlib.pyplot as plt
# from scipy import signal, stats
# import pickle

# from context_adaptation.context_adaptation_common.utils import pose_msg_to_se3

# import torch
# import torch.nn.functional as F
# from context_adaptation.context_adaptation_common.gridmap import GridMapConvert

# import gpytorch
# gpytorch.settings.debug(False)

# import matplotlib
# CMAP = matplotlib.cm.get_cmap('magma')
# SMAP = matplotlib.cm.get_cmap('jet')

# from ament_index_python.packages import get_package_share_directory


# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, lengthscale):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
#         self.covar_module.base_kernel.lengthscale = lengthscale

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class ContextClustererNode(Node):
#     def __init__(self, cost_topic, odom_topic, gridmap_topic, costmap_topic, vel_pub_topic, config):
#         super().__init__('context_publisher')

#         self._lock = Lock()

#         assets_dir = os.path.join(get_package_share_directory('context_adaptation'), 'assets') + '/'

#         self.odom_msg = None
#         self.hz_counter = 0
#         self.image = None
#         self.new_msg = False
#         self.cost = 0.0
#         self.speed_mismatch = 0.0
#         self.terrain_mismatch = 0.0
#         self.pose_se3 = None

#         self.costmap_pub = self.create_publisher(GridMap, costmap_topic, 2)
#         self.speedmap_pub = self.create_publisher(GridMap, '/shortrange_speedmap', 2)
#         self.max_vel_pub = self.create_publisher(Float32, vel_pub_topic, 2)
#         self.cvar_pub = self.create_publisher(Float32, '/hdif_speedmap_cvar', 2)

#         height_diff = -1 * config['ROBOT']['frame_height']
#         front_x = config['ROBOT']['front_x']
#         back_x = config['ROBOT']['back_x']
#         width = config['ROBOT']['width']

#         def make_tf(x, y, z):
#             t = np.identity(4)
#             t[0:3, -1] = np.array([x, y, z])
#             return t

#         self.tire_transforms = [
#             make_tf(front_x, width, height_diff),
#             make_tf(front_x, -width, height_diff),
#             make_tf(back_x, width, height_diff),
#             make_tf(back_x, -width, height_diff),
#         ]

#         self.channels = []
#         self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])

#         self.VLAD_CLUSTS = config['VLAD']['n_clusters']
#         self.gridmap_size = None
#         self.residual_max = config['BEHAVIOR']['residual_max']
#         self.uc_thresh = config['BEHAVIOR']['uncertainty_threshold']
#         self.cost_lengthscale = config['BEHAVIOR']['cost_lengthscale']
#         self.speed_lengthscale = config['BEHAVIOR']['speed_lengthscale']
#         self.train_kernel = config['BEHAVIOR']['train_kernel']

#         buffer_len = config['BEHAVIOR']['buffer_size']
#         self.buffer_update_freq = config['BEHAVIOR']['buffer_update_freq']
#         self.train_in_buffer = torch.zeros((buffer_len, self.VLAD_CLUSTS + 2)).cuda()
#         self.train_label_buffer = torch.zeros((buffer_len, 1)).cuda()
#         self.train_buffer_classes = np.zeros((buffer_len))
#         self.toi = np.zeros((buffer_len))
#         self.buffer_idx = 0
#         self.buffer_full = False
#         self.in_std = 0.0
#         self.in_mean = 0.0

#         self.cost_offset = .0
#         self.velocity = 0.0
#         self.fake_velocity = 0.0

#         self.max_roughness = config['BEHAVIOR']['max_cost']
#         self.max_velocity = config['BEHAVIOR']['max_velocity']
#         self.velocity_margin = config['BEHAVIOR']['velocity_margin']
#         self.cvar_alpha = .0
#         self.rough_history = 0.2
#         self.vel_history = 0.0
#         self.costmap_cvar = config['BEHAVIOR']['costmap_cvar']
#         self.unknown_cost = config['BEHAVIOR']['unknown_cost']
#         self.unknown_speed = config['BEHAVIOR']['unknown_speed']
#         self.maxpool_speedmap = config['BEHAVIOR']['maxpool_speedmap']
        
#         self.speed_init = config['BEHAVIOR']['speed_init']
#         self.cvar_speed_threshold = self.max_velocity * (2.8 / 6.0)

#         avoid_class = 3
#         num_insert = 8
#         spread = self.max_velocity / (num_insert - 1)
#         avoid_data = torch.ones(num_insert, self.VLAD_CLUSTS + 2).cuda() * .9
#         avoid_classes = np.zeros((num_insert)) + avoid_class
#         avoid_labels = torch.ones((num_insert, 1)).cuda()
#         avoid_data[:, -1] = 1.0
#         for i in range(num_insert):
#             avoid_data[i, -2] = i * spread
#             avoid_data[i, :self.VLAD_CLUSTS] = torch.Tensor([
#                 # Github
#                 # 19.777752, 23.18438, 21.908875, 18.12688,
#                 # 25.28553, 23.31651, 21.984331, 24.273615,
#                 # Road
#                 # 21.88132095336914, 24.390525817871094, 22.666915893554688, 20.71625518798828,
#                 # 25.843544006347656, 24.023813247680664, 20.96671485900879, 24.57954216003418,
#                 # Tree 1
#                 23.78504180908203, 24.359973907470703, 24.878049850463867, 23.789031982421875, 
#                 25.906394958496094, 25.542871475219727, 23.74051284790039, 25.713401794433594,
#                 # Tree 2
#                 # 23.406383514404297, 25.758296966552734, 25.794809341430664, 24.845937728881836,
#                 # 26.53348159790039, 27.235206604003906, 25.33324432373047, 25.70024871826172,
#             ]).cuda() / self.residual_max

#         self.train_in_buffer = torch.vstack((avoid_data, self.train_in_buffer))
#         self.train_label_buffer = torch.vstack((avoid_labels, self.train_label_buffer))
#         self.train_buffer_classes = np.concatenate((avoid_classes, self.train_buffer_classes))
#         self.num_insert = num_insert
#         self.buffer_idx += self.num_insert

#         self.gp = None
#         self.likelihood = None
#         self.gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'gp_params_speed_input'))

#         self.speed_gp = None
#         self.speed_likelihood = None
#         self.speed_gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'speed_gp_params_new'))

#         self.speed_gp_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]

#         self.create_subscription(GridMap, gridmap_topic, self.handle_map, 5)
#         self.create_subscription(Float32, cost_topic, self.handle_cost, 1)
#         self.create_subscription(Odometry, odom_topic, self.handle_odom, 1)
#         self.create_subscription(Float32, '/hdif_max_roughness', self.handle_max_roughness, 1)
#         self.create_subscription(Float32, '/hdif_cvar', self.handle_cvar, 1)

#         self.timer = self.create_timer(1.0 / 10.0, self.run_map)
#         self.get_logger().info('DONE WITH INIT')

#     def handle_odom(self, msg):
#         vel = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
#         self.velocity = vel
#         self.odom_msg = msg
#         self.pose_se3 = pose_msg_to_se3(msg)
#         self.vel_history = self.vel_history + (self.velocity - self.vel_history) * .1

#     def handle_cost(self, msg):
#         self.cost = msg.data - self.cost_offset
#         self.rough_history = self.rough_history + (msg.data - self.rough_history) * .2

#     def handle_max_roughness(self, msg):
#         self.max_roughness = msg.data
#         self.fake_velocity = msg.data

#     def handle_cvar(self, msg):
#         self.cvar_alpha = msg.data

#     def handle_map(self, msg):
#         with self._lock:
#             self.dino_map = msg
#             self.new_msg = True
#             if len(self.channels) == 0:
#                 for layer in msg.layers:
#                     if 'VLAD' in layer:
#                         self.channels.append(layer)
#                 self.grid_map_cvt.channels = self.channels

#     def update_train_buffer(self, input, label, class_id):
#         self.train_in_buffer[self.buffer_idx] = input
#         self.train_label_buffer[self.buffer_idx] = label
#         self.train_buffer_classes[self.buffer_idx] = class_id
#         self.buffer_idx = min(self.buffer_idx + 1, self.train_in_buffer.shape[0])
#         if self.buffer_idx == self.train_in_buffer.shape[0]:
#             self.buffer_full = True

#     def insert_train_buffer(self, input, label, class_id, idx):
#         self.train_in_buffer[idx] = input
#         self.train_label_buffer[idx] = label
#         self.train_buffer_classes[idx] = class_id

#     def estimate_tire_points(self):
#         tire_points = np.ones((4, 3))
#         for i, transform in enumerate(self.tire_transforms):
#             tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]
#         return tire_points

#     def run_map(self):
#         if self.pose_se3 is None:
#           print('waiting for odom...')
#           return

#         now = time.perf_counter()
#         print('----')
#         if self.hz_counter == 5000000:
#             self.hz_counter = 0
#         self.hz_counter += 1

#         if not self.new_msg:
#             print('no new map')
#             return

#         with self._lock:
#             if self.dino_map is not None:
#                 info = self.dino_map.info
#                 nx = int(info.length_x / info.resolution)
#                 ny = int(info.length_y / info.resolution)
#                 self.grid_map_cvt.size = [nx, ny]
#                 self.gridmap_size = nx
#                 gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map)
#                 msg_header = self.dino_map.header
#                 self.new_msg = False
#             else:
#                 gridmap = None

#         if gridmap is None:
#             print('NO MAP')
#             return

#         da = gridmap['data'].argmin(axis=0)
#         unc_map = gridmap['data'].min(axis=0)
#         unc_map /= self.residual_max
#         unc_map[unc_map < self.uc_thresh] = 0

#         e_kernel = np.ones((2, 2), np.float32)
#         unc_map = cv2.erode(unc_map, e_kernel, iterations=1)
#         unc_map = cv2.dilate(unc_map, e_kernel, iterations=1)

#         P = gridmap['metadata']['origin']
#         res = gridmap['metadata']['resolution']
#         tp = self.estimate_tire_points()
#         tx = np.floor((tp[:, 0] - P[0]) / res)
#         ty = np.floor((tp[:, 1] - P[1]) / res)
#         xloc, yloc = ty.astype(int), tx.astype(int)

#         if np.any(xloc > self.gridmap_size) or np.any(yloc > self.gridmap_size):
#             print('WARNING: OUT OF BOUNDS! Not computing costmap')
#             return

#         classes = da[xloc, yloc]
#         spot = stats.mode(classes)[0]

#         input_t = torch.from_numpy(gridmap['data'] / self.residual_max)
#         zero_mask = (input_t == 0).all(dim=0).cpu().numpy()
#         known_mask = ~zero_mask

#         if (self.velocity > .3) and self.hz_counter % self.buffer_update_freq == 0:
#             input_sample = input_t[:, xloc, yloc].mean(dim=1)
#             input_sample = torch.cat((input_sample,
#                                       torch.Tensor([self.velocity]),
#                                       torch.Tensor([self.cost]))).cuda()
#             if torch.count_nonzero(input_sample[:-2]) == 0:
#                 print('IN UNKNOWN SPACE')
#             else:
#                 if self.buffer_full:
#                     most_class = stats.mode(self.train_buffer_classes)[0]
#                     vel_hist = torch.histc(
#                         self.train_in_buffer[self.train_buffer_classes == most_class, -2],
#                         bins=10, min=0, max=10)
#                     most_vel = torch.argmax(vel_hist)
#                     edges = np.arange(0, 11)
#                     sel_min = edges[most_vel]
#                     sel_max = edges[most_vel + 1]
#                     train_vels = self.train_in_buffer[:, -2].cpu().numpy()
#                     insert_idx = np.random.choice(np.where(
#                         (self.train_buffer_classes == most_class) &
#                         (train_vels > sel_min) & (train_vels < sel_max))[0])
#                     if insert_idx > self.num_insert:
#                         self.insert_train_buffer(input_sample, self.cost, spot, insert_idx)
#                 else:
#                     self.update_train_buffer(input_sample, self.cost, spot)

#         costmap = np.zeros(self.gridmap_size * self.gridmap_size)
#         speedmap = np.zeros(self.gridmap_size * self.gridmap_size) + self.speed_init

#         if self.hz_counter % 10 == 0 or self.gp is None:
#             print('****************************************************')
#             self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
#             self.speed_likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
#             if not self.buffer_full:
#                 train_in = self.train_in_buffer[:self.buffer_idx]
#                 train_label = self.train_label_buffer[:self.buffer_idx, 0]
#             else:
#                 train_in = self.train_in_buffer
#                 train_label = self.train_label_buffer[:, 0]

#             self.in_mean = torch.mean(train_in, dim=0)
#             self.in_std = torch.std(train_in, dim=0).clamp(min=1e-6)
#             train_in_norm = (train_in - self.in_mean) / self.in_std
#             self.gp = ExactGPModel(train_in_norm[:, :-1], train_in_norm[:, -1],
#                                    self.likelihood, self.cost_lengthscale).cuda()
#             self.speed_gp = ExactGPModel(train_in_norm[:, self.speed_gp_indices], train_in_norm[:, -2],
#                                          self.speed_likelihood, self.speed_lengthscale).cuda()
#             if self.gp_params is not None:
#                 self.gp.load_state_dict(self.gp_params)
#             if self.speed_gp_params is not None:
#                 self.speed_gp.load_state_dict(self.speed_gp_params)

#         elif self.train_kernel:
#             if not self.buffer_full:
#                 train_in = self.train_in_buffer[:self.buffer_idx]
#             else:
#                 train_in = self.train_in_buffer
#             if train_in.shape[0] < 13:
#                 return
#             optimizer = torch.optim.SGD(self.speed_gp.parameters(), lr=0.01)
#             with gpytorch.settings.debug(False):
#                 self.speed_gp.train()
#                 self.speed_likelihood.train()
#                 mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.speed_likelihood, self.speed_gp)
#                 optimizer.zero_grad()
#                 output = self.speed_gp(train_in[:, self.speed_gp_indices])
#                 loss = -mll(output, train_in[:, -2])
#                 print('LOSS - ', loss.item())
#                 if not torch.isnan(loss):
#                     loss.backward()
#                     optimizer.step()
#                 self.speed_gp_params = self.speed_gp.state_dict()

#         with torch.no_grad():
#             if self.buffer_idx < 5 + self.num_insert:
#                 costmap = np.zeros((self.gridmap_size, self.gridmap_size))
#                 speedmap = np.zeros((self.gridmap_size, self.gridmap_size)) + self.speed_init
#                 costmap_var = np.zeros_like(costmap)
#                 speedmap_var = np.zeros_like(speedmap)
#             else:
#                 self.gp.eval()
#                 self.speed_gp.eval()
#                 self.likelihood.eval()
#                 self.speed_likelihood.eval()
#                 input_t = input_t.permute(1, 2, 0).view(-1, self.VLAD_CLUSTS)
#                 zero_mask_f = zero_mask.flatten()
#                 keep_mask = ~zero_mask_f
#                 input_t = input_t[keep_mask]

#                 vel_append = torch.ones(input_t.shape[0], 1) * self.velocity
#                 rough_append = torch.ones(input_t.shape[0], 1) * self.max_roughness
#                 input_t = torch.hstack((input_t, vel_append, rough_append)).cuda()
#                 input_t = (input_t - self.in_mean) / self.in_std

#                 observed_pred = self.likelihood(self.gp(input_t[:, :-1]))
#                 cost_pred = observed_pred.mean
#                 cost_var = observed_pred.variance

#                 observed_pred = self.speed_likelihood(self.speed_gp(input_t[:, self.speed_gp_indices]))
#                 speed_pred = observed_pred.mean
#                 speed_var = observed_pred.variance

#                 phi = stats.norm.pdf(stats.norm.ppf(self.cvar_alpha))
#                 cvar = speed_pred + (speed_var * phi) / (1.0 - self.cvar_alpha)
#                 speed_pred = (cvar * self.in_std[-2]) + self.in_mean[-2]

#                 cmap_cvar = self.costmap_cvar
#                 phi = stats.norm.pdf(stats.norm.ppf(cmap_cvar))
#                 cvar = cost_pred + (cost_var * phi) / (1.0 - cmap_cvar)
#                 cost_pred = (cvar * self.in_std[-1]) + self.in_mean[-1]

#                 costmap[keep_mask] = cost_pred.cpu().numpy()
#                 speedmap[keep_mask] = speed_pred.cpu().numpy()
#                 costmap = costmap.reshape(self.gridmap_size, self.gridmap_size)
#                 speedmap = speedmap.reshape(self.gridmap_size, self.gridmap_size)

#                 if self.maxpool_speedmap:
#                     speedmap_t = torch.from_numpy(speedmap).cuda()
#                     ksize = 3
#                     padding = (ksize - 1) // 2
#                     m = torch.nn.MaxPool2d(ksize, stride=1, padding=padding)
#                     speedmap = m(speedmap_t.unsqueeze(0))[0].cpu().numpy()

#                 if ((self.rough_history < self.max_roughness) and
#                         np.abs(self.vel_history - speedmap[xloc[0], yloc[0]]) < self.velocity_margin):
#                     self.cvar_alpha += 0.005
#                 elif self.rough_history > self.max_roughness:
#                     self.cvar_alpha -= 0.02
#                 self.cvar_alpha = np.clip(self.cvar_alpha, 0, .99)

#                 if (np.abs(self.vel_history - speedmap[xloc[0], yloc[0]]) > .4 and
#                         speedmap[xloc[0], yloc[0]] < self.cvar_speed_threshold):
#                     self.cvar_alpha += .01

#             if self.buffer_idx < 30 + self.num_insert:
#                 speedmap = np.zeros((self.gridmap_size, self.gridmap_size)) + self.speed_init
#                 self.cvar_alpha = 0.0

#         costmap /= 2.0
#         ids = np.where(unc_map != 0)
#         costmap[ids] = unc_map[ids]
#         costmap[~np.isfinite(costmap)] = 0.0
#         costmap[zero_mask] = self.unknown_cost
#         speedmap[zero_mask] = self.unknown_speed

#         print('Speedmap Min/Max', speedmap.min(), speedmap.max(), speedmap[xloc[0], yloc[0]])
#         speedmap = np.clip(speedmap, 0, self.max_velocity)

#         costmap_msg = self.costmap_to_gridmap(costmap, info, header=msg_header)
#         self.costmap_pub.publish(costmap_msg)

#         speedmap_msg = self.costmap_to_gridmap(speedmap, info, header=msg_header, costmap_layer='speedmap',
#                                                norm_factor=self.max_velocity)
#         self.speedmap_pub.publish(speedmap_msg)

#         cvar_msg = Float32()
#         cvar_msg.data = self.cvar_alpha
#         self.cvar_pub.publish(cvar_msg)

#         print(time.perf_counter() - now, 'total time', self.buffer_idx, 'buffer_idx')

#     def costmap_to_gridmap(self, costmap, info, header=None, costmap_layer='costmap', variance=None, norm_factor=None):
#         costmap_msg = GridMap()
#         costmap_msg.info = info
#         if header is not None:
#             costmap_msg.header = header
#         costmap_msg.layers = [costmap_layer]

#         def make_layer(data):
#             layer_msg = Float32MultiArray()
#             layer_msg.layout.dim.append(
#                 MultiArrayDimension(label='column_index', size=data.shape[0], stride=data.shape[0]))
#             layer_msg.layout.dim.append(
#                 MultiArrayDimension(label='row_index', size=data.shape[0],
#                                     stride=data.shape[0] * data.shape[1]))
#             layer_msg.data = data[::-1, ::-1].flatten().tolist()
#             return layer_msg

#         costmap_msg.data.append(make_layer(costmap))

#         costmap_msg.layers.append('elevation')
#         if variance is not None:
#             layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 - variance * 5
#         else:
#             layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73
#         costmap_msg.data.append(make_layer(layer_data))

#         if norm_factor is None:
#             vcostmap = np.clip(costmap * 2., 0, 1)
#         else:
#             vcostmap = np.clip(costmap, 0, norm_factor) / norm_factor

#         if costmap_layer == 'costmap':
#             gridmap_cs = (CMAP(vcostmap) * 255).astype(np.int32)
#         else:
#             gridmap_cs = (SMAP(vcostmap) * 255).astype(np.int32)

#         gridmap_color = (gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2])
#         gridmap_color = gridmap_color.view(dtype=np.float32)

#         costmap_msg.layers.append('rgb_viz')
#         rgb_layer_msg = Float32MultiArray()
#         rgb_layer_msg.layout.dim.append(
#             MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
#         rgb_layer_msg.layout.dim.append(
#             MultiArrayDimension(label='row_index', size=layer_data.shape[0],
#                                 stride=layer_data.shape[0] * layer_data.shape[1]))
#         rgb_layer_msg.data = gridmap_color[::-1, ::-1].flatten().tolist()
#         costmap_msg.data.append(rgb_layer_msg)

#         return costmap_msg


# def main(args=None):
#     rclpy.init(args=args)
#     node = rclpy.create_node('_param_reader')
#     node.declare_parameter('cost_topic', '')
#     node.declare_parameter('odom_topic', '')
#     node.declare_parameter('gridmap_topic', '')
#     node.declare_parameter('costmap_topic', '')
#     node.declare_parameter('vel_pub_topic', '')
#     node.declare_parameter('config_file', '')

#     cost_topic = node.get_parameter('cost_topic').value
#     odom_topic = node.get_parameter('odom_topic').value
#     gridmap_topic = node.get_parameter('gridmap_topic').value
#     costmap_topic = node.get_parameter('costmap_topic').value
#     vel_pub_topic = node.get_parameter('vel_pub_topic').value
#     config_file = node.get_parameter('config_file').value
#     node.destroy_node()

#     config_dict = yaml.safe_load(open(config_file, 'r'))

#     context_node = ContextClustererNode(
#         cost_topic, odom_topic, gridmap_topic, costmap_topic, vel_pub_topic, config_dict)

#     try:
#         rclpy.spin(context_node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         context_node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
from array import array

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
from collections import deque, Counter
import json
from std_srvs.srv import Trigger

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
        self.processing_counts = Counter()
        self.processing_durations = deque(maxlen=2048)
        self.create_service(Trigger, "~/processing_stats", self.processing_status)

        assets_dir = os.path.join(get_package_share_directory('context_adaptation'), 'assets') + '/'
        behavior_cfg = config['BEHAVIOR']
        # Keep legacy configs backward compatible.  The active runtime config
        # disables this because the current MPPI setup has w_speed=0.0.
        enable_speedmap = behavior_cfg.get('enable_speedmap', True)
        if not isinstance(enable_speedmap, bool):
            raise ValueError(
                'BEHAVIOR.enable_speedmap must be a YAML boolean '
                '(true or false), got {!r}'.format(enable_speedmap))
        self.enable_speedmap = enable_speedmap

        self.odom_msg = None
        self.hz_counter = 0
        self.image = None
        self.new_msg = False
        self.cost = 0.0
        self.speed_mismatch = 0.0
        self.terrain_mismatch = 0.0
        self.pose_se3 = None

        self.costmap_pub = self.create_publisher(GridMap, costmap_topic, 2)
        self.speedmap_pub = None
        self.max_vel_pub = self.create_publisher(Float32, vel_pub_topic, 2)
        self.cvar_pub = None
        if self.enable_speedmap:
            self.speedmap_pub = self.create_publisher(
                GridMap, '/shortrange_speedmap', 2)
            self.cvar_pub = self.create_publisher(
                Float32, '/hdif_speedmap_cvar', 2)

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
        self.residual_max = behavior_cfg['residual_max']
        self.uc_thresh = behavior_cfg['uncertainty_threshold']
        self.cost_lengthscale = behavior_cfg['cost_lengthscale']
        self.speed_lengthscale = (
            behavior_cfg['speed_lengthscale']
            if self.enable_speedmap else None)
        self.train_kernel = behavior_cfg['train_kernel']

        buffer_len = behavior_cfg['buffer_size']
        self.buffer_update_freq = behavior_cfg['buffer_update_freq']
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

        self.max_roughness = behavior_cfg['max_cost']
        self.max_velocity = behavior_cfg['max_velocity']
        self.velocity_margin = behavior_cfg['velocity_margin']
        self.cvar_alpha = .0

        # ------------------------------------------------------------------
        # Heavy runtime profiler
        # ------------------------------------------------------------------
        # Add this to your YAML if you want to tune it:
        #
        # PROFILE:
        #   enable: true          # turn profiling on/off
        #   period: 1             # print every N run_map cycles
        #   warn_ms: 80.0         # always print if total cycle time exceeds this
        #   cuda_sync: true       # synchronize CUDA before/after each mark for real GPU timing
        #   topk: 12              # number of slowest sections to show
        #   gridmap_detail: true  # print costmap_to_gridmap serialization breakdown
        #   hz_window: 30         # rolling rate window
        #   quiet_console: true   # suppress legacy high-rate print() calls
        #
        # NOTE: cuda_sync=true makes the node slower, but gives accurate GPU timing.
        profile_cfg = config.get('PROFILE', {})
        self.profile_enable = bool(profile_cfg.get('enable', False))
        self.profile_period = max(1, int(profile_cfg.get('period', 1)))
        self.profile_warn_ms = float(profile_cfg.get('warn_ms', 80.0))
        self.profile_cuda_sync = bool(profile_cfg.get('cuda_sync', True))
        self.profile_topk = max(1, int(profile_cfg.get('topk', 12)))
        self.profile_gridmap_detail = bool(profile_cfg.get('gridmap_detail', True))
        self.profile_ema_alpha = min(
            1.0, max(0.0, float(profile_cfg.get('ema_alpha', 0.10))))
        self.profile_hz_window = max(
            2, int(profile_cfg.get('hz_window', 30)))
        self.profile_quiet_console = bool(
            profile_cfg.get('quiet_console', False))
        self.profile_seq = 0
        self.profile_processed_count = 0
        self.profile_ema = {}
        self.profile_last_total_ms = 0.0
        self.profile_cycle_times = deque(maxlen=self.profile_hz_window)
        self.profile_processed_times = deque(maxlen=self.profile_hz_window)

        if self.profile_enable:
            self.get_logger().info(
                'online_hdif profiler enabled: period={}, cuda_sync={}, '
                'hz_window={}, quiet_console={}'.format(
                    self.profile_period,
                    self.profile_cuda_sync,
                    self.profile_hz_window,
                    self.profile_quiet_console,
                ))

        self.rough_history = 0.2
        self.vel_history = 0.0
        self.costmap_cvar = behavior_cfg['costmap_cvar']
        self.unknown_cost = behavior_cfg['unknown_cost']
        self.unknown_speed = (
            behavior_cfg['unknown_speed']
            if self.enable_speedmap else None)
        self.maxpool_speedmap = (
            behavior_cfg['maxpool_speedmap']
            if self.enable_speedmap else False)

        self.speed_init = (
            behavior_cfg['speed_init']
            if self.enable_speedmap else None)
        self.cvar_speed_threshold = self.max_velocity * (2.8 / 6.0)

        avoid_class = 3
        num_insert = 8
        spread = self.max_velocity / (num_insert - 1)
        avoid_data = torch.ones(num_insert, self.VLAD_CLUSTS + 2).cuda() * .9
        avoid_velocities = np.empty(num_insert, dtype=np.float32)
        avoid_classes = np.zeros((num_insert)) + avoid_class
        avoid_labels = torch.ones((num_insert, 1)).cuda()
        avoid_data[:, -1] = 1.0
        for i in range(num_insert):
            avoid_velocity = i * spread
            avoid_data[i, -2] = avoid_velocity
            avoid_velocities[i] = avoid_velocity
            avoid_data[i, :self.VLAD_CLUSTS] = torch.Tensor([
                # Github
                # 19.777752, 23.18438, 21.908875, 18.12688,
                # 25.28553, 23.31651, 21.984331, 24.273615,
                # Road
                # 21.88132095336914, 24.390525817871094, 22.666915893554688, 20.71625518798828,
                # 25.843544006347656, 24.023813247680664, 20.96671485900879, 24.57954216003418,
                # Tree 1
                # 23.78504180908203, 24.359973907470703, 24.878049850463867, 23.789031982421875, 
                # 25.906394958496094, 25.542871475219727, 23.74051284790039, 25.713401794433594,
                # Tree 2
                # 23.406383514404297, 25.758296966552734, 25.794809341430664, 24.845937728881836,
                # 26.53348159790039, 27.235206604003906, 25.33324432373047, 25.70024871826172,
                # ViT-S/14, layer 10, value facet, four-bag centers:
                # clicked obstacle sample 1 (3x3 feature-cell average).
                # 17.825313568115234, 16.624393463134766, 17.05434799194336, 15.66226577758789,
                # 17.130983352661133, 18.56365394592285, 15.951054573059082, 12.038537979125977,
                # Vit-S/14 Tree
                17.47641944885254, 16.395057678222656, 16.922996520996094, 16.886404037475586,
                16.075742721557617, 17.39919090270996, 16.039724349975586, 13.313456535339355,
            ]).cuda() / self.residual_max

        self.train_in_buffer = torch.vstack((avoid_data, self.train_in_buffer))
        self.train_label_buffer = torch.vstack((avoid_labels, self.train_label_buffer))
        self.train_buffer_classes = np.concatenate((avoid_classes, self.train_buffer_classes))
        # Replacement selection only needs the velocity column. Keep an exact
        # float32 CPU mirror so the full CUDA column does not have to be copied
        # and synchronized every time the full buffer replaces one sample.
        self.train_velocity_buffer = np.zeros(
            self.train_in_buffer.shape[0], dtype=np.float32)
        self.train_velocity_buffer[:num_insert] = avoid_velocities
        self.num_insert = num_insert
        self.buffer_idx += self.num_insert

        self.gp = None
        self.likelihood = None
        self.gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'gp_params_speed_input'))

        self.speed_gp = None
        self.speed_likelihood = None
        self.speed_gp_params = None
        if self.enable_speedmap:
            self.speed_gp_params = torch.load(
                os.path.join(assets_dir, 'gp_params', 'speed_gp_params_new'))

        self.speed_gp_indices = (
            [0, 1, 2, 3, 4, 5, 6, 7, 9]
            if self.enable_speedmap else None)

        # ------------------------------------------------------------------
        # Fast GP inference
        # ------------------------------------------------------------------
        # GPyTorch ExactGP posterior is very slow when we query every known grid cell
        # (often 20k+ cells). For runtime inference, use the same RBF GP equations
        # but compute only predictive mean and diagonal variance in small chunks.
        #
        # YAML:
        # GP_OPT:
        #   fast_predict: true
        #   pred_chunk_size: 8192
        #   min_variance: 1.0e-6
        #   jitter: 1.0e-5
        #   fallback_to_gpytorch: true
        gp_opt_cfg = config.get('GP_OPT', {})
        self.fast_gp_enable = bool(gp_opt_cfg.get('fast_predict', True))
        self.fast_gp_chunk_size = max(256, int(gp_opt_cfg.get('pred_chunk_size', 8192)))
        self.fast_gp_min_var = float(gp_opt_cfg.get('min_variance', 1.0e-6))
        self.fast_gp_jitter = float(gp_opt_cfg.get('jitter', 1.0e-5))
        self.fast_gp_fallback_to_gpytorch = bool(gp_opt_cfg.get('fallback_to_gpytorch', True))
        self.fast_gp_triangular_variance = bool(
            gp_opt_cfg.get('triangular_variance', True))
        self.fast_cost_cache = None
        self.fast_speed_cache = None

        self.create_subscription(GridMap, gridmap_topic, self.handle_map, 5)
        self.create_subscription(Float32, cost_topic, self.handle_cost, 1)
        self.create_subscription(Odometry, odom_topic, self.handle_odom, 1)
        if self.enable_speedmap:
            self.create_subscription(
                Float32, '/hdif_max_roughness',
                self.handle_max_roughness, 1)
            self.create_subscription(
                Float32, '/hdif_cvar', self.handle_cvar, 1)

        # Single mutually-exclusive callback group: inference and online GP
        # updates are serialized, with one latest pending map and no work queue.
        self.processing = False
        self.process_guard = self.create_guard_condition(self._process_pending_map)
        self.timer = self.create_timer(0.02, self._request_processing)
        self.get_logger().info(
            'Speedmap pipeline {}'.format(
                'enabled' if self.enable_speedmap else 'disabled'))
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
        self.processing_counts["received"] += 1
        if self.new_msg:
            self.processing_counts["pending_replaced"] += 1
        with self._lock:
            self.dino_map = msg
            self.new_msg = True
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'VLAD' in layer:
                        self.channels.append(layer)
                self.grid_map_cvt.channels = self.channels
        self._request_processing()

    def _request_processing(self):
        if self.new_msg and self.pose_se3 is not None and not self.processing:
            self.process_guard.trigger()

    def _process_pending_map(self):
        if self.processing or not self.new_msg or self.pose_se3 is None:
            return
        self.processing = True
        try:
            self.run_map()
        finally:
            self.processing = False

    def update_train_buffer(self, input, label, class_id, velocity):
        self.train_in_buffer[self.buffer_idx] = input
        self.train_label_buffer[self.buffer_idx] = label
        self.train_buffer_classes[self.buffer_idx] = class_id
        self.train_velocity_buffer[self.buffer_idx] = np.float32(velocity)
        self.buffer_idx = min(self.buffer_idx + 1, self.train_in_buffer.shape[0])
        if self.buffer_idx == self.train_in_buffer.shape[0]:
            self.buffer_full = True

    def insert_train_buffer(self, input, label, class_id, idx, velocity):
        self.train_in_buffer[idx] = input
        self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id
        self.train_velocity_buffer[idx] = np.float32(velocity)

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))
        for i, transform in enumerate(self.tire_transforms):
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]
        return tire_points


    def _rbf_kernel_fast(self, xa, xb, lengthscale, outputscale):
        """
        RBF kernel with ARD lengthscale.
        xa: [N, D], xb: [M, D]
        returns: [N, M]
        """
        xa = xa.float()
        xb = xb.float()
        lengthscale = lengthscale.to(device=xa.device, dtype=xa.dtype).view(1, -1).clamp_min(1e-6)
        outputscale = outputscale.to(device=xa.device, dtype=xa.dtype).view(()).clamp_min(1e-12)

        xa_s = xa / lengthscale
        xb_s = xb / lengthscale
        xa2 = (xa_s * xa_s).sum(dim=1, keepdim=True)
        xb2 = (xb_s * xb_s).sum(dim=1).view(1, -1)
        dist2 = xa2 + xb2 - 2.0 * (xa_s @ xb_s.t())
        dist2 = dist2.clamp_min(0.0)
        return outputscale * torch.exp(-0.5 * dist2)

    def _extract_gp_hyperparams(self, model, likelihood, input_dim):
        """
        Extract actual constrained GP hyperparameters from the GPyTorch modules.
        This lets the fast path stay numerically close to the original model.
        """
        device = self.train_in_buffer.device
        dtype = torch.float32

        try:
            mean = model.mean_module.constant.detach().reshape(-1)[0].to(device=device, dtype=dtype)
        except Exception:
            mean = torch.tensor(0.0, device=device, dtype=dtype)

        try:
            outputscale = model.covar_module.outputscale.detach().reshape(-1)[0].to(device=device, dtype=dtype)
        except Exception:
            outputscale = torch.tensor(1.0, device=device, dtype=dtype)

        try:
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().reshape(-1).to(device=device, dtype=dtype)
        except Exception:
            lengthscale = torch.ones(input_dim, device=device, dtype=dtype)

        if lengthscale.numel() == 1:
            lengthscale = lengthscale.repeat(input_dim)
        elif lengthscale.numel() < input_dim:
            pad = lengthscale[-1].repeat(input_dim - lengthscale.numel())
            lengthscale = torch.cat([lengthscale, pad], dim=0)
        else:
            lengthscale = lengthscale[:input_dim]

        try:
            noise = likelihood.noise.detach().reshape(-1)[0].to(device=device, dtype=dtype)
        except Exception:
            noise = torch.tensor(1.0e-4, device=device, dtype=dtype)

        return {
            'mean': mean,
            'outputscale': outputscale.clamp_min(1.0e-12),
            'lengthscale': lengthscale.clamp_min(1.0e-6),
            'noise': noise.clamp_min(1.0e-8),
        }

    def _build_fast_gp_cache(self, train_x, train_y, model, likelihood, name='gp'):
        """
        Precompute Cholesky factor and alpha for exact RBF GP prediction.

        This replaces repeated GPyTorch posterior construction for large query sets.
        Complexity per frame becomes roughly O(N_query * N_train * D), with small
        chunked matrix multiplies, instead of a heavy posterior object over all cells.
        """
        if train_x is None or train_y is None or train_x.numel() == 0:
            return None

        device = self.train_in_buffer.device
        train_x = train_x.detach().to(device=device, dtype=torch.float32).contiguous()
        train_y = train_y.detach().reshape(-1).to(device=device, dtype=torch.float32).contiguous()

        n_train = train_x.shape[0]
        input_dim = train_x.shape[1]
        hp = self._extract_gp_hyperparams(model, likelihood, input_dim)

        K = self._rbf_kernel_fast(train_x, train_x, hp['lengthscale'], hp['outputscale'])
        eye = torch.eye(n_train, device=device, dtype=torch.float32)

        # Try increasing jitter if the train kernel is close to singular.
        L = None
        last_err = None
        base_diag = hp['noise']
        for scale in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
            jitter = self.fast_gp_jitter * scale
            try:
                L = torch.linalg.cholesky(K + (base_diag + jitter) * eye)
                break
            except RuntimeError as exc:
                last_err = exc

        if L is None:
            self.get_logger().warning(
                'Fast GP cache build failed for {}. Falling back to GPyTorch if enabled. err={}'.format(
                    name, str(last_err)))
            return None

        y_centered = train_y - hp['mean']
        alpha = torch.cholesky_solve(y_centered.view(-1, 1), L).view(-1)

        return {
            'name': name,
            'train_x': train_x,
            'train_y': train_y,
            'mean': hp['mean'],
            'outputscale': hp['outputscale'],
            'lengthscale': hp['lengthscale'],
            'noise': hp['noise'],
            'L': L,
            'alpha': alpha,
            'n_train': n_train,
            'input_dim': input_dim,
        }

    def _refresh_fast_gp_caches(self, train_in_norm):
        """Refresh fast prediction caches after GP model rebuild/load/training."""
        if not self.fast_gp_enable:
            self.fast_cost_cache = None
            self.fast_speed_cache = None
            return

        try:
            self.fast_cost_cache = self._build_fast_gp_cache(
                train_in_norm[:, :-1],
                train_in_norm[:, -1],
                self.gp,
                self.likelihood,
                name='cost',
            )
        except Exception as exc:
            self.fast_cost_cache = None
            self.get_logger().warning(
                'Fast cost GP cache refresh failed. Falling back to GPyTorch '
                'if enabled. err={}'.format(str(exc)))

        if not self.enable_speedmap:
            self.fast_speed_cache = None
            return

        try:
            self.fast_speed_cache = self._build_fast_gp_cache(
                train_in_norm[:, self.speed_gp_indices],
                train_in_norm[:, -2],
                self.speed_gp,
                self.speed_likelihood,
                name='speed',
            )
        except Exception as exc:
            self.fast_speed_cache = None
            self.get_logger().warning(
                'Fast speed GP cache refresh failed. Falling back to GPyTorch '
                'if enabled. err={}'.format(str(exc)))

    def _fast_gp_predict_cached(self, query_x, cache):
        """
        Predict mean and observed diagonal variance for query_x using cached exact RBF GP.

        Returns tensors on the same CUDA device:
          mean: [N]
          variance: [N]
        """
        if cache is None:
            raise RuntimeError('fast GP cache is None')

        device = self.train_in_buffer.device
        query_x = query_x.detach().to(device=device, dtype=torch.float32).contiguous()
        n_query = query_x.shape[0]
        if n_query == 0:
            return (
                torch.empty(0, device=device, dtype=torch.float32),
                torch.empty(0, device=device, dtype=torch.float32),
            )

        train_x = cache['train_x']
        lengthscale = cache['lengthscale']
        outputscale = cache['outputscale']
        noise = cache['noise']
        mean_const = cache['mean']
        L = cache['L']
        alpha = cache['alpha']

        means = torch.empty(
            n_query, device=device, dtype=torch.float32)
        variances = torch.empty_like(means)
        chunk = max(256, int(self.fast_gp_chunk_size))

        for start in range(0, n_query, chunk):
            end = min(start + chunk, n_query)
            xq = query_x[start:end]

            K_qn = self._rbf_kernel_fast(xq, train_x, lengthscale, outputscale)
            pred_mean = mean_const + K_qn @ alpha

            # Diagonal predictive variance:
            # k(x,x) - k_xX K_XX^-1 k_Xx + likelihood noise.
            if self.fast_gp_triangular_variance:
                # K_XX = L L^T, therefore:
                #   k K_XX^-1 k^T = ||L^-1 k^T||^2
                # Only one triangular solve is needed for the diagonal,
                # instead of the forward+backward solves in cholesky_solve.
                solved_lower = torch.linalg.solve_triangular(
                    L, K_qn.t(), upper=False)
                latent_var = outputscale - solved_lower.square().sum(dim=0)
            else:
                # Diagnostic rollback path for comparing against the previous
                # implementation on a target platform.
                solved = torch.cholesky_solve(K_qn.t(), L)
                latent_var = outputscale - (K_qn * solved.t()).sum(dim=1)
            pred_var = latent_var.clamp_min(self.fast_gp_min_var) + noise

            means[start:end] = pred_mean
            variances[start:end] = pred_var

        return means, variances

    def _gpytorch_predict_fallback(self, query_x, model, likelihood):
        """Original GPyTorch prediction path, used only if fast GP is disabled/fails."""
        observed_pred = likelihood(model(query_x))
        return observed_pred.mean, observed_pred.variance

    def _safe_map_lookup(self, arr, xloc, yloc):
        """Safe map lookup for tire-local speed/cost values."""
        sx = int(np.clip(int(xloc[0]), 0, arr.shape[0] - 1))
        sy = int(np.clip(int(yloc[0]), 0, arr.shape[1] - 1))
        return arr[sx, sy], sx, sy

    def _ensure_map_shape(self, arr, target_shape, fill_value=0.0):
        """
        Make sure costmap/speedmap remains in the original GridMap resolution.
        This prevents index errors if any optional operation changes map size.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == target_shape:
            return arr
        if arr.size == 0:
            return np.full(target_shape, fill_value, dtype=np.float32)
        return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


    def _profile_sync_cuda(self):
        """Synchronize CUDA so profiler times include queued GPU kernels."""
        if self.profile_enable and self.profile_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _profile_rolling_hz(timestamps):
        """Return the effective rate over the current rolling time window."""
        if len(timestamps) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(timestamps) - 1) / elapsed

    def _legacy_print(self, *args):
        """Silence high-rate legacy prints when quiet-console mode is enabled."""
        if not self.profile_quiet_console:
            print(*args)

    def _profile_start(self):
        """Start a per-run_map profiling record."""
        if not self.profile_enable:
            return None

        self._profile_sync_cuda()
        t = time.perf_counter()
        self.profile_cycle_times.append(t)

        cuda_alloc = 0
        cuda_reserved = 0
        if torch.cuda.is_available():
            # Reset process-local peak counters so max_memory_* describes this
            # run_map callback, including short-lived GP inference allocations.
            torch.cuda.reset_peak_memory_stats()
            cuda_alloc = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()

        return {
            't0': t,
            'last': t,
            'cpu0': time.process_time(),
            'items': [],
            'cuda_alloc0': cuda_alloc,
            'cuda_reserved0': cuda_reserved,
            'meta': {},
        }

    def _profile_mark(self, prof, name, extra=''):
        """Record elapsed time since the previous mark."""
        if prof is None:
            return

        self._profile_sync_cuda()
        now = time.perf_counter()
        dt_ms = (now - prof['last']) * 1000.0
        prof['last'] = now
        prof['items'].append((name, dt_ms, extra))

        old = self.profile_ema.get(name, dt_ms)
        self.profile_ema[name] = (1.0 - self.profile_ema_alpha) * old + self.profile_ema_alpha * dt_ms

    def _profile_report(self, prof, total_extra='', output_published=False):
        """Print slow-section report for one run_map cycle."""
        if prof is None:
            return

        self._profile_sync_cuda()
        now = time.perf_counter()
        total_ms = (now - prof['t0']) * 1000.0
        cpu_ms = (time.process_time() - prof['cpu0']) * 1000.0
        self.profile_last_total_ms = total_ms
        self.profile_seq += 1

        if output_published:
            self.profile_processed_count += 1
            self.profile_processed_times.append(now)

        total_key = (
            '__total_output__' if output_published else '__total_early__')
        old_total = self.profile_ema.get(total_key, total_ms)
        alpha = self.profile_ema_alpha
        total_ema = (1.0 - alpha) * old_total + alpha * total_ms
        self.profile_ema[total_key] = total_ema

        period_counter = (
            self.profile_processed_count if output_published else self.profile_seq)
        should_print = (
            (period_counter % self.profile_period) == 0
            or total_ms >= self.profile_warn_ms
        )
        if not should_print:
            return

        sorted_items = sorted(prof['items'], key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:self.profile_topk]

        lines = []
        # IMPORTANT: rclpy can raise 'Logger severity cannot be changed between calls'
        # if the same call site sometimes logs with info() and sometimes with warning().
        # Keep a single severity here and mark slow frames in the message text instead.
        level = self.get_logger().info

        lines.append(
            (
                '[PROFILE online_hdif #{:d}] total={:.2f} ms '
                '(ema={:.2f} ms, capacity={:.2f} Hz){} '
                '| kind={} | buffer_idx={} | grid={} | vel={:.3f} | cvar={:.3f}'
            ).format(
                self.profile_seq,
                total_ms,
                total_ema,
                1000.0 / max(total_ema, 1e-9),
                '  SLOW' if total_ms >= self.profile_warn_ms else '',
                'output' if output_published else 'early-return',
                self.buffer_idx,
                self.gridmap_size,
                float(self.velocity),
                float(self.cvar_alpha),
            )
        )
        lines.append(
            (
                '  rate: timer_callback={:.2f} Hz, costmap_output={:.2f} Hz '
                '(published={}) | mode={}'
            ).format(
                self._profile_rolling_hz(self.profile_cycle_times),
                self._profile_rolling_hz(self.profile_processed_times),
                self.profile_processed_count,
                (
                    'accurate-gpu-sync'
                    if self.profile_cuda_sync
                    else 'low-overhead-async'
                ),
            )
        )
        lines.append(
            '  CPU process: {:.2f} ms ({:.1f}% core-equivalent)'.format(
                cpu_ms,
                cpu_ms / max(total_ms, 1e-9) * 100.0,
            )
        )

        if total_extra:
            lines.append('  meta: {}'.format(total_extra))

        for name, dt_ms, extra in top_items:
            pct = (dt_ms / max(total_ms, 1e-9)) * 100.0
            ema = self.profile_ema.get(name, dt_ms)
            suffix = ' | {}'.format(extra) if extra else ''
            lines.append('  {:>7.2f} ms ({:>5.1f}%)  ema={:>7.2f}  {}'.format(
                dt_ms, pct, ema, name + suffix))

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            d_alloc = alloc - prof.get('cuda_alloc0', 0)
            d_reserved = reserved - prof.get('cuda_reserved0', 0)
            lines.append(
                (
                    '  CUDA mem: live alloc={:.1f}MB ({:+.1f}), '
                    'reserved={:.1f}MB ({:+.1f}); '
                    'frame peak alloc={:.1f}MB, reserved={:.1f}MB'
                ).format(
                    alloc / 1024**2,
                    d_alloc / 1024**2,
                    reserved / 1024**2,
                    d_reserved / 1024**2,
                    peak_alloc / 1024**2,
                    peak_reserved / 1024**2,
                )
            )

        level('\n'.join(lines))

    def processing_status(self, request, response):
        samples = sorted(self.processing_durations)
        response.success = True
        response.message = json.dumps({
            'sample_wall_s': time.monotonic(),
            'schedule_mode': 'input_guard',
            'counts': dict(self.processing_counts),
            'processing_ms': {
                'n': len(samples),
                'mean': sum(samples) / len(samples) if samples else 0,
                'p95': samples[int((len(samples)-1)*.95)] if samples else 0,
                'max': samples[-1] if samples else 0,
            },
        })
        return response

    def run_map(self):
        self.processing_counts['calls'] += 1
        if self.pose_se3 is None:
            self.processing_counts['no_odom'] += 1
        elif not self.new_msg:
            self.processing_counts['no_new_map'] += 1
        published = self.processing_counts['published']
        start = time.perf_counter()
        try:
            return self._run_map_impl()
        finally:
            if self.processing_counts['published'] > published:
                self.processing_durations.append((time.perf_counter()-start)*1000)

    def _run_map_impl(self):
        prof = self._profile_start()
        total_extra = ''

        if self.pose_se3 is None:
            self._legacy_print('waiting for odom...')
            self._profile_mark(prof, 'early_wait_odom')
            self._profile_report(prof, 'early_return=waiting_for_odom')
            return

        now = time.perf_counter()
        self._legacy_print('----')
        if self.hz_counter == 5000000:
            self.hz_counter = 0
        self.hz_counter += 1
        self._profile_mark(prof, 'precheck_hz')

        if not self.new_msg:
            self._legacy_print('no new map')
            self._profile_mark(prof, 'early_no_new_map')
            self._profile_report(prof, 'early_return=no_new_map')
            return

        with self._lock:
            if self.dino_map is not None:
                info = self.dino_map.info
                # Grid lengths/resolution are float32-derived. A 200-cell
                # map can divide to 199.999996; truncation silently resizes
                # every feature layer to 199 and blends unknown boundaries.
                nx = round(info.length_x / info.resolution)
                ny = round(info.length_y / info.resolution)
                self.grid_map_cvt.size = [nx, ny]
                self.gridmap_size = nx
                gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map)
                msg_header = self.dino_map.header
                self.new_msg = False
            else:
                gridmap = None

        self._profile_mark(
            prof,
            'lock+ros_to_numpy',
            'grid={}x{} layers={}'.format(
                nx if 'nx' in locals() else 'NA',
                ny if 'ny' in locals() else 'NA',
                len(self.channels),
            )
        )

        if gridmap is None:
            self._legacy_print('NO MAP')
            self._profile_mark(prof, 'early_gridmap_none')
            self._profile_report(prof, 'early_return=gridmap_none')
            return

        # ------------------------------------------------------------------
        # Uncertainty map / class map
        # ------------------------------------------------------------------
        da = gridmap['data'].argmin(axis=0)
        unc_map = gridmap['data'].min(axis=0)
        unc_map /= self.residual_max
        unc_map[unc_map < self.uc_thresh] = 0
        self._profile_mark(
            prof,
            'uncertainty_argmin_min',
            'data_shape={}'.format(gridmap['data'].shape)
        )

        e_kernel = np.ones((2, 2), np.float32)
        unc_map = cv2.erode(unc_map, e_kernel, iterations=1)
        unc_map = cv2.dilate(unc_map, e_kernel, iterations=1)
        self._profile_mark(prof, 'cv2_erode+dilate')

        # ------------------------------------------------------------------
        # Robot tire footprint lookup
        # ------------------------------------------------------------------
        P = gridmap['metadata']['origin']
        res = gridmap['metadata']['resolution']
        tp = self.estimate_tire_points()
        tx = np.floor((tp[:, 0] - P[0]) / res)
        ty = np.floor((tp[:, 1] - P[1]) / res)
        xloc, yloc = ty.astype(int), tx.astype(int)
        self._profile_mark(prof, 'tire_points+index')

        if (np.any(xloc < 0) or np.any(yloc < 0) or
                np.any(xloc >= self.gridmap_size) or np.any(yloc >= self.gridmap_size)):
            print('WARNING: OUT OF BOUNDS! Not computing costmap')
            self._profile_mark(prof, 'early_out_of_bounds')
            self._profile_report(
                prof,
                'early_return=out_of_bounds xloc={} yloc={} grid={}'.format(
                    xloc.tolist(), yloc.tolist(), self.gridmap_size)
            )
            return

        classes = da[xloc, yloc]
        spot = stats.mode(classes)[0]
        self._profile_mark(prof, 'stats_mode_tire_class')

        # ------------------------------------------------------------------
        # Torch input build / known mask
        # ------------------------------------------------------------------
        input_t = torch.from_numpy(gridmap['data'] / self.residual_max)
        zero_mask = (input_t == 0).all(dim=0).cpu().numpy()
        known_mask = ~zero_mask
        known_ratio = float(np.count_nonzero(known_mask)) / float(known_mask.size)
        total_extra = 'known_ratio={:.3f}, zero_cells={}, total_cells={}'.format(
            known_ratio, int(np.count_nonzero(zero_mask)), int(zero_mask.size))
        self._profile_mark(prof, 'torch_from_numpy+zero_mask', total_extra)

        # ------------------------------------------------------------------
        # Training buffer update
        # ------------------------------------------------------------------
        buffer_update_reason = 'skip_velocity'
        if (self.velocity > .3) and self.hz_counter % self.buffer_update_freq == 0:
            buffer_update_reason = 'attempt'
            feature_sample = input_t[:, xloc, yloc].mean(dim=1)
            # input_t is still on CPU here. Test unknown space before the
            # transfer so a Python branch does not synchronize the CUDA queue.
            if torch.count_nonzero(feature_sample).item() == 0:
                buffer_update_reason = 'unknown_tire_sample'
                self._legacy_print('IN UNKNOWN SPACE')
            else:
                input_sample = torch.cat((
                    feature_sample,
                    torch.tensor(
                        [self.velocity, self.cost],
                        dtype=feature_sample.dtype),
                )).cuda()
                if self.buffer_full:
                    most_class = stats.mode(self.train_buffer_classes)[0]
                    class_mask = self.train_buffer_classes == most_class
                    # np.histogram has the same 10 equal-width [0, 10]
                    # bins as torch.histc, while avoiding CUDA histc,
                    # argmax synchronization, and a full-column D2H copy.
                    vel_hist, _ = np.histogram(
                        self.train_velocity_buffer[class_mask],
                        bins=10,
                        range=(0, 10),
                    )
                    most_vel = int(np.argmax(vel_hist))
                    edges = np.arange(0, 11)
                    sel_min = edges[most_vel]
                    sel_max = edges[most_vel + 1]
                    insert_idx = np.random.choice(np.where(
                        class_mask &
                        (self.train_velocity_buffer > sel_min) &
                        (self.train_velocity_buffer < sel_max))[0])
                    if insert_idx > self.num_insert:
                        self.insert_train_buffer(
                            input_sample,
                            self.cost,
                            spot,
                            insert_idx,
                            self.velocity,
                        )
                        buffer_update_reason = 'replace_existing'
                    else:
                        buffer_update_reason = 'replacement_guard'
                else:
                    self.update_train_buffer(
                        input_sample,
                        self.cost,
                        spot,
                        self.velocity,
                    )
                    buffer_update_reason = 'append'
        elif self.velocity > .3:
            buffer_update_reason = 'skip_frequency'
        total_extra += ', buffer_update={}, buffer_idx={}'.format(
            buffer_update_reason, self.buffer_idx)
        self._profile_mark(
            prof,
            'train_buffer_update',
            'reason={}, buffer_idx={}'.format(
                buffer_update_reason, self.buffer_idx),
        )

        costmap = np.zeros(self.gridmap_size * self.gridmap_size)
        speedmap = (
            np.zeros(self.gridmap_size * self.gridmap_size) + self.speed_init
            if self.enable_speedmap else None)
        self._profile_mark(
            prof, 'init_map_arrays',
            'speedmap={}'.format('enabled' if self.enable_speedmap else 'disabled'))

        # ------------------------------------------------------------------
        # GP model refresh / optional kernel training
        # ------------------------------------------------------------------
        if (self.hz_counter % 10 == 0 or self.gp is None or
                (self.enable_speedmap and self.speed_gp is None)):
            self._legacy_print('****************************************************')
            if not self.buffer_full:
                train_in = self.train_in_buffer[:self.buffer_idx]
            else:
                train_in = self.train_in_buffer

            self.in_mean = torch.mean(train_in, dim=0)
            self.in_std = torch.std(train_in, dim=0).clamp(min=1e-6)
            train_in_norm = (train_in - self.in_mean) / self.in_std
            model_refresh_mode = (
                'initialize' if self.gp is None else 'reuse')
            if self.gp is None:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
                self.gp = ExactGPModel(
                    train_in_norm[:, :-1],
                    train_in_norm[:, -1],
                    self.likelihood,
                    self.cost_lengthscale,
                ).cuda()
                if self.gp_params is not None:
                    self.gp.load_state_dict(self.gp_params)
            else:
                # Hyperparameters are unchanged (train_kernel is false in the
                # active configuration), so update only the training snapshot.
                # set_train_data also clears GPyTorch's prediction cache, which
                # keeps the fallback path correct without recreating modules.
                self.gp.set_train_data(
                    inputs=train_in_norm[:, :-1],
                    targets=train_in_norm[:, -1],
                    strict=False,
                )

            if self.enable_speedmap:
                if self.speed_gp is None:
                    self.speed_likelihood = (
                        gpytorch.likelihoods.GaussianLikelihood().cuda())
                    self.speed_gp = ExactGPModel(
                        train_in_norm[:, self.speed_gp_indices],
                        train_in_norm[:, -2],
                        self.speed_likelihood,
                        self.speed_lengthscale,
                    ).cuda()
                    if self.speed_gp_params is not None:
                        self.speed_gp.load_state_dict(self.speed_gp_params)
                else:
                    self.speed_gp.set_train_data(
                        inputs=train_in_norm[:, self.speed_gp_indices],
                        targets=train_in_norm[:, -2],
                        strict=False,
                    )

            if self.fast_gp_enable:
                self._refresh_fast_gp_caches(train_in_norm)
            self._profile_mark(
                prof,
                'gp_refresh',
                'train_n={}, fast_gp={}, model={}'.format(
                    train_in.shape[0],
                    self.fast_gp_enable,
                    model_refresh_mode,
                ),
            )

        elif self.enable_speedmap and self.train_kernel:
            if not self.buffer_full:
                train_in = self.train_in_buffer[:self.buffer_idx]
            else:
                train_in = self.train_in_buffer
            if train_in.shape[0] < 13:
                self._profile_mark(prof, 'train_kernel_skip_lt13')
                self._profile_report(prof, total_extra + ', early_return=train_kernel_lt13')
                return
            optimizer = torch.optim.SGD(self.speed_gp.parameters(), lr=0.01)
            with gpytorch.settings.debug(False):
                self.speed_gp.train()
                self.speed_likelihood.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.speed_likelihood, self.speed_gp)
                optimizer.zero_grad()
                output = self.speed_gp(train_in[:, self.speed_gp_indices])
                loss = -mll(output, train_in[:, -2])
                self._legacy_print('LOSS - ', loss.item())
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                self.speed_gp_params = self.speed_gp.state_dict()
                if self.fast_gp_enable:
                    train_in_norm = (train_in - self.in_mean) / self.in_std
                    self._refresh_fast_gp_caches(train_in_norm)
            self._profile_mark(prof, 'train_kernel_sgd', 'train_n={}, fast_gp={}'.format(
                train_in.shape[0], self.fast_gp_enable))
        else:
            self._profile_mark(prof, 'gp_reuse_no_train')

        # ------------------------------------------------------------------
        # GP inference / cost map generation (+ optional speed map)
        # ------------------------------------------------------------------
        with torch.inference_mode():
            if self.buffer_idx < 5 + self.num_insert:
                costmap = np.zeros(
                    (self.gridmap_size, self.gridmap_size),
                    dtype=np.float32)
                if self.enable_speedmap:
                    speedmap = np.zeros_like(costmap) + self.speed_init
                self._profile_mark(
                    prof, 'gp_skip_small_buffer',
                    'buffer_idx={}'.format(self.buffer_idx))
            else:
                self.gp.eval()
                self.likelihood.eval()
                if self.enable_speedmap:
                    self.speed_gp.eval()
                    self.speed_likelihood.eval()
                self._profile_mark(
                    prof, 'gp_eval_mode',
                    'speedmap={}'.format(
                        'enabled' if self.enable_speedmap else 'disabled'))

                feature_t = input_t.permute(
                    1, 2, 0).view(-1, self.VLAD_CLUSTS)
                zero_mask_f = zero_mask.flatten()
                keep_mask = ~zero_mask_f
                feature_t = feature_t[keep_mask]
                self._profile_mark(
                    prof,
                    'flatten+keep_known',
                    'known_points={}/{}'.format(
                        feature_t.shape[0], zero_mask_f.shape[0])
                )

                # Cost GP input is [VLAD, velocity].  Speed GP input, when
                # enabled, is [VLAD, roughness/cost].  Keeping these tensors
                # separate avoids even constructing the speed-only query when
                # speedmap generation is disabled.
                vel_append = torch.full(
                    (feature_t.shape[0], 1),
                    float(self.velocity),
                    dtype=torch.float32)
                cost_input_t = torch.hstack(
                    (feature_t.float(), vel_append)).cuda(
                        non_blocking=True)
                cost_input_t = (
                    (cost_input_t - self.in_mean[:-1])
                    / self.in_std[:-1])

                speed_input_t = None
                if self.enable_speedmap:
                    rough_append = torch.full(
                        (feature_t.shape[0], 1),
                        float(self.max_roughness),
                        dtype=torch.float32)
                    speed_input_t = torch.hstack(
                        (feature_t.float(), rough_append)).cuda(
                            non_blocking=True)
                    speed_input_t = (
                        (speed_input_t
                         - self.in_mean[self.speed_gp_indices])
                        / self.in_std[self.speed_gp_indices])

                self._profile_mark(
                    prof,
                    'build_gp_queries+cuda+norm',
                    'cost={}, speed={}'.format(
                        tuple(cost_input_t.shape),
                        (tuple(speed_input_t.shape)
                         if speed_input_t is not None else 'disabled')),
                )

                # Fast exact RBF GP prediction.
                # This computes only mean and diagonal variance, which is all
                # this node uses.
                if self.fast_gp_enable and self.fast_cost_cache is not None:
                    try:
                        cost_pred, cost_var = (
                            self._fast_gp_predict_cached(
                                cost_input_t, self.fast_cost_cache))
                        self._profile_mark(
                            prof,
                            'fast_gp_cost_predict',
                            'query_n={}, train_n={}, chunk={}'.format(
                                cost_input_t.shape[0],
                                self.fast_cost_cache.get('n_train', -1),
                                self.fast_gp_chunk_size,
                            )
                        )
                    except Exception as exc:
                        if not self.fast_gp_fallback_to_gpytorch:
                            raise
                        self.get_logger().warning(
                            'fast cost GP failed; fallback to GPyTorch. '
                            'err={}'.format(str(exc)))
                        cost_pred, cost_var = (
                            self._gpytorch_predict_fallback(
                                cost_input_t, self.gp, self.likelihood))
                        self._profile_mark(
                            prof, 'gpytorch_cost_predict_fallback')
                else:
                    cost_pred, cost_var = (
                        self._gpytorch_predict_fallback(
                            cost_input_t, self.gp, self.likelihood))
                    self._profile_mark(prof, 'gpytorch_cost_predict')

                if self.enable_speedmap:
                    if (self.fast_gp_enable
                            and self.fast_speed_cache is not None):
                        try:
                            speed_pred, speed_var = (
                                self._fast_gp_predict_cached(
                                    speed_input_t,
                                    self.fast_speed_cache))
                            self._profile_mark(
                                prof,
                                'fast_gp_speed_predict',
                                'query_n={}, train_n={}, chunk={}'.format(
                                    speed_input_t.shape[0],
                                    self.fast_speed_cache.get(
                                        'n_train', -1),
                                    self.fast_gp_chunk_size,
                                )
                            )
                        except Exception as exc:
                            if not self.fast_gp_fallback_to_gpytorch:
                                raise
                            self.get_logger().warning(
                                'fast speed GP failed; fallback to GPyTorch. '
                                'err={}'.format(str(exc)))
                            speed_pred, speed_var = (
                                self._gpytorch_predict_fallback(
                                    speed_input_t,
                                    self.speed_gp,
                                    self.speed_likelihood))
                            self._profile_mark(
                                prof, 'gpytorch_speed_predict_fallback')
                    else:
                        speed_pred, speed_var = (
                            self._gpytorch_predict_fallback(
                                speed_input_t,
                                self.speed_gp,
                                self.speed_likelihood))
                        self._profile_mark(
                            prof, 'gpytorch_speed_predict')

                    phi = stats.norm.pdf(
                        stats.norm.ppf(self.cvar_alpha))
                    cvar = (
                        speed_pred
                        + (speed_var * phi)
                        / (1.0 - self.cvar_alpha))
                    speed_pred = (
                        (cvar * self.in_std[-2])
                        + self.in_mean[-2])
                    self._profile_mark(prof, 'speed_cvar_math')

                cmap_cvar = self.costmap_cvar
                phi = stats.norm.pdf(stats.norm.ppf(cmap_cvar))
                cvar = (
                    cost_pred
                    + (cost_var * phi)
                    / (1.0 - cmap_cvar))
                cost_pred = (
                    (cvar * self.in_std[-1])
                    + self.in_mean[-1])
                self._profile_mark(prof, 'cost_cvar_math')

                costmap[keep_mask] = (
                    cost_pred.detach().cpu().numpy())
                costmap = costmap.reshape(
                    self.gridmap_size,
                    self.gridmap_size).astype(
                        np.float32, copy=False)
                if self.enable_speedmap:
                    speedmap[keep_mask] = (
                        speed_pred.detach().cpu().numpy())
                    speedmap = speedmap.reshape(
                        self.gridmap_size,
                        self.gridmap_size).astype(
                            np.float32, copy=False)
                self._profile_mark(
                    prof, 'cpu_copy+reshape_maps',
                    'speedmap={}'.format(
                        'enabled' if self.enable_speedmap else 'disabled'))

                target_shape = (
                    self.gridmap_size, self.gridmap_size)
                costmap = self._ensure_map_shape(
                    costmap, target_shape, fill_value=0.0)

                if self.enable_speedmap:
                    if self.maxpool_speedmap:
                        speedmap_t = torch.from_numpy(
                            speedmap).cuda(non_blocking=True)
                        ksize = 3
                        padding = (ksize - 1) // 2
                        # stride must stay 1. stride=2 shrinks 200x200 ->
                        # 100x100 and breaks tire indexing.
                        speedmap = F.max_pool2d(
                            speedmap_t.unsqueeze(0),
                            ksize,
                            stride=1,
                            padding=padding)[0].detach().cpu().numpy()
                        self._profile_mark(
                            prof, 'speedmap_maxpool_cuda')
                    else:
                        self._profile_mark(
                            prof, 'speedmap_maxpool_skip')

                    speedmap = self._ensure_map_shape(
                        speedmap,
                        target_shape,
                        fill_value=self.speed_init)
                    local_speed, sx, sy = self._safe_map_lookup(
                        speedmap, xloc, yloc)

                    if ((self.rough_history < self.max_roughness)
                            and np.abs(
                                self.vel_history - local_speed)
                            < self.velocity_margin):
                        self.cvar_alpha += 0.005
                    elif self.rough_history > self.max_roughness:
                        self.cvar_alpha -= 0.02
                    self.cvar_alpha = np.clip(
                        self.cvar_alpha, 0, .99)

                    if (np.abs(self.vel_history - local_speed) > .4
                            and local_speed
                            < self.cvar_speed_threshold):
                        self.cvar_alpha += .01
                    self._profile_mark(
                        prof,
                        'adaptive_cvar_update',
                        'local_speed={:.3f} at ({},{})'.format(
                            float(local_speed), sx, sy))

            if (self.enable_speedmap
                    and self.buffer_idx < 30 + self.num_insert):
                speedmap = np.zeros(
                    (self.gridmap_size, self.gridmap_size),
                    dtype=np.float32) + self.speed_init
                self.cvar_alpha = 0.0
                self._profile_mark(
                    prof, 'speedmap_force_init_small_buffer')
            elif self.enable_speedmap:
                self._profile_mark(
                    prof, 'speedmap_force_init_skip')

        # ------------------------------------------------------------------
        # Postprocess maps
        # ------------------------------------------------------------------
        target_shape = (self.gridmap_size, self.gridmap_size)
        costmap = self._ensure_map_shape(
            costmap, target_shape, fill_value=0.0)

        costmap /= 2.0
        ids = np.where(unc_map != 0)
        costmap[ids] = unc_map[ids]
        costmap[~np.isfinite(costmap)] = 0.0
        costmap[zero_mask] = self.unknown_cost
        self._profile_mark(
            prof, 'postprocess_masks_unknown',
            'speedmap={}'.format(
                'enabled' if self.enable_speedmap else 'disabled'))

        if self.enable_speedmap:
            speedmap = self._ensure_map_shape(
                speedmap, target_shape, fill_value=self.speed_init)
            speedmap[zero_mask] = self.unknown_speed
            local_speed, _, _ = self._safe_map_lookup(
                speedmap, xloc, yloc)
            self._legacy_print(
                'Speedmap Min/Max',
                speedmap.min(),
                speedmap.max(),
                local_speed)
            speedmap = np.clip(
                speedmap, 0, self.max_velocity)
            self._profile_mark(prof, 'speedmap_clip')

        # ------------------------------------------------------------------
        # ROS GridMap serialization + publish
        # ------------------------------------------------------------------
        costmap_msg = self.costmap_to_gridmap(
            costmap, info, header=msg_header)
        self._profile_mark(prof, 'serialize_costmap_msg')

        self.costmap_pub.publish(costmap_msg)
        self.processing_counts["published"] += 1
        self._profile_mark(prof, 'publish_costmap')

        if self.enable_speedmap:
            speedmap_msg = self.costmap_to_gridmap(
                speedmap,
                info,
                header=msg_header,
                costmap_layer='speedmap',
                norm_factor=self.max_velocity)
            self._profile_mark(prof, 'serialize_speedmap_msg')

            self.speedmap_pub.publish(speedmap_msg)
            self._profile_mark(prof, 'publish_speedmap')

            cvar_msg = Float32()
            cvar_msg.data = self.cvar_alpha
            self.cvar_pub.publish(cvar_msg)
            self._profile_mark(prof, 'publish_cvar')

        self._legacy_print(
            time.perf_counter() - now, 'total time',
            self.buffer_idx, 'buffer_idx')
        self._profile_report(
            prof, total_extra, output_published=True)

    def costmap_to_gridmap(self, costmap, info, header=None, costmap_layer='costmap', variance=None, norm_factor=None):
        # Keep message layout/orientation unchanged while avoiding creation of
        # one Python float object per grid cell.
        _detail_profile = (
            getattr(self, 'profile_enable', False)
            and getattr(self, 'profile_gridmap_detail', False)
            and (self.hz_counter % max(1, getattr(self, 'profile_period', 1)) == 0)
        )
        _t0 = time.perf_counter()
        _last = _t0
        _marks = []

        def _mark(name):
            nonlocal _last
            if not _detail_profile:
                return
            now_t = time.perf_counter()
            _marks.append((name, (now_t - _last) * 1000.0))
            _last = now_t

        costmap_msg = GridMap()
        costmap_msg.info = info
        if header is not None:
            costmap_msg.header = header
        costmap_msg.layers = [costmap_layer]
        _mark('init_msg')

        def make_layer(data, label):
            lt0 = time.perf_counter()
            layer_msg = Float32MultiArray()
            layer_msg.layout.dim.append(
                MultiArrayDimension(label='column_index', size=data.shape[0], stride=data.shape[0]))
            layer_msg.layout.dim.append(
                MultiArrayDimension(label='row_index', size=data.shape[0],
                                    stride=data.shape[0] * data.shape[1]))

            pack_t0 = time.perf_counter()
            oriented = np.ascontiguousarray(
                data[::-1, ::-1], dtype=np.float32)
            packed = array('f')
            packed.frombytes(oriented.tobytes())
            layer_msg.data = packed
            pack_ms = (time.perf_counter() - pack_t0) * 1000.0

            if _detail_profile:
                _marks.append(('make_layer:{} total'.format(label), (time.perf_counter() - lt0) * 1000.0))
                _marks.append(('make_layer:{} pack_array'.format(label), pack_ms))
            return layer_msg

        costmap_msg.data.append(make_layer(costmap, costmap_layer))
        _mark('append_{}_layer'.format(costmap_layer))

        costmap_msg.layers.append('elevation')
        if variance is not None:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 - variance * 5
        else:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73
        _mark('make_elevation_numpy')

        costmap_msg.data.append(make_layer(layer_data, 'elevation'))
        _mark('append_elevation_layer')

        if norm_factor is None:
            vcostmap = np.clip(costmap * 2., 0, 1)
        else:
            vcostmap = np.clip(costmap, 0, norm_factor) / norm_factor
        _mark('normalize_for_rgb')

        if costmap_layer == 'costmap':
            gridmap_cs = (CMAP(vcostmap) * 255).astype(np.int32)
        else:
            gridmap_cs = (SMAP(vcostmap) * 255).astype(np.int32)
        _mark('matplotlib_colormap')

        gridmap_color = (gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2])
        gridmap_color = gridmap_color.view(dtype=np.float32)
        _mark('pack_rgb_float32')

        costmap_msg.layers.append('rgb_viz')
        rgb_layer_msg = Float32MultiArray()
        rgb_layer_msg.layout.dim.append(
            MultiArrayDimension(label='column_index', size=layer_data.shape[0], stride=layer_data.shape[0]))
        rgb_layer_msg.layout.dim.append(
            MultiArrayDimension(label='row_index', size=layer_data.shape[0],
                                stride=layer_data.shape[0] * layer_data.shape[1]))

        rgb_pack_t0 = time.perf_counter()
        rgb_oriented = np.ascontiguousarray(
            gridmap_color[::-1, ::-1], dtype=np.float32)
        rgb_packed = array('f')
        rgb_packed.frombytes(rgb_oriented.tobytes())
        rgb_layer_msg.data = rgb_packed
        rgb_pack_ms = (time.perf_counter() - rgb_pack_t0) * 1000.0
        if _detail_profile:
            _marks.append(('rgb_viz pack_array', rgb_pack_ms))

        costmap_msg.data.append(rgb_layer_msg)
        _mark('append_rgb_viz_layer')

        if _detail_profile:
            total_ms = (time.perf_counter() - _t0) * 1000.0
            sorted_marks = sorted(_marks, key=lambda x: x[1], reverse=True)
            lines = ['[PROFILE costmap_to_gridmap:{}] total={:.2f} ms shape={}'.format(
                costmap_layer, total_ms, costmap.shape)]
            for name, dt_ms in sorted_marks[:10]:
                lines.append('  {:>7.2f} ms  {}'.format(dt_ms, name))
            self.get_logger().info('\n'.join(lines))

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
