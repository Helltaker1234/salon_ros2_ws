import numpy as np
import torch
import cv2
from functools import reduce
import time
import struct
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header


DUMMY_FIELD_PREFIX = '__'

type_mappings = [
    (PointField.INT8, np.dtype('int8')),
    (PointField.UINT8, np.dtype('uint8')),
    (PointField.INT16, np.dtype('int16')),
    (PointField.UINT16, np.dtype('uint16')),
    (PointField.INT32, np.dtype('int32')),
    (PointField.UINT32, np.dtype('uint32')),
    (PointField.FLOAT32, np.dtype('float32')),
    (PointField.FLOAT64, np.dtype('float64')),
]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)


def remove_invalid(lidar_points):
    if lidar_points.shape[1] >= 4:
        lidar_points = lidar_points[:, :3]
    lidar_points_norm = torch.norm(lidar_points, dim=1)
    return lidar_points[lidar_points_norm > 1e-5]


def get_intrinsics(intrinsics_matrix, tf_in_optical=True):
    if tf_in_optical:
        I = torch.eye(4)
        I[:3, :3] = intrinsics_matrix
        return I
    else:
        T_p_i = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
        intrinsics_matrix = torch.matmul(T_p_i, intrinsics_matrix)
        I = torch.ones(4)
        I[:3, :3] = intrinsics_matrix
        return I


def get_extrinsics(extrinsics_matrix, tf_in_optical=True):
    return extrinsics_matrix


def obtain_projection_matrix(intrinsics, extrinsics):
    P = torch.matmul(intrinsics, extrinsics)
    return P[:-1, :]


def get_pixel_from_3D_source(lidar_points, P, return_depth=False):
    ones = torch.ones((lidar_points.shape[0], 1)).to(lidar_points.device)
    homo_3D = torch.cat((lidar_points, ones), dim=1)
    homo_pixel = (torch.matmul(P, homo_3D.T)).T
    homo_pixel_norm = homo_pixel / homo_pixel[:, [2]]
    pixels = homo_pixel_norm[:, :-1]
    return (pixels, homo_pixel[:, 2]) if return_depth else pixels


def get_points_and_pixels_in_frame(
        lidar_points,
        pixel_coordinates,
        image_height,
        image_width,
        height_thresh=10.0,
        return_lidar_points=True,
        camera_depth=None):
    pixel_coords_x = pixel_coordinates[:, 0]
    pixel_coords_y = pixel_coordinates[:, 1]
    lidar_points_x = lidar_points[:, 0]
    lidar_points_z = lidar_points[:, 2]

    cond1 = pixel_coords_x >= 0
    cond2 = pixel_coords_x < image_width
    cond3 = pixel_coords_y >= 0
    cond4 = pixel_coords_y < image_height
    cond5 = lidar_points_x > 0
    cond6 = lidar_points_z < height_thresh

    ind_in_frame = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    if camera_depth is not None:
        ind_in_frame &= torch.isfinite(camera_depth) & (camera_depth > 0.0)
    pixels_in_frame = pixel_coordinates[ind_in_frame, :].long()
    if not return_lidar_points:
        return pixels_in_frame, ind_in_frame

    lidar_points_in_frame = lidar_points[ind_in_frame, :]
    return lidar_points_in_frame, pixels_in_frame, ind_in_frame


def get_rgb_from_pixel_coords(image, pixel_coords):
    return image[pixel_coords[:, 1], pixel_coords[:, 0]]


def xyz_array_to_point_cloud_msg(points, frame, timestamp=None, rgb_values=None):
    header = Header()
    header.frame_id = frame
    if timestamp is not None:
        header.stamp = timestamp

    msg = PointCloud2()
    msg.header = header
    if len(points.shape) == 3:
        msg.width = points.shape[0]
        msg.height = points.shape[1]
    else:
        msg.width = points.shape[0]
        msg.height = 1
    msg.is_bigendian = False
    msg.is_dense = False

    if rgb_values is None:
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        xyz = points.astype(np.float32)
        msg.data = xyz.tobytes()
    else:
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width

        xyzcolor = np.zeros((points.shape[0], 1), dtype={
            'names': ('x', 'y', 'z', 'rgba'),
            'formats': ('f4', 'f4', 'f4', 'u4'),
        })
        xyzcolor['x'] = points[:, 0].reshape((-1, 1))
        xyzcolor['y'] = points[:, 1].reshape((-1, 1))
        xyzcolor['z'] = points[:, 2].reshape((-1, 1))
        color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
        color_rgba[:, :3] = rgb_values[:, :3]
        xyzcolor['rgba'] = color_rgba.view('uint32')
        msg.data = xyzcolor.tobytes()

    return msg


def create_point_cloud(points, parent_frame='lidar', colors=None):
    return xyz_array_to_point_cloud_msg(points, frame=parent_frame, timestamp=None, rgb_values=colors)


def rotation(axis, angle):
    angle = angle * np.pi / 180
    if axis in ('x', 'X'):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis in ('y', 'Y'):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
