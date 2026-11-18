#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import matplotlib.pyplot as plt

MPLIB_CMAP = plt.get_cmap('jet')

DUMMY_FIELD_PREFIX = '__'


def fields_to_dtype(fields, point_step):
    dtype_map = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
        if f.datatype == 0:
            continue
        dtype = np.dtype(dtype_map[f.datatype])
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))
        np_dtype_list.append((f.name, dtype))
        offset += np.dtype(dtype_map[f.datatype]).itemsize * f.count
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
    return np_dtype_list


def pointcloud2_to_array(cloud_msg, squeeze=True):
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
    cloud_arr = np.frombuffer(bytes(cloud_msg.data), np.dtype(dtype_list))
    cloud_arr = cloud_arr[[fname for fname, _type in dtype_list
                           if not fname.startswith(DUMMY_FIELD_PREFIX)]]
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    translation_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    rotation_matrix = np.array([
        [np.cos(yaw)*np.cos(pitch),
         np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll),
         np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll), 0],
        [np.sin(yaw)*np.cos(pitch),
         np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll),
         np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll), 0],
        [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll), 0],
        [0, 0, 0, 1],
    ])
    return np.dot(translation_matrix, rotation_matrix)


def transform_pointcloud(pointcloud, transformation_matrix):
    num_points = pointcloud.shape[0]
    homogeneous_coords = np.hstack((pointcloud, np.ones((num_points, 1))))
    transformed = np.dot(transformation_matrix, homogeneous_coords.T).T
    return transformed[:, :3]


def np_points_to_pointcloud2(points, header, color=True):
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    data = points.astype(dtype).tobytes()

    if color:
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                  for i, n in enumerate('xyzrgba')]
        return PointCloud2(header=header, height=1, width=points.shape[0],
                           is_dense=False, is_bigendian=False, fields=fields,
                           point_step=itemsize*7, row_step=itemsize*7*points.shape[0], data=data)
    if points.shape[1] == 4:
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                  for i, n in enumerate('xyzi')]
        return PointCloud2(header=header, height=1, width=points.shape[0],
                           is_dense=False, is_bigendian=False, fields=fields,
                           point_step=itemsize*4, row_step=itemsize*4*points.shape[0], data=data)
    fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
              for i, n in enumerate('xyz')]
    return PointCloud2(header=header, height=1, width=points.shape[0],
                       is_dense=False, is_bigendian=False, fields=fields,
                       point_step=itemsize*3, row_step=itemsize*3*points.shape[0], data=data)


V2_V1 = create_transformation_matrix(-0.06511276, -0.00034881, 0.21072615, .000146970671, .168982719, -.0103227238)
L_V1 = create_transformation_matrix(-0.14460644, 0.09369309, -0.21868021, 0.01003703, 0.22189415, -0.01221614)


class CloudRepubNode(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')

        self.V1 = np.zeros((3, 4))
        self.V2 = np.zeros((3, 4))
        self.L = np.zeros((3, 4))

        self.pub1 = self.create_publisher(PointCloud2, '/full_cloud', 10)

        self.create_subscription(PointCloud2, '/velodyne_1/velodyne_points', self.pointcloud_callback1, 10)
        self.create_subscription(PointCloud2, '/livox/lidar', self.pointcloud_callback3, 10)

    def _msg_to_pts(self, msg):
        pc = pointcloud2_to_array(msg)
        pts = np.ones((pc.shape[0], 4), dtype=np.float32)
        pts[:, 0] = pc['x']
        pts[:, 1] = pc['y']
        pts[:, 2] = pc['z']
        pts[:, 3] = pc['intensity']
        return pts

    def pointcloud_callback1(self, msg):
        V1 = self._msg_to_pts(msg)
        ids = np.where(np.linalg.norm(V1[:, :3], axis=1) > 2)[0]
        V1 = V1[ids]

        v2 = np.copy(self.V2)
        v2[:, :3] = transform_pointcloud(v2[:, :3], V2_V1)

        l = np.copy(self.L)
        l[:, :3] = transform_pointcloud(l[:, :3], L_V1)

        V1 = np.vstack([V1, v2, l])
        out_msg = np_points_to_pointcloud2(V1, msg.header, color=False)
        self.pub1.publish(out_msg)

    def pointcloud_callback3(self, msg):
        pts = self._msg_to_pts(msg)
        ids = np.where(np.linalg.norm(pts[:, :3], axis=1) > 2)[0]
        self.L = pts[ids]


def main(args=None):
    rclpy.init(args=args)
    node = CloudRepubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
