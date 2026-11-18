import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


DTYPE_MAP = {
    PointField.INT8:    np.dtype('int8'),
    PointField.UINT8:   np.dtype('uint8'),
    PointField.INT16:   np.dtype('int16'),
    PointField.UINT16:  np.dtype('uint16'),
    PointField.INT32:   np.dtype('int32'),
    PointField.UINT32:  np.dtype('uint32'),
    PointField.FLOAT32: np.dtype('float32'),
    PointField.FLOAT64: np.dtype('float64'),
}


def pcl2_to_array(msg: PointCloud2) -> np.ndarray:
    dtype = np.dtype({
        'names':   [f.name for f in msg.fields],
        'formats': [DTYPE_MAP[f.datatype] for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    return np.frombuffer(bytes(msg.data), dtype=dtype)


def array_to_pcl2(arr: np.ndarray, header: Header, point_step: int,
                  fields: list) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(arr)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * len(arr)
    msg.is_dense = False
    msg.data = arr.tobytes()
    return msg


class ZFilterNode(Node):
    def __init__(self):
        super().__init__('pcl_z_filter')

        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/livox/lidar_filtered')

        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        self.get_logger().info(
            f'ZFilterNode ready  |  {input_topic} -> {output_topic}  |  '
            f'z_max = {self.get_parameter("z_max").value:.3f} m'
        )

    def callback(self, msg: PointCloud2):
        z_max = self.get_parameter('z_max').value

        arr = pcl2_to_array(msg)
        mask = arr['z'] <= z_max
        filtered = arr[mask]

        out = array_to_pcl2(filtered, msg.header, msg.point_step, msg.fields)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ZFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
