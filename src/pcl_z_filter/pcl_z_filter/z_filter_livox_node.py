import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from livox_interfaces.msg import CustomMsg

from .z_filter_node import pcl2_to_array, array_to_pcl2


def custom_points_to_pointcloud2(points, header) -> PointCloud2:
    """Convert a filtered list of livox_interfaces/CustomPoint into a
    PointCloud2 with fields [x, y, z, intensity] (all float32)."""
    n = len(points)
    arr = np.zeros(n, dtype=[
        ('x', np.float32), ('y', np.float32),
        ('z', np.float32), ('intensity', np.float32),
    ])
    if n > 0:
        arr['x'] = [p.x for p in points]
        arr['y'] = [p.y for p in points]
        arr['z'] = [p.z for p in points]
        arr['intensity'] = [float(p.reflectivity) for p in points]

    fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
    return array_to_pcl2(arr, header, point_step, fields)


class ZFilterLivoxNode(Node):
    """
    Z-axis point filter for the Livox driver topic `/livox/lidar`.

    `ros2 topic type /livox/lidar` can report either of:
        livox_interfaces/msg/CustomMsg   (native Livox format, xfer_format=1)
        sensor_msgs/msg/PointCloud2      (xfer_format=0, or after a converter)

    Only one of these is the actual type of the topic at any given time — set
    the `input_format` parameter to match it ('custom' or 'pointcloud2').
    The filtered cloud is always republished as sensor_msgs/msg/PointCloud2,
    regardless of the input format.
    """

    def __init__(self):
        super().__init__('z_filter_livox')

        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/livox/lidar_filtered')
        self.declare_parameter('input_format', 'custom')  # 'custom' | 'pointcloud2'

        input_topic   = self.get_parameter('input_topic').value
        output_topic  = self.get_parameter('output_topic').value
        input_format  = self.get_parameter('input_format').value.lower()

        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        if input_format == 'custom':
            self.sub = self.create_subscription(
                CustomMsg, input_topic, self._callback_custom, 10)
        elif input_format == 'pointcloud2':
            self.sub = self.create_subscription(
                PointCloud2, input_topic, self._callback_pointcloud2, 10)
        else:
            raise ValueError(
                f"input_format must be 'custom' or 'pointcloud2', got '{input_format}'")

        self.get_logger().info(
            f'ZFilterLivoxNode ready  |  {input_topic} ({input_format}) -> '
            f'{output_topic} (PointCloud2)  |  '
            f'z_max = {self.get_parameter("z_max").value:.3f} m'
        )

    def _callback_custom(self, msg: CustomMsg):
        z_max = self.get_parameter('z_max').value

        filtered_points = [p for p in msg.points if p.z <= z_max]

        out = custom_points_to_pointcloud2(filtered_points, msg.header)
        self.pub.publish(out)

    def _callback_pointcloud2(self, msg: PointCloud2):
        z_max = self.get_parameter('z_max').value

        arr = pcl2_to_array(msg)
        mask = arr['z'] <= z_max
        filtered = arr[mask]

        out = array_to_pcl2(filtered, msg.header, msg.point_step, msg.fields)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ZFilterLivoxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
