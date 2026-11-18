from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('z_max',         default_value='2.0'),
        # DeclareLaunchArgument('input_topic',   default_value='/cloud_registered_local'),
        DeclareLaunchArgument('input_topic',   default_value='/livox/lidar'),
        DeclareLaunchArgument('output_topic',  default_value='/livox/lidar_filtered'),

        Node(
            package='pcl_z_filter',
            executable='z_filter_node',
            name='pcl_z_filter',
            parameters=[{
                'z_max':         LaunchConfiguration('z_max'),
                'input_topic':   LaunchConfiguration('input_topic'),
                'output_topic':  LaunchConfiguration('output_topic'),
            }],
            output='screen',
        ),
    ])
