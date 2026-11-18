from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('context_adaptation')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            pkg_share,
            'assets',
            'context_configs',
            'CC_rough_vits14_4bags.yaml',
        ),
        description='Path to context config YAML',
    )
    context_node = Node(
        package='context_adaptation',
        executable='dino_costmap_gp_speed_input_output.py',
        name='context_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn'],
        parameters=[{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'cost_topic': '/traversability_cost',
            'odom_topic': '/aft_mapped_to_init',
            'gridmap_topic': '/dino_gridmap',
            'costmap_topic': '/shortrange_costmap',
            'vel_pub_topic': '/controller/target_input',
            'config_file': LaunchConfiguration('config_file'),
        }],
    )

    cost_publisher_node = Node(
        package='context_adaptation',
        executable='cost_publisher.py',
        name='traversability_cost_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn'],
        parameters=[{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'cost_stats_dir': 'wanda_cost_statistics.yaml',
            "imu_topic": "/imu/data",
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        config_file_arg,
        context_node,
        cost_publisher_node,
    ])
