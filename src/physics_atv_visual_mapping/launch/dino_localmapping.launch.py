from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('physics_atv_visual_mapping')

    visual_mapping_config_arg = DeclareLaunchArgument(
        'visual_mapping_config',
        default_value= 'dino_vlad_rough_vits14_252x448_4bags.yaml',
        # default_value= 'dino_vlad_rough.yaml',
        description='Config file for visual mapping',
    )
    config_name = LaunchConfiguration('visual_mapping_config')
    reliability_arg = DeclareLaunchArgument('input_reliability', default_value='',
        description='Optional reliable/best_effort override for sensor subscriptions')

    visual_mapping_node = Node(
        package='physics_atv_visual_mapping',
        executable='dino_localmapping.py',
        name='visual_localmapping',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn'],
        parameters=[{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'input_reliability': ParameterValue(LaunchConfiguration('input_reliability'), value_type=str),
            'config_fp': PathJoinSubstitution([
                pkg_share,
                'config',
                'ros',
                config_name,
            ]),
        }],
    )

    lidar_stitching_node = Node(
        package='physics_atv_visual_mapping',
        executable='cloud_repub.py',
        name='lidar_stitching',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn'],
        parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        visual_mapping_config_arg,
        reliability_arg,
        visual_mapping_node,
        lidar_stitching_node,
    ])
