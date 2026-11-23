from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
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
            'odom_topic': '/lio/odom',
            'gridmap_topic': '/dino_gridmap',
            'costmap_topic': '/shortrange_costmap',
            'vel_pub_topic': '/controller/target_input',
            'config_file': LaunchConfiguration('config_file'),
        }],
    )

    cost_publisher_node = Node(
        package='context_adaptation',
        executable='cost_publisher.py',
        condition=LaunchConfigurationEquals('mode', 'vehicle'),
        name='traversability_cost_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn'],
        parameters=[{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'cost_stats_dir': 'wanda_cost_statistics.yaml',
            "imu_topic": "/imu/data",
        }],
    )

    quadruped_cost_node = Node(
        package='context_adaptation',
        executable='quadruped_cost_publisher.py',
        condition=LaunchConfigurationEquals('mode', 'quad'),
        name='quadruped_cost_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'state_topic': LaunchConfiguration('state_topic'),
            'command_topic': LaunchConfiguration('command_topic'),
            'state_type': LaunchConfiguration('state_type'),
            'command_type': LaunchConfiguration('command_type'),
            'velocity_frame': LaunchConfiguration('velocity_frame'),
            'base_pitch_deg': ParameterValue(LaunchConfiguration('base_pitch_deg'), value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode', default_value='vehicle', choices=['quad', 'vehicle'],
            description='Cost source: quad velocity tracking or vehicle IMU'),
        DeclareLaunchArgument('state_topic', default_value='/lio/imu/odom',
                              description='Quad state topic'),
        DeclareLaunchArgument('command_topic', default_value='/go2/cmd_vel',
                              description='Quad velocity command topic'),
        DeclareLaunchArgument('state_type', default_value='Odometry',
                              choices=['Odometry', 'RobotState']),
        DeclareLaunchArgument('command_type', default_value='Twist',
                              choices=['Twist', 'TwistStamped']),
        DeclareLaunchArgument('velocity_frame', default_value='world',
                              choices=['world', 'imu']),
        DeclareLaunchArgument('base_pitch_deg', default_value='-21.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        config_file_arg,
        context_node,
        cost_publisher_node,
        quadruped_cost_node,
    ])
