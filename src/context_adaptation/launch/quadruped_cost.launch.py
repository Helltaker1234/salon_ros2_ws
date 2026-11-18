from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    state_topic_arg = DeclareLaunchArgument(
        'state_topic', default_value='/lio/imu/odom')
    command_topic_arg = DeclareLaunchArgument(
        'command_topic', default_value='/cmd_vel')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    state_type_arg = DeclareLaunchArgument('state_type', default_value='Odometry')
    command_type_arg = DeclareLaunchArgument('command_type', default_value='Twist')
    velocity_frame_arg = DeclareLaunchArgument('velocity_frame', default_value='world')
    base_pitch_arg = DeclareLaunchArgument('base_pitch_deg', default_value='-21.0')

    cost_publisher_node = Node(
        package='context_adaptation',
        executable='quadruped_cost_publisher.py',
        name='quadruped_cost_publisher',
        output='screen',
        parameters=[{
            'state_topic': LaunchConfiguration('state_topic'),
            'command_topic': LaunchConfiguration('command_topic'),
            'state_type': LaunchConfiguration('state_type'),
            'command_type': LaunchConfiguration('command_type'),
            'velocity_frame': LaunchConfiguration('velocity_frame'),
            'base_pitch_deg': ParameterValue(LaunchConfiguration('base_pitch_deg'), value_type=float),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
    )

    return LaunchDescription([
        state_topic_arg,
        command_topic_arg,
        use_sim_time_arg,
        state_type_arg,
        command_type_arg,
        velocity_frame_arg,
        base_pitch_arg,
        cost_publisher_node,
    ])
