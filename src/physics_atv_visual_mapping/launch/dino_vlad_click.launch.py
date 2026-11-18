from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('physics_atv_visual_mapping')

    default_config_fp = os.path.join(
        pkg_share,
        'config',
        'ros',
        'dino_vlad_rough.yaml'
    )

    config_fp_arg = DeclareLaunchArgument(
        'config_fp',
        default_value=default_config_fp,
        description='Config file for DINO VLAD image click node',
    )

    config_fp = LaunchConfiguration('config_fp')

    dino_vlad_click_node = Node(
        package='physics_atv_visual_mapping',
        executable='dino_vlad_image_click_node.py',
        name='dino_vlad_image_click_node',
        output='screen',
        parameters=[{
            'config_fp': config_fp,
        }],
    )

    return LaunchDescription([
        config_fp_arg,
        dino_vlad_click_node,
    ])