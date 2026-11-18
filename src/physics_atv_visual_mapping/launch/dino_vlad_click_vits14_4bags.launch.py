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
        'dino_vlad_rough_vits14_252x448_4bags.yaml',
    )

    config_fp_arg = DeclareLaunchArgument(
        'config_fp',
        default_value=default_config_fp,
        description='ViT-S/14 four-bag VLAD click config',
    )

    click_node = Node(
        package='physics_atv_visual_mapping',
        executable='dino_vlad_image_click_node.py',
        name='dino_vlad_click_vits14_4bags',
        output='screen',
        parameters=[{
            'config_fp': LaunchConfiguration('config_fp'),
        }],
    )

    return LaunchDescription([
        config_fp_arg,
        click_node,
    ])
