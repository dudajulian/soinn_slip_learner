import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    demo_share = get_package_share_directory('soislip_demo')
    default_params_file = os.path.join(demo_share, 'config', 'soislip_params.yaml')
    params_file = LaunchConfiguration('params_file')

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the YAML file with node parameters.',
    )

    return LaunchDescription([
        declare_params_file_arg,
        Node(
            package='soislip_core',
            executable='gridmap_feature_extractor_node',
            name='gridmap_feature_extractor_node',
            output='screen',
            parameters=[params_file],
        ),
        # Node(
        #     package='soislip_core',
        #     executable='latent_feature_extractor_node.py',
        #     name='latent_feature_extractor_node',
        #     output='screen',
        #     parameters=['config/feature_extractor.yaml'],
        # ),
        Node(
            package='soislip_core',
            executable='robot_experience_collector_node',
            name='robot_experience_collector_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='soislip_core',
            executable='soinn_training_node.py',
            name='soinn_training_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='soislip_core',
            executable='soinn_prediction_node.py',
            name='soinn_prediction_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='soislip_core',
            executable='slip_prediction_manager',
            name='slip_prediction_manager',
            output='screen',
            parameters=[params_file],
        ),
    ])
