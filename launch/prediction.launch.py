from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='soinn_slip_learner',
            executable='gridmap_feature_extractor_node',
            name='gridmap_feature_extractor_node',
            output='screen',
            parameters=['config/soinn_params.yaml'],
        ),
        Node(
            package='soinn_slip_learner',
            executable='latent_feature_extractor_node.py',
            name='latent_feature_extractor_node',
            output='screen',
            parameters=['config/feature_extractor.yaml'],
        ),
        Node(
            package='soinn_slip_learner',
            executable='soinn_prediction_node.py',
            name='soinn_prediction_node',
            output='screen',
            parameters=['config/soinn_params.yaml'],
        ),
        Node(
            package='soinn_slip_learner',
            executable='slip_prediction_manager',
            name='slip_prediction_manager',
            output='screen',
            parameters=['config/soinn_params.yaml'],
        ),
    ])
