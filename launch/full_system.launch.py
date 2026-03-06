import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    param_file = os.path.join(
        get_package_share_directory('soinn_slip_learner'),
        'config',
        'soinn_params.yaml'
    )
    return LaunchDescription([
        Node(
            package='soinn_slip_learner',
            executable='gridmap_feature_extractor_node',
            name='gridmap_feature_extractor_node',
            output='screen',
            parameters=[param_file],
        ),
        Node(
            package='soinn_slip_learner',
            executable='robot_experience_collector_node',
            name='robot_experience_collector_node',
            output='screen',
            parameters=[param_file],
        ),
        # Node(
        #     package='soinn_slip_learner',
        #     executable='latent_feature_extractor_node.py',
        #     name='latent_feature_extractor_node',
        #     output='screen',
        #     parameters=['config/feature_extractor.yaml'],
        # ),
        Node(
            package='soinn_slip_learner',
            executable='soinn_training_node.py',
            name='soinn_training_node',
            output='screen',
            parameters=[param_file],
        ),
        Node(
            package='soinn_slip_learner',
            executable='soinn_prediction_node.py',
            name='soinn_prediction_node',
            output='screen',
            parameters=[param_file],
        ),
        Node(
            package='soinn_slip_learner',
            executable='slip_prediction_manager',
            name='slip_prediction_manager',
            output='screen',
            parameters=[param_file],
        ),
    ])
