from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description() -> LaunchDescription:
    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        description='Path to the YAML file with node parameters (REQUIRED).',
    )
    declare_tf_topic_arg = DeclareLaunchArgument(
        'tf_topic',
        default_value='/tf',
        description='Topic name to remap /tf to.',
    )
    declare_tf_static_topic_arg = DeclareLaunchArgument(
        'tf_static_topic',
        default_value='/tf_static',
        description='Topic name to remap /tf_static to.',
    )

    params_file = LaunchConfiguration('params_file')
    tf_topic = LaunchConfiguration('tf_topic')
    tf_static_topic = LaunchConfiguration('tf_static_topic')
    params_with_substitutions = ParameterFile(
        param_file=params_file,
        allow_substs=True,
    )
    tf_remaps = [('/tf', tf_topic), ('/tf_static', tf_static_topic)]

    return LaunchDescription([
        declare_params_file_arg,
        declare_tf_topic_arg,
        declare_tf_static_topic_arg,
        Node(
            package='soislip_core',
            executable='gridmap_feature_extractor_node',
            name='gridmap_feature_extractor_node',
            output='screen',
            parameters=[params_with_substitutions],
            remappings=tf_remaps,
        ),
        # Node(
        #     package='soislip_core',
        #     executable='latent_feature_extractor_node.py',
        #     name='latent_feature_extractor_node',
        #     output='screen',
        #     parameters=[params_with_substitutions],
        # ),
        Node(
            package='soislip_core',
            executable='robot_experience_collector_node',
            name='robot_experience_collector_node',
            output='screen',
            parameters=[params_with_substitutions],
            remappings=tf_remaps,
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        Node(
            package='soislip_core',
            executable='soinn_training_node.py',
            name='soinn_training_node',
            output='screen',
            parameters=[params_with_substitutions],
            remappings=tf_remaps,
        ),
        Node(
            package='soislip_core',
            executable='soinn_prediction_node.py',
            name='soinn_prediction_node',
            output='screen',
            parameters=[params_with_substitutions],
            remappings=tf_remaps,
        ),
        Node(
            package='soislip_core',
            executable='slip_prediction_manager',
            name='slip_prediction_manager',
            output='screen',
            parameters=[params_with_substitutions],
            remappings=tf_remaps,
        ),
    ])
