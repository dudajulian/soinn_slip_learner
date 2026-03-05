import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    launch_robot_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('coppelia_ros2_control'),
                'launch',
                'coppelia_control.launch.py',
            ])
        ),
        launch_arguments={
            'controller_name': 'platform_velocity_controller',
            'controller_config_path': 'resources/husky/control.yaml',
            'robot_description_path': 'resources/husky/robot.urdf',
        }.items(),
    )

    elevation_config_dir = get_package_share_directory('soinn_plus_traverse')
    elevation_config_dir = os.path.join(elevation_config_dir, 'config')
    elevation_params = [
        os.path.join(elevation_config_dir, 'zed_create_robot.yaml'),
        os.path.join(elevation_config_dir, 'elevation_map.yaml'),
        os.path.join(elevation_config_dir, 'aslam.yaml'),
        os.path.join(elevation_config_dir, 'postprocessor_pipeline.yaml'),
    ]

    elevation_mapping = Node(
        package='elevation_mapping',
        executable='elevation_mapping',
        name='elevation_mapping',
        output='screen',
        parameters=elevation_params,
    )

    full_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('soinn_slip_learner'),
                'launch',
                'full_system.launch.py',
            ])
        )
    )

    return LaunchDescription([
        launch_robot_control,
        elevation_mapping,
        full_system,
    ])
