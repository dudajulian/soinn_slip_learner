import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    demo_share = get_package_share_directory('soislip_demo')

    huksy_resources_dir = os.path.join(demo_share, 'resources', 'husky')
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
            'controller_config_path': os.path.join(huksy_resources_dir, 'control.yaml'),
            'robot_description_path': os.path.join(huksy_resources_dir, 'robot.urdf'),       
        }.items(),
    )

    elevation_config_dir = os.path.join(demo_share, 'config', 'elevation_mapping')
    elevation_params = [
        os.path.join(elevation_config_dir, 'zed2_robot.yaml'),
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

    soislip_core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('soislip_core'),
                'launch',
                'soislip.launch.py',
            ])
        ),
        launch_arguments={
            'params_file': os.path.join(demo_share, 'config', 'soislip_params.yaml'),
        }.items(),
    )

    teleop_joy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('teleop_twist_joy'),
                'launch',
                'teleop-launch.py',
            ])
        ),
        launch_arguments={
            'joy_config': 'xbox',
            'publish_stamped_twist': 'true',
            'use_sim_time': 'true',
            'joy_vel': '/platform_velocity_controller/cmd_vel',
        }.items(),
    )

    rviz_config_file = os.path.join(demo_share, 'config', 'rviz', 'custom_rviz2.rviz')
    rviz = Node(
        package= 'rviz2',
        executable= 'rviz2',
        name= 'rviz',
        arguments= ['--display-config', rviz_config_file],
        output= 'screen'
    )

    return LaunchDescription([
        launch_robot_control,
        elevation_mapping,
        soislip_core,
        rviz,
        teleop_joy,
    ])
