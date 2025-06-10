#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """
    Setup function that gets called with the launch context.
    This allows us to evaluate LaunchConfiguration values.
    """
    # Get launch configuration values
    num_robots = int(LaunchConfiguration('num_robots').perform(context))
    ns_prefix = LaunchConfiguration('ns_prefix').perform(context)
    params_file = LaunchConfiguration('params_file')
    learning_params_file = LaunchConfiguration('learning_params_file')
    model_state_dict = LaunchConfiguration('model_state_dict')
    remap_robot_poses = LaunchConfiguration('remap_robot_poses')
    remap_sim_get_parameters = LaunchConfiguration('remap_sim_get_parameters')
    remap_get_world_map = LaunchConfiguration('remap_get_world_map')
    remap_get_system_info = LaunchConfiguration('remap_get_system_info')
    
    # Create list to hold LPAC nodes
    lpac_nodes = []
    
    # Generate nodes for each robot
    for i in range(1, num_robots + 1):
        # Create LPAC node for each robot
        lpac_node = Node(
            package='async_pac_gnn_py',
            executable='lpac_l1',
            namespace=f'{ns_prefix}{i}',
            name='lpac_node',
            parameters=[{
                'params_file': params_file,
                'learning_params_file': learning_params_file,
                'model_state_dict': model_state_dict
            }],
            remappings=[
                ('robot_poses', remap_robot_poses),
                ('sim_get_parameters', remap_sim_get_parameters),
                ('get_world_map', remap_get_world_map),
                ('get_system_info', remap_get_system_info)
            ]
        )
        
        lpac_nodes.append(lpac_node)
    
    return lpac_nodes


def generate_launch_description():
    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        description='Number of robots to spawn LPAC nodes for'
    )
    
    ns_prefix_arg = DeclareLaunchArgument(
        'ns_prefix',
        default_value='fr',
        description='Namespace prefix for robots (default: fr)'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/workspace/pt/models_256/coverage_control_params.toml',
        description='Path to parameters file'
    )
    
    learning_params_file_arg = DeclareLaunchArgument(
        'learning_params_file',
        default_value='/workspace/pt/models_256/learning_params.toml',
        description='Path to learning parameters file'
    )
    
    model_state_dict_arg = DeclareLaunchArgument(
        'model_state_dict',
        default_value='/workspace/pt/models_256/model_k3_1024_256_state_dict.pt',
        description='Path to model state dictionary file'
    )
    
    remap_robot_poses_arg = DeclareLaunchArgument(
        'remap_robot_poses',
        default_value='/sim/all_robot_sim_poses',
        description='Remapping target for robot_poses topic'
    )
    
    remap_sim_get_parameters_arg = DeclareLaunchArgument(
        'remap_sim_get_parameters',
        default_value='/sim/sim_centralized/get_parameters',
        description='Remapping target for sim_get_parameters service'
    )
    
    remap_get_world_map_arg = DeclareLaunchArgument(
        'remap_get_world_map',
        default_value='/sim/get_world_map',
        description='Remapping target for get_world_map service'
    )
    
    remap_get_system_info_arg = DeclareLaunchArgument(
        'remap_get_system_info',
        default_value='/sim/get_system_info',
        description='Remapping target for get_system_info service'
    )
    
    # Use OpaqueFunction to handle dynamic node creation
    opaque_function = OpaqueFunction(function=launch_setup)
    
    return LaunchDescription([
        num_robots_arg,
        ns_prefix_arg,
        params_file_arg,
        learning_params_file_arg,
        model_state_dict_arg,
        remap_robot_poses_arg,
        remap_sim_get_parameters_arg,
        remap_get_world_map_arg,
        remap_get_system_info_arg,
        opaque_function
    ])


################################################################################
# Basic usage with required argument
# ros2 launch async_pac_gnn_py lpac_l1_launch.py num_robots:=3

# With custom parameters
# ros2 launch async_pac_gnn_py lpac_l1_launch.py num_robots:=5 ns_prefix:=robot params_file:=/path/to/custom/params.toml

# With custom remappings
# ros2 launch async_pac_gnn_py lpac_l1_launch.py num_robots:=2 remap_robot_poses:=/custom/robot_poses
################################################################################
