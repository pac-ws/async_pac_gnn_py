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
    ns = LaunchConfiguration('namespace').perform(context)
    params_file = LaunchConfiguration('params_file')
    learning_params_file = LaunchConfiguration('learning_params_file')
    model_state_dict = LaunchConfiguration('model_state_dict')
    remap_robot_poses = LaunchConfiguration('remap_robot_poses')
    remap_sim_get_parameters = LaunchConfiguration('remap_sim_get_parameters')
    remap_get_world_map = LaunchConfiguration('remap_get_world_map')
    remap_get_system_info = LaunchConfiguration('remap_get_system_info')
    
    # Create list to hold LPAC nodes
    lpac_nodes = []
    
    lpac_node = Node(
        package='async_pac_gnn_py',
        executable='lpac',
        namespace=f'{ns}',
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
    
    ns_arg = DeclareLaunchArgument(
        'namespace',
        default_value='pac',
        description='Namespace for robot (default: fr)'
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
        ns_arg,
        params_file_arg,
        learning_params_file_arg,
        model_state_dict_arg,
        remap_robot_poses_arg,
        remap_sim_get_parameters_arg,
        remap_get_world_map_arg,
        remap_get_system_info_arg,
        opaque_function
    ])
