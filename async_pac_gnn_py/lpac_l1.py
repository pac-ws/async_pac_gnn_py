import os
import sys
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import TwistStamped
from async_pac_gnn_interfaces.srv import WorldMap
from rcl_interfaces.srv import GetParameters

from coverage_control import Point2
from coverage_control import PointVector
from coverage_control import Parameters
from coverage_control import CoverageSystem
from coverage_control import IOUtils
from coverage_control import WorldIDF
from coverage_control import CoverageEnvUtils
from coverage_control import IOUtils
import coverage_control.nn as cc_nn

class LPAC_Controller:

    def __init__(self, config: dict, params: Parameters, env: CoverageSystem):
        self.config = config
        self.params = params
        self.name = self.config["Name"]
        self.use_cnn = self.config["UseCNN"]
        self.use_comm_map = self.config["UseCommMap"]
        self.cnn_map_size = self.config["CNNMapSize"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")

        if "ModelFile" in self.config:
            self.model_file = IOUtils.sanitize_path(self.config["ModelFile"])
            self.model = torch.load(self.model_file).to(self.device)
        else:  # Load from ModelStateDict
            self.learning_params_file = IOUtils.sanitize_path(
                    self.config["LearningParams"]
                    )
            self.learning_params = IOUtils.load_toml(self.learning_params_file)
            self.model = cc_nn.LPAC(self.learning_params).to(self.device)
            self.model.load_model(IOUtils.sanitize_path(self.config["ModelStateDict"]))

        self.actions_mean = self.model.actions_mean.to(self.device)
        self.actions_std = self.model.actions_std.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        # self.model = torch.compile(self.model, dynamic=True)

    def step(self, env):
        pyg_data = CoverageEnvUtils.get_torch_geometric_data(
                env, self.params, True, self.use_comm_map, self.cnn_map_size
                ).to(self.device)
        with torch.no_grad():
            actions = self.model(pyg_data)
            actions = actions * self.actions_std + self.actions_mean
        point_vector_actions = PointVector(actions.cpu().numpy())
        return point_vector_actions
    # env.StepActions(point_vector_actions)

        # return env.GetObjectiveValue(), False

class LPAC(Node):
    def __init__(self):
        super().__init__('lpac_coveragecontrol')

        # self.declare_parameter('namespaces_of_robots', rclpy.Parameter.Type.STRING_ARRAY)
        # self.namespaces_of_robots = self.get_parameter('namespaces_of_robots').get_parameter_value().string_array_value

        # self.namespaces_of_robots_client = self.create_client(NamespacesRobots, '/sim/namespaces_robots')
        self.sim_parameters_client = self.create_client(GetParameters, '/sim/sim_centralized/get_parameters')
        while not self.sim_parameters_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('sim parameters service not available, waiting again...')

        if not self.sim_parameters_client.service_is_ready():
            self.get_logger().error('Service not available')
            return

        request = GetParameters.Request()
        request.names = ['namespaces_of_robots', 'env_scale_factor', 'vel_scale_factor']
        future = self.sim_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.namespaces_of_robots = future.result().values[0].string_array_value
            self.env_scale_factor = future.result().values[1].double_value
            self.vel_scale_factor = future.result().values[2].double_value
        else:
            self.get_logger().error('Service call failed')
        self.get_logger().info(f'Namespaces of robots: {self.namespaces_of_robots}')

        self.declare_parameter('buffer_size', 10)
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value

        # self.declare_parameter('vel_scale_factor', 1.0)
        # self.vel_scale_factor = self.get_parameter('vel_scale_factor').get_parameter_value().double_value
        self.get_logger().info(f'Velocity Scale factor: {self.vel_scale_factor}')

        self.declare_parameter('params_file', '')
        self.params_file = self.get_parameter('params_file').get_parameter_value().string_value
        self.cc_parameters = Parameters(self.params_file)
        self.cc_env = None

        self.status_pac = 2
        self.pac_status_subscription = self.create_subscription(
                Int32,
                '/pac_gcs/status_pac',
                self.pac_status_callback,
                10)

        while self.status_pac:
            self.get_logger().info(f'Waiting for status_pac to be ready. Current status: {self.status_pac}')
            rclpy.spin_once(self)

        # Log status_pac
        self.get_logger().info(f'status_pac: {self.status_pac}')
        self.robot_poses = PointVector()
        for i in range(len(self.namespaces_of_robots)):
            self.robot_poses.append(Point2(0, 0))

        self.poses_subscription = self.create_subscription(
                PoseArray,
                'robot_poses',
                self.poses_callback,
                10)

        while self.robot_poses[0][0] == 0 and self.robot_poses[0][1] == 0:
            self.get_logger().info('Waiting for robot poses')
            rclpy.spin_once(self)

        self.get_logger().info('Robot poses received')

        # Get world map from service call to /sim/sim_centralized/get_world_map
        self.sim_world_map_client = self.create_client(WorldMap, '/sim/get_world_map')
        while not self.sim_world_map_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('sim get_world_map service not available, waiting again...')
        if not self.sim_world_map_client.service_is_ready():
            self.get_logger().error('Service not available')
            return

        request = WorldMap.Request()
        request.map_size = self.cc_parameters.pWorldMapSize
        future = self.sim_world_map_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            sim_map = future.result().map
            self.get_logger().info('Received world map')
        else:
            self.get_logger().error('Service call failed')

        np_map = self.float32_multiarray_to_numpy(sim_map)
        self.cc_parameters.pNumRobots = len(self.namespaces_of_robots)
        self.world_idf = WorldIDF(self.cc_parameters, np_map)
        self.cc_env = CoverageSystem(self.cc_parameters, self.world_idf, self.robot_poses)
        # self.world_map = self.cc_env.GetWorldMapMutable()
        # self.world_map[:, :] = np_map

        cmd_vel_topic = 'cmd_vel'
        self.ns = self.get_namespace()
        # Remove the leading '/'
        self.ns = self.ns[1:]
        self.get_logger().info(f'Namespace: {self.ns}')
        # Identify the index of the robot in the namespace
        for i in range(len(self.namespaces_of_robots)):
            if self.ns == self.namespaces_of_robots[i]:
                self.robot_index = i
                break
            else:
                self.robot_index = -1
        if self.robot_index == -1:
            self.get_logger().error('Robot index not found in the namespace')
            rclpy.shutdown()

        self.publisher_cmd_vel = self.create_publisher(TwistStamped, f'{cmd_vel_topic}', 10)

        timer_period = self.cc_parameters.pTimeStep
        self.lpac_step_timer = self.create_timer(timer_period, self.lpac_step_callback)

        self.robot_controller_params = {}
        self.robot_controller_params["Name"] = "lpac"
        self.robot_controller_params["UseCNN"] = True
        self.robot_controller_params["UseCommMap"] = True
        self.robot_controller_params["CNNMapSize"] = 32
        self.declare_parameter('model_state_dict', '')
        model_state_dict = self.get_parameter('model_state_dict').get_parameter_value().string_value
        # Check if file exists
        if os.path.isfile(model_state_dict):
            self.robot_controller_params["ModelStateDict"] = model_state_dict
        else:
            self.get_logger().error('ModelStateDict file does not exist or is not provided')

        self.declare_parameter('learning_params_file', '')
        learning_params_file = self.get_parameter('learning_params_file').get_parameter_value().string_value
        # Check if file exists
        if os.path.isfile(learning_params_file):
            self.robot_controller_params["LearningParams"] = learning_params_file
        else:
            self.get_logger().error('LearningParams file does not exist or is not provided')


        self.controller = LPAC_Controller(
                self.robot_controller_params, self.cc_parameters, self.cc_env
                )

    def pac_status_callback(self, msg):
        self.status_pac = msg.data

    def float32_multiarray_to_numpy(self, msg: Float32MultiArray) -> np.ndarray:
        # Get the row and column sizes from the layout dimensions
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size

        # Map the flat data into a NumPy array with the specified shape (row-major by default)
        matrix = np.array(msg.data, dtype=np.float32).reshape((rows, cols))

        return matrix

    def poses_callback(self, msg):
        for i in range(self.cc_parameters.pNumRobots):
            self.robot_poses[i][0] = msg.poses[i].position.x
            self.robot_poses[i][1] = msg.poses[i].position.y
            if self.cc_env is not None:
                self.cc_env.SetGlobalRobotPosition(i, self.robot_poses[i])

    def lpac_step_callback(self):
        if self.status_pac == 2:
            return
        if self.cc_env is not None:
            actions = self.controller.step(self.cc_env)
            self_action = actions[self.robot_index]
            if self.status_pac == 1:
                self_action[0] = 0
                self_action[1] = 0
            twist_msg = TwistStamped()
            t = self.get_clock().now()
            twist_msg.header.stamp = t.to_msg()
            twist_msg.header.frame_id = self.ns
            twist_msg.twist.linear.x = self_action[0] * self.vel_scale_factor
            twist_msg.twist.linear.y = self_action[1] * self.vel_scale_factor
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0
            self.publisher_cmd_vel.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    lpac_node = LPAC()
    rclpy.spin(lpac_node)
    lpac_node.destroy_node()
    rclpy.shutdown()
