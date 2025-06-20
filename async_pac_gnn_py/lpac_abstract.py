import os
from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

from async_pac_gnn_interfaces.srv import SystemInfo
from async_pac_gnn_interfaces.srv import UpdateWorldFile
from async_pac_gnn_interfaces.srv import NamespacesRobots
import coverage_control

class LPACAbstract(Node, ABC):
    def __init__(self, node_name: str, is_solo: bool = False):
        super().__init__(node_name)

        self.is_solo = is_solo
        self._ns = self.get_namespace()
        self._ns = self._ns[1:] if self._ns.startswith('/') else self._ns  # Remove leading '/'
        self._ns_index = None
        self._ns_robots = None

        self._num_robots = 1 # Will get update in Initialize

        self._cmd_vel_topic = 'cmd_vel'
        self._cmd_vel_publishers = []

        self._qos_profile = self._initialize_qos_profile()

        self._initialize_parameters()
        self._cc_parameters = coverage_control.Parameters(self.params_file)

        self._status_pac = 2
        self._status_topic = '/pac_gcs/status_pac'
        self._pac_status_subscription = None

        self._vel_scale = 1.0
        self._env_scale = 1.0
        self._idf_path = None
        self._idf_file = None

        self._sim_poses = None
        self._cc_env = None
        self._controller_params = None
        self._update_world_file_srv = None

        self._pac_status_initialized = False
        self._sim_system_info_client = self.create_client(SystemInfo, 'get_system_info')
        self._system_info_done = False
        self._system_info_waiting = False
        self._base_initialized = False
        
        self._namespaces_client = self.create_client(NamespacesRobots, '/sim/get_namespaces_robots')
        self._namespaces_done = False
        self._namespaces_waiting = False

    def InitializeBase(self):
        if not self._pac_status_initialized:
            self._initialize_pac_status_subscriber()
            self._pac_status_initialized = True
        if self._status_pac != 0:
            return

        if not self._system_info_done:
            if self._system_info_waiting:
                return
            if not self._sim_system_info_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn('sim get_system_info service not available, waiting again...')
                return
            request = SystemInfo.Request()
            request.name = self._ns
            future = self._sim_system_info_client.call_async(request)
            future.add_done_callback(self._system_info_callback)
            self._system_info_waiting = True
            return

        if not self._namespaces_done:
            if self._namespaces_waiting:
                return
            if not self._namespaces_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn('get_namespaces_robots service not available, waiting again...')
                return
            request = NamespacesRobots.Request()
            future = self._namespaces_client.call_async(request)
            future.add_done_callback(self._namespaces_callback)
            self._namespaces_waiting = True
            return

        if self._idf_file is None:
            raise RuntimeError('System info retrieval failed')
        if self._ns_robots is None:
            raise RuntimeError('Robot namespaces retrieval failed')
        
        self.get_logger().info(f'System info and namespaces received')

        if not self.is_solo:
            self._num_robots = len(self._ns_robots)

        self._cc_parameters.pNumRobots = self._num_robots

        self._idf_path = '/workspace/' + self._idf_file

        self._controller_params = self._get_controller_params()

        self._update_world_file_srv = self.create_service(
                UpdateWorldFile, 
                'update_world_file',
                self._update_world_file_callback
                )
        self._base_initialized = True

    def _create_cc_env(self, idf_path: str, robot_poses: coverage_control.PointVector):
        if not os.path.isfile(idf_path):
            return False
        try:
            self._cc_env = None
            self._idf_path = idf_path
            self._world_idf = coverage_control.WorldIDF(self._cc_parameters, self._idf_path)
            self._cc_env = coverage_control.CoverageSystem(
                    self._cc_parameters,
                    self._world_idf,
                    robot_poses)
        except Exception as e:
            self.get_logger().error(f'Failed to create CoverageSystem: {e}')
            return False
        return True

    def _update_world_file_callback(self, request, response):
        if not request.file:
            self.get_logger().error('Received empty world file request')
            response.success = False
            response.message = f'[{self._ns}] world file empty'
            return response
        idf_path = '/workspace/' + request.file
        if not os.path.isfile(idf_path):
            self.get_logger().error(f'World file does not exist: {idf_path}')
            response.success = False
            response.message = f'[{self._ns}] world file does not exist: {idf_path}'
            return response
        self._idf_path = idf_path
        if self._cc_env is None or self._sim_poses is None:
            response.success = False
            response.message = f'[{self._ns}] Uninitialized system'
            return response
        if not self._create_cc_env(idf_path, self._sim_poses):
            self.get_logger().error('Failed to create CoverageSystem with new world file')
            response.success = False
            response.message = f'[{self._ns}] Update failed: {idf_path}'
            return response
        response.success = True
        self.get_logger().info(f'CoverageSystem created with new world file: {idf_path}')
        return response

    def _get_controller_params(self):
        """Get controller parameters with proper validation."""
        params = {
                "UseCNN": True,
                "UseCommMap": True,
                "CNNMapSize": 32
                }

        # Model state dict parameter
        self.declare_parameter('model_state_dict', '')
        model_state_dict = self.get_parameter('model_state_dict').get_parameter_value().string_value

        if not model_state_dict:
            raise RuntimeError('model_state_dict parameter is required')

        if not os.path.isfile(model_state_dict):
            raise RuntimeError(f'ModelStateDict file does not exist: {model_state_dict}')

        params["ModelStateDict"] = model_state_dict

        # Learning params file parameter
        self.declare_parameter('learning_params_file', '')
        learning_params_file = self.get_parameter('learning_params_file').get_parameter_value().string_value

        if not learning_params_file:
            raise RuntimeError(f'learning_params_file parameter is required but not provided')

        if not os.path.isfile(learning_params_file):
            raise RuntimeError(f'LearningParams file does not exist: {learning_params_file}')

        params["LearningParams"] = learning_params_file

        self.get_logger().info('Controller parameters loaded successfully')
        return params

    def _create_cmd_vel_publishers(self, is_solo = True):
        """Create command velocity publisher."""
        if is_solo:
            topic_name = f'/{self._ns}/{self._cmd_vel_topic}' if self._ns else self._cmd_vel_topic
            publisher = self.create_publisher(
                    TwistStamped,
                    topic_name,
                    self._qos_profile
                    )
            self._cmd_vel_publishers.append(publisher)
        if not is_solo:
            for ns in self._ns_robots:
                topic_name = f'/{ns}/{self._cmd_vel_topic}'
                publisher = self.create_publisher(
                        TwistStamped,
                        topic_name,
                        self._qos_profile
                        )
                self._cmd_vel_publishers.append(publisher)

    def _get_namespace_index(self):
        """Find the index of this robot in the namespace list."""
        self._ns_index = None
        for i, robot_ns in enumerate(self._ns_robots):
            if self._ns == robot_ns:
                self._ns_index = i
                break

        if self._ns_index is None:
            raise RuntimeError(
                    f'Robot namespace "{self._ns}" not found in robot list: {self._ns_robots}'
                    )

        self.get_logger().info(f'Robot index: {self._ns_index} (namespace: {self._ns})')

    def _initialize_qos_profile(self):
        """Initialize QoS profile for subscriptions and publishers."""
        return rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=5
                )

    def _initialize_parameters(self):
        """Initialize and validate required parameters."""
        self.declare_parameter('params_file', '')
        self.params_file = self.get_parameter('params_file').get_parameter_value().string_value

        if not self.params_file:
            raise RuntimeError('params_file parameter is required')

        if not os.path.isfile(self.params_file):
            raise RuntimeError(f'Parameters file does not exist: {self.params_file}')

        self.get_logger().info(f'Using parameters file: {self.params_file}')

    def _system_info_callback(self, future):
        """Callback for async system info service call."""
        try:
            result = future.result()
            if result is None:
                self.get_logger().error('System info service call returned None')
                self._system_info_waiting = False
                return
            
            self._idf_file = result.idf_file
            self._vel_scale = result.velocity_scale_factor
            self._env_scale = result.env_scale_factor
            self._system_info_done = True
            self._system_info_waiting = False
            
            self.get_logger().info(
                f'System info received: IDF: {result.idf_file}, vel_scale: {result.velocity_scale_factor}'
            )
            
        except Exception as e:
            self.get_logger().error(f'System info service call exception: {e}')
            self._system_info_waiting = False

    def _namespaces_callback(self, future):
        """Callback for async namespaces service call."""
        try:
            result = future.result()
            if result is None:
                self.get_logger().error('Namespaces service call returned None')
                self._namespaces_waiting = False
                return
            
            self._ns_robots = result.namespaces
            self._namespaces_done = True
            self._namespaces_waiting = False
            
            self.get_logger().info(f'Robot namespaces received: {len(result.namespaces)} robots')
            
        except Exception as e:
            self.get_logger().error(f'Namespaces service call exception: {e}')
            self._namespaces_waiting = False

    def _initialize_pac_status_subscriber(self):
        self._pac_status_subscription = self.create_subscription(
                Int32,
                self._status_topic,
                lambda msg: setattr(self, '_status_pac', msg.data),
                qos_profile=self._qos_profile)
        self._pac_status_initialized = True

    def _get_system_info(self):
        while not self._sim_system_info_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('sim get_system_info service not available, waiting again...')

        request = SystemInfo.Request()
        request.name = self._ns
        try:
            future = self._sim_system_info_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is None:
                self.get_logger().error('Service call returned None')
                return None, None, None

            self.get_logger().info(
                    f'System info received: IDF: {result.idf_file}, vel_scale: {result.velocity_scale_factor}'
                    )

            return result.idf_file, result.velocity_scale_factor, result.env_scale_factor

        except Exception as e:
            self.get_logger().error(f'Service call exception: {e}')
            return None, None, None

    def destroy_node(self):
        """Clean up resources before destroying the node."""
        super().destroy_node()
