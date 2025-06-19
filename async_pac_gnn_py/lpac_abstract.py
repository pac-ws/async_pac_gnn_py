import os
from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

from async_pac_gnn_interfaces.srv import SystemInfo
from async_pac_gnn_interfaces.srv import UpdateWorldFile
import coverage_control

class LPACAbstract(Node, ABC):
    def __init__(self, node_name: str, is_solo: bool = False):
        super().__init__(node_name)

        self._ns = self.get_namespace()
        self._ns = self._ns[1:] if self._ns.startswith('/') else self._ns  # Remove leading '/'
        self._ns_index = None

        self._qos_profile = self._initialize_qos_profile()

        self._initialize_parameters()
        self._cc_parameters = coverage_control.Parameters(self.params_file)

        self._status_pac = 2
        self._status_topic = '/pac_gcs/status_pac'
        self._pac_status_subscription = None
        self._block_until_pac_status_ready()
        self._initialize_pac_status_subscriber()

        idf_file, self._ns_robots, self._vel_scale, self._env_scale = self._get_system_info()
        if self._ns_robots is None:
            raise RuntimeError('System info retrieval failed')

        if is_solo:
            self._num_robots = 1
        else:
            self._num_robots = len(self._ns_robots)

        self._cc_parameters.pNumRobots = self._num_robots
        self._robot_poses = None

        self._idf_path = '/workspace/' + idf_file
        self._cc_env = None

        self._cmd_vel_topic = 'cmd_vel'
        self._cmd_vel_publishers = []

        self._controller_params = self._get_controller_params()

        self._update_world_file_srv = self.create_service(
                UpdateWorldFile, 
                'update_world_file',
                self._update_world_file_callback
                )


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
        if self._cc_env is None or self._robot_poses is None:
            response.success = False
            response.message = f'[{self._ns}] Uninitialized system'
            return response
        if not self._create_cc_env(idf_path, self._robot_poses):
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

    def _block_until_pac_status_ready(self):
        while True:
            try:
                is_success, msg = wait_for_message(
                        Int32, self, self._status_topic,
                        qos_profile=self._qos_profile,
                        time_to_wait=5.0)
                if is_success == True and msg.data in [0, 1]:
                    self._status_pac = msg.data
                    self.get_logger().info(f'PAC status ready: {self._status_pac}')
                    return
                elif is_success == True:
                    self.get_logger().warn(
                            f'Waiting for status_pac to be ready. Current status: {msg.data}', 
                            once=True
                            )
                else:
                    self.get_logger().warn('Waiting for status_pac message...')

            except rclpy.exceptions.ROSInterruptException:
                self.get_logger().info('Node shutdown requested during PAC status wait')
                raise
            except Exception as e:
                self.get_logger().error(f'Error waiting for PAC status: {e}')
                raise

    def _initialize_pac_status_subscriber(self):
        self._pac_status_subscription = self.create_subscription(
                Int32,
                self._status_topic,
                lambda msg: setattr(self, '_status_pac', msg.data),
                qos_profile=self._qos_profile)

    def _get_system_info(self):
        sim_system_info_client = self.create_client(SystemInfo, 'get_system_info')
        while not sim_system_info_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('sim get_system_info service not available, waiting again...')

        request = SystemInfo.Request()
        request.name = self._ns
        try:
            future = sim_system_info_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is None:
                self.get_logger().error('Service call returned None')
                return None, None, None

            self.get_logger().info(
                    f'System info received: {len(result.namespaces)} robots, '
                    f'IDF: {result.idf_file}, vel_scale: {result.velocity_scale_factor}'
                    )

            return result.idf_file, result.namespaces, result.velocity_scale_factor, result.env_scale_factor

        except Exception as e:
            self.get_logger().error(f'Service call exception: {e}')
            return None, None, None

    def destroy_node(self):
        """Clean up resources before destroying the node."""
        super().destroy_node()
