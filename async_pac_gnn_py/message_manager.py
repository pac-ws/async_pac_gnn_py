#!/usr/bin/env python3

import torch
import numpy as np
from async_pac_gnn_interfaces.msg import RobotPositions, Features
from std_msgs.msg import Header


class MessageManager:
    """
    Utility class for LPAC distributed processing
    """
    
    def __init__(self, comm_range: float, ns: str, ns_index: int, world_map_size: int, 
                 node=None, robot_namespaces=None):
        """
        Initialize MessageManager for distributed processing
        
        Args:
            comm_range: Communication range for filtering neighbors
            ns: Namespace of this robot
            ns_index: Index of this robot in the pose array
            world_map_size: Size of the world map for position normalization
            node: ROS2 node for creating subscriptions (optional)
            robot_namespaces: List of robot namespaces for CNN feature subscriptions (optional)
        """
        self.comm_range = comm_range
        self.ns = ns
        self.ns_index = ns_index
        self.world_map_size = world_map_size
        
        # Cached data members
        self.relative_neighbors = None
        self.own_position = None
        self.edge_index = None
        self.neighbor_indices = None
        self.neighbors_cnn_features = None
        
        # Subscription data
        self.latest_robot_positions = None
        self.cnn_features_by_ns = {}  # {namespace: Features message}
        self.last_position_update_time = None
        
        # Create subscriptions if node provided
        if node is not None and robot_namespaces is not None:
            self._setup_subscriptions(node, robot_namespaces)
    
    def get_relative_neighbors_positions(self, robot_positions: RobotPositions):
        """
        Extract relative neighbor positions from RobotPositions message
        
        Args:
            robot_positions: RobotPositions message containing robot positions in fixed order
            
        Returns:
            tuple: (relative_neighbors_positions, own_position_xy)
                - relative_neighbors_positions: torch.Tensor of shape [num_neighbors, 2]
                - own_position_xy: torch.Tensor of shape [2] containing own x,y position
        """
        # Validate positions array length (must be even)
        if len(robot_positions.positions) % 2 != 0:
            raise ValueError(f"Positions array length {len(robot_positions.positions)} must be even (x,y pairs)")
        
        num_robots = len(robot_positions.positions) // 2
        
        # Validate robot index
        if self.ns_index >= num_robots:
            raise ValueError(f"ns_index {self.ns_index} out of range for {num_robots} robots")
        
        # Extract own position
        own_x = robot_positions.positions[self.ns_index * 2]
        own_y = robot_positions.positions[self.ns_index * 2 + 1]
        own_position_xy = torch.tensor([own_x, own_y], dtype=torch.float32)
        
        # Extract all positions efficiently
        positions_array = np.array(robot_positions.positions, dtype=np.float32)
        all_positions = torch.from_numpy(positions_array.reshape(-1, 2))
        
        # Calculate relative positions
        relative_positions = all_positions - own_position_xy.unsqueeze(0)
        
        # Filter out self and robots outside communication range
        neighbor_indices = []
        comm_range_squared = self.comm_range**2
        
        for i, rel_pos in enumerate(relative_positions):
            if i == self.ns_index:
                continue  # Skip self
            
            distance_squared = torch.sum(rel_pos**2)
            if distance_squared < comm_range_squared:
                neighbor_indices.append(i)
        
        # Extract neighbor positions
        if len(neighbor_indices) == 0:
            relative_neighbors_positions = torch.empty(0, 2, dtype=torch.float32)
        else:
            relative_neighbors_positions = relative_positions[neighbor_indices]
        
        return relative_neighbors_positions, own_position_xy
    
    def _setup_subscriptions(self, node, robot_namespaces):
        """
        Setup ROS2 subscriptions for robot positions and CNN features
        
        Args:
            node: ROS2 node for creating subscriptions
            robot_namespaces: List of robot namespaces
        """
        import rclpy.qos
        
        # QoS profile for subscriptions
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Subscribe to robot positions
        self.positions_subscription = node.create_subscription(
            RobotPositions,
            'robot_poses',
            self._robot_positions_callback,
            qos_profile
        )
        
        # Subscribe to CNN features from each robot
        self.cnn_feature_subscriptions = {}
        for robot_ns in robot_namespaces:
            topic_name = f'/{robot_ns}/cnn_features'
            subscription = node.create_subscription(
                Features,
                topic_name,
                lambda msg, ns=robot_ns: self._cnn_features_callback(msg, ns),
                qos_profile
            )
            self.cnn_feature_subscriptions[robot_ns] = subscription
    
    def _robot_positions_callback(self, msg: RobotPositions):
        """Store latest robot positions message"""
        self.latest_robot_positions = msg
    
    def _cnn_features_callback(self, msg: Features, robot_ns: str):
        """Store latest CNN features message for each robot"""
        self.cnn_features_by_ns[robot_ns] = msg
    
    def process_robot_positions(self, robot_positions: RobotPositions = None):
        """
        Process robot positions and cache computed data
        
        Args:
            robot_positions: RobotPositions message (optional, uses stored if None)
        """
        # Use provided message or latest stored message
        if robot_positions is None:
            robot_positions = self.latest_robot_positions
        
        if robot_positions is None:
            raise ValueError("No robot positions available")
        
        # Record timestamp for filtering CNN features
        import rclpy
        self.last_position_update_time = rclpy.clock.Clock().now()
        # Validate positions array length (must be even)
        if len(robot_positions.positions) % 2 != 0:
            raise ValueError(f"Positions array length {len(robot_positions.positions)} must be even (x,y pairs)")
        
        num_robots = len(robot_positions.positions) // 2
        
        # Validate robot index
        if self.ns_index >= num_robots:
            raise ValueError(f"ns_index {self.ns_index} out of range for {num_robots} robots")
        
        # Extract own position
        own_x = robot_positions.positions[self.ns_index * 2]
        own_y = robot_positions.positions[self.ns_index * 2 + 1]
        self.own_position = torch.tensor([own_x, own_y], dtype=torch.float32)
        
        # Extract all positions efficiently
        positions_array = np.array(robot_positions.positions, dtype=np.float32)
        all_positions = torch.from_numpy(positions_array.reshape(-1, 2))
        
        # Calculate relative positions
        relative_positions = all_positions - self.own_position.unsqueeze(0)
        
        # Filter out self and robots outside communication range
        neighbor_indices = []
        comm_range_squared = self.comm_range**2
        
        for i, rel_pos in enumerate(relative_positions):
            if i == self.ns_index:
                continue  # Skip self
            
            distance_squared = torch.sum(rel_pos**2)
            if distance_squared < comm_range_squared:
                neighbor_indices.append(i)
        
        # Cache neighbor data
        self.neighbor_indices = neighbor_indices
        if len(neighbor_indices) == 0:
            self.relative_neighbors = torch.empty(0, 2, dtype=torch.float32)
            self.edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            self.relative_neighbors = relative_positions[neighbor_indices]
            # Create local edge index: self (index 0) connects to all neighbors (indices 1, 2, ...)
            num_neighbors = len(neighbor_indices)
            # Bidirectional edges: self->neighbors and neighbors->self
            edge_source = torch.cat([torch.zeros(num_neighbors), torch.arange(1, num_neighbors + 1)])
            edge_target = torch.cat([torch.arange(1, num_neighbors + 1), torch.zeros(num_neighbors)])
            self.edge_index = torch.stack([edge_source, edge_target], dim=0).long()
        
        # Process CNN features from neighbors
        self._update_neighbors_cnn_features()
    
    def _update_neighbors_cnn_features(self):
        """
        Update neighbors_cnn_features with filtered and fresh CNN feature data
        """
        if not hasattr(self, 'neighbor_indices') or self.neighbor_indices is None:
            self.neighbors_cnn_features = {}
            return
        
        # Get robot namespaces for neighbors only
        filtered_features = {}
        
        # Get robot namespaces from the global robot list (assuming we have access)
        # For now, we'll use the available CNN features and filter by timestamp
        for robot_ns, features_msg in self.cnn_features_by_ns.items():
            # Simple timestamp filtering - keep features newer than position update
            if (self.last_position_update_time is None or 
                features_msg.header.stamp.sec >= self.last_position_update_time.seconds_nanoseconds()[0]):
                filtered_features[robot_ns] = features_msg
        
        # Switch reference (efficient) rather than copying
        self.neighbors_cnn_features = filtered_features
        
        # Clear old CNN features to start fresh accumulation
        self.cnn_features_by_ns.clear()
    
    def get_relative_neighbors_positions_cached(self):
        """
        Get cached relative neighbor positions
        
        Returns:
            tuple: (relative_neighbors_positions, own_position_xy)
        """
        if self.relative_neighbors is None or self.own_position is None:
            raise ValueError("Must call process_robot_positions first")
        return self.relative_neighbors, self.own_position
    
    def get_edge_index(self):
        """
        Get cached edge index for GNN
        
        Returns:
            torch.Tensor: Edge indices in COO format [2, num_edges]
        """
        if self.edge_index is None:
            raise ValueError("Must call process_robot_positions first")
        return self.edge_index
    
    def get_own_position(self):
        """
        Get cached own position
        
        Returns:
            torch.Tensor: Own position [x, y]
        """
        if self.own_position is None:
            raise ValueError("Must call process_robot_positions first")
        return self.own_position
    
    def cnn_output_to_features(self, cnn_output: torch.Tensor, frame_id: str = "map") -> Features:
        """
        Convert CNN output tensor concatenated with normalized own position to Features message
        
        Args:
            cnn_output: CNN output tensor
            frame_id: Frame ID for the header
            
        Returns:
            Features: ROS2 message with CNN output + normalized position
        """
        import rclpy
        
        if self.own_position is None:
            raise ValueError("Must call process_robot_positions first")
        
        # Normalize own position using same formula as CoverageEnvUtils.normalize_robot_positions
        normalized_position = (self.own_position + self.world_map_size / 2.0) / self.world_map_size
        
        # Concatenate CNN output with normalized position
        combined_features = torch.cat([cnn_output.flatten(), normalized_position])
        
        # Convert to Features message
        msg = Features()
        msg.header = Header()
        msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Flatten tensor and convert to float32 list
        flattened = combined_features.detach().cpu().numpy().astype(np.float32)
        msg.features = flattened.tolist()
        
        return msg
    
    def get_neighbors_cnn_features(self):
        """
        Get cached neighbors CNN features
        
        Returns:
            dict: Dictionary of {robot_namespace: Features message} for neighbors
        """
        if self.neighbors_cnn_features is None:
            return {}
        return self.neighbors_cnn_features
    
    @staticmethod
    def create_manager(comm_range: float, ns: str, ns_index: int, world_map_size: int, 
                      node=None, robot_namespaces=None):
        """
        Static factory method to create MessageManager instance
        
        Args:
            comm_range: Communication range for filtering neighbors
            ns: Namespace of this robot
            ns_index: Index of this robot in the pose array
            world_map_size: Size of the world map for position normalization
            node: ROS2 node for creating subscriptions (optional)
            robot_namespaces: List of robot namespaces for CNN feature subscriptions (optional)
            
        Returns:
            MessageManager: Configured manager instance
        """
        return MessageManager(comm_range, ns, ns_index, world_map_size, node, robot_namespaces)


def tensor_to_features(tensor: torch.Tensor, frame_id: str = "map") -> Features:
    """
    Convert a PyTorch tensor to Features message
    
    Args:
        tensor: Input tensor to convert
        frame_id: Frame ID for the header
        
    Returns:
        Features: ROS2 message with flattened tensor data
    """
    import rclpy
    
    msg = Features()
    msg.header = Header()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = frame_id
    
    # Flatten tensor and convert to float32 list
    flattened = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    msg.features = flattened.tolist()
    
    return msg


def features_to_tensor(features_msg: Features, device: str = "cpu") -> torch.Tensor:
    """
    Convert Features message to PyTorch tensor
    
    Args:
        features_msg: Features message to convert
        device: Device to place the tensor on ("cpu" or "cuda")
        
    Returns:
        torch.Tensor: Tensor containing the feature data
    """
    # Convert list to numpy array then to tensor
    features_array = np.array(features_msg.features, dtype=np.float32)
    tensor = torch.from_numpy(features_array).to(device)
    
    return tensor
