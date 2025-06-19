import torch
import numpy as np
from collections import OrderedDict
from async_pac_gnn_interfaces.msg import Features
from std_msgs.msg import Header
from coverage_control import IOUtils


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


def load_and_split_state_dict(model_path: str) -> dict:
    """
    Load and split LPAC model state dict into component-specific dictionaries

    Args:
        model_path: Path to the model state dict file

    Returns:
        dict: Dictionary containing separate state dicts for each component:
            - 'cnn_backbone': CNN backbone state dict
            - 'gnn_backbone': GNN backbone state dict  
            - 'output_linear': Output linear layer state dict
            - 'actions': Actions normalization parameters
    """
    # Sanitize the path and load the full state dict
    sanitized_path = IOUtils.sanitize_path(model_path)
    lpac_state_dict = torch.load(sanitized_path, weights_only=True)

    # Initialize component state dicts
    cnn_backbone_dict = OrderedDict()
    gnn_backbone_dict = OrderedDict()
    output_linear_dict = OrderedDict()
    actions_dict = OrderedDict()

    # Split the state dict by component prefixes
    for key, value in lpac_state_dict.items():
        if key.startswith("cnn_backbone."):
            # Remove the "cnn_backbone." prefix
            new_key = key[len("cnn_backbone."):]
            cnn_backbone_dict[new_key] = value
        elif key.startswith("gnn_backbone."):
            # Remove the "gnn_backbone." prefix
            new_key = key[len("gnn_backbone."):]
            gnn_backbone_dict[new_key] = value
        elif key.startswith("output_linear."):
            # Remove the "output_linear." prefix
            new_key = key[len("output_linear."):]
            output_linear_dict[new_key] = value
        elif key in ["actions_mean", "actions_std"]:
            # Keep actions keys as-is
            actions_dict[key] = value

    return {
            "cnn_backbone": cnn_backbone_dict,
            "gnn_backbone": gnn_backbone_dict,
            "output_linear": output_linear_dict,
            "actions": actions_dict
            }
