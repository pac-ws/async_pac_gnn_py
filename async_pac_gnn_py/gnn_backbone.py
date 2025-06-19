import torch
from coverage_control import Parameters
from coverage_control import IOUtils
from coverage_control.nn.models.gnn_backbone import GNNBackBone


class LPACGNNBackbone:

    def __init__(self, config: dict, params: Parameters, state_dicts: dict):
        self.config = config
        self.params = params
        
        self.device = torch.device("cpu")  # Force CPU for small computations
        
        self.learning_params_file = IOUtils.sanitize_path(
                self.config["LearningParams"]
                )
        self.learning_params = IOUtils.load_toml(self.learning_params_file)["GNNBackBone"]
        self.model = GNNBackBone(self.learning_params).to(self.device)

        self.model.load_state_dict(state_dicts["gnn_backbone"], strict=True)
        self.model.eval()

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed GNN processing
        
        Args:
            node_features: Features for this robot and its neighbors [num_nodes, feature_dim]
            edge_index: Local edge connectivity in COO format [2, num_edges]
            
        Returns:
            torch.Tensor: Processed features for this robot
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            output = self.model(node_features, edge_index, edge_weight=None)
        
        return output[0:1]
