import torch
from coverage_control import Parameters
from coverage_control.nn.models.gnn_backbone import GNNBackBone


class LPACGNNBackbone:

    def __init__(self, config: dict, params: Parameters):
        self.config = config
        self.params = params
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = GNNBackBone(self.config).to(self.device)
        self.model.eval()

    def step(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
        
        # Return features for the first node (this robot)
        return output[0:1]  # Shape: [1, feature_dim]