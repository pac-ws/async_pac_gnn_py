import torch
from coverage_control import Parameters
from coverage_control import IOUtils
from coverage_control.nn.models.gnn_backbone import GNNBackBone


class LPACGNNBackbone:

    def __init__(self, learning_params: dict, state_dicts: dict):
        self.device = torch.device("cpu")  # Force CPU for small computations
        
        self.lparams = learning_params
        
        self.latent_size = self.lparams["LatentSize"]

        self.model = GNNBackBone(self.lparams).to(self.device)
        self.model.load_state_dict(state_dicts, strict=True)
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
        
        with torch.no_grad():
            output = self.model(node_features, edge_index, edge_weight=None)
        
        return output[0:1]
