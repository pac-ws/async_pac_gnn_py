import torch
from torch_geometric.nn import MLP
from coverage_control import Parameters
from coverage_control import CoverageSystem
from coverage_control import PointVector
from coverage_control import IOUtils
from coverage_control import CoverageEnvUtils
import coverage_control.nn as cc_nn

class LPACController:

    def __init__(self, config: dict, params: Parameters):
        self.config = config
        self.params = params
        self.use_cnn = self.config["UseCNN"]
        self.use_comm_map = self.config["UseCommMap"]
        self.cnn_map_size = self.config["CNNMapSize"]

        self.device = torch.device("cpu")  # Force CPU for small computations
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

class LPACActionNN(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.gnn_mlp = MLP([
            config["InputDim"],
            config["MLP"]["HiddenUnits"],
            config["MLP"]["OutDim"]
            ])

        self.output_linear = torch.nn.Linear(
            config["MLP"]["OutDim"],
            config["OutDim"]
            )

        self.register_buffer("actions_mean", torch.zeros(config["OutDim"]))
        self.register_buffer("actions_std", torch.ones(config["OutDim"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gnn_mlp(x)
        x = self.output_linear(x)
        return x

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        super().load_state_dict(state_dict, strict=strict)
    

class LPACAction:
    def __init__(self, lparams: dict, state_dicts: dict):
        self.device = torch.device("cpu")  # Force CPU for small computations
        self.lparams = lparams

        self.model = LPACActionNN(lparams)
        self.model.load_state_dict(state_dicts, strict=True)
        self.model.eval()

    def step(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.model(x)
            action = action * self.model.actions_std + self.model.actions_mean
        return action
