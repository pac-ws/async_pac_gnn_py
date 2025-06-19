import torch
from coverage_control import IOUtils
from coverage_control import Parameters
from coverage_control import CoverageEnvUtils
from coverage_control.nn.models.cnn_backbone import CNNBackBone


class LPACCNNBackbone:

    def __init__(self, config: dict, params: Parameters, state_dicts: dict):
        self.config = config
        self.params = params
        self.use_comm_map = self.config.get("UseCommMap", False)
        self.cnn_map_size = self.config["CNNMapSize"]

        self.device = torch.device("cpu")  # Force CPU for small computations

        self.learning_params_file = IOUtils.sanitize_path(
                self.config["LearningParams"]
                )
        self.learning_params = IOUtils.load_toml(self.learning_params_file)["CNNBackBone"]
        self.model = CNNBackBone(self.learning_params).to(self.device)

        self.model.load_state_dict(state_dicts["cnn_backbone"], strict=True)
        self.model.eval()

    def get_communication_maps_from_positions(
            self, relative_positions: torch.Tensor
            ) -> torch.Tensor:
        # For single robot: relative_positions contains relative positions of neighbors
        # Assumes relative_positions already filtered to be within communication range
        # Output shape: [1, 2, map_size, map_size] for single robot with batch dimension
        map_size = self.cnn_map_size
        comm_maps = torch.zeros((1, 2, map_size, map_size), device=relative_positions.device)

        if len(relative_positions) == 0:
            return comm_maps

        # Precompute constants
        comm_range = self.params.pCommunicationRange
        resolution = self.params.pResolution
        scale_factor = map_size / (comm_range * resolution * 2.0)
        offset = map_size / 2.0 - resolution / 2.0

        # Vectorized scaling and indexing (no need to filter - already within range)
        scaled_indices = torch.round(relative_positions * scale_factor + offset)

        # Clamp indices to valid range to avoid out-of-bounds
        scaled_indices = torch.clamp(scaled_indices, 0, map_size - 1)

        indices = scaled_indices.long().t()  # [2, num_neighbors]
        values = relative_positions / comm_range

        # Use sparse tensor approach for better indexing
        comm_maps[0, 0] = torch.sparse_coo_tensor(
                indices, values[:, 0], torch.Size([map_size, map_size])
                ).to_dense()
        comm_maps[0, 1] = torch.sparse_coo_tensor(
                indices, values[:, 1], torch.Size([map_size, map_size])
                ).to_dense()

        return comm_maps

    def get_maps(self, env, relative_positions: torch.Tensor) -> torch.Tensor:
        # For single robot distributed setup
        # env contains only single robot's data
        raw_local_maps = CoverageEnvUtils.get_raw_local_maps(env, self.params)
        resized_local_maps = CoverageEnvUtils.resize_maps(
                raw_local_maps, self.cnn_map_size
                )
        raw_obstacle_maps = CoverageEnvUtils.get_raw_obstacle_maps(env, self.params)
        resized_obstacle_maps = CoverageEnvUtils.resize_maps(
                raw_obstacle_maps, self.cnn_map_size
                )

        if self.use_comm_map:
            comm_maps = self.get_communication_maps_from_positions(relative_positions)
            # For single robot: shape [1, channels, height, width]
            maps = torch.cat(
                    [
                        resized_local_maps.unsqueeze(1),  # [1, 1, h, w]
                        comm_maps,                        # [1, 2, h, w]
                        resized_obstacle_maps.unsqueeze(1), # [1, 1, h, w]
                        ],
                    1,  # Result: [1, 4, h, w]
                    )
        else:
            maps = torch.cat(
                    [resized_local_maps.unsqueeze(1), resized_obstacle_maps.unsqueeze(1)], 1
                    )  # Result: [1, 2, h, w]

        return maps

    def forward(self, env, relative_positions: torch.Tensor):
        maps = self.get_maps(env, relative_positions)
        maps = maps.to(self.device)

        with torch.no_grad():
            output = self.model(maps)

        return output
