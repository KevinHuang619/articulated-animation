import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor

num_regions = 10
num_channels = 3
revert_axis_swap = True
pixelwise_flow_predictor_params = dict(
    block_expansion=64,
    max_features=1024,
    num_blocks=5,
    scale_factor=0.25,
    use_deformed_source=True,
    use_covar_heatmap=True,
    estimate_occlusion_map=True
)

pixelwise_flow_predictor = PixelwiseFlowPredictor(
    num_regions=num_regions, 
    num_channels=num_channels,
    revert_axis_swap=revert_axis_swap,
    **pixelwise_flow_predictor_params)

pixelwise_flow_predictor()
