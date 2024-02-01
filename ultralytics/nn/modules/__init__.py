# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    Efficient_TRT_NMS,
    ONNX_NMS,
    ResNetLayer,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    ResBlockCBAM,
    SimFusion4In,
    SimFusion3In,
    IFM,
    InjectionMultiSumAutoPool,
    PyramidPoolAgg,
    AdvPoolFusion,
    TopBasicLayer,
    DropPath,
    trunc_normal_,
)
from .head import (
    Classify,
    Detect,
    Pose,
    OBB,
    RTDETRDecoder,
    Segment,
    PostDetectTRTNMS,
    PostDetectONNXNMS,
    DetectEfficient,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .activation import (
    MemoryEfficientSwish,
    HSigmoid,
    Squareplus,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "PostDetectTRTNMS",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "Efficient_TRT_NMS",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ONNX_NMS",
    "PostDetectONNXNMS",
    "ResBlockCBAM",
    "SimFusion4In",
    "SimFusion3In",
    "IFM",
    "InjectionMultiSumAutoPool",
    "PyramidPoolAgg",
    "AdvPoolFusion",
    "TopBasicLayer",
    "DropPath",
    "trunc_normal_",
    "MemoryEfficientSwish",
    "HSigmoid",
    "DetectEfficient",
    "Squareplus",
    "OBB",
)
