# Ultralytics YOLO 🚀, AGPL-3.0 license
"""ExecuTorch (*.pte) export for on-device PyTorch inference via XNNPACK."""

from __future__ import annotations

import types
from functools import partial
from pathlib import Path

import torch

from ultralytics.nn.modules import Pose, Pose26
from ultralytics.utils import LOGGER, YAML


def executorch_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """Patch Pose heads with XNNPACK-safe keypoint decoding for ExecuTorch export.

    XNNPACK requires explicit dim matching for broadcasting, so 2D anchor/stride tensors are
    expanded to 4D before combination with keypoint predictions.
    """
    for m in model.modules():
        if not isinstance(m, Pose):
            continue
        m.kpts_decode = types.MethodType(partial(_executorch_kpts_decode, is_pose26=isinstance(m, Pose26)), m)
    return model


def _executorch_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    """Decode pose keypoints for ExecuTorch export with XNNPACK-safe broadcasting."""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    y = kpts.view(bs, *self.kpt_shape, -1)

    anchors = self.anchors[None, None]
    strides = self.strides[None, None]
    a = ((y[:, :, :2] + anchors) if is_pose26 else (y[:, :, :2] * 2.0 + (anchors - 0.5))) * strides
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def torch2executorch(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> tuple[str, None]:
    """Export a PyTorch model to ExecuTorch ``*.pte`` format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing/export.
        output_dir (Path | str): Directory to save the exported ExecuTorch model.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (tuple[str, None]): Path to the exported ``_executorch_model`` directory and ``None``.
    """
    from executorch import version as executorch_version
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = executorch_wrapper(model)
    pte_file = output_dir / "model.pte"
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (im,)),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    pte_file.write_bytes(et_program.buffer)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir), None
