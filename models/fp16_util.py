import torch.nn as nn


def convert_module_to_f16(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.half()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.half()


def convert_module_to_f32(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.float()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.float()
