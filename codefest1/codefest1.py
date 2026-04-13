import torch
import torchvision.models as models
from torchinfo import summary

model = models.resnet18(pretrained=True)

# Basic summary — just pass input size (batch, channels, height, width)
summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"],
    col_width=20,
    depth=5,          # How many nested levels to show (default: 3)
    device="cpu",     # or "cuda"
    verbose=1,        # 0=quiet, 1=normal, 2=verbose
)