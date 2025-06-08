import torch

# Try loading a pretrained model from torchvision
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()
