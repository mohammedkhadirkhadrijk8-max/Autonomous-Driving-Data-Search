import torch

# 1. Verify that the MPS backend is available and built
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("MPS device not found, falling back to CPU.")

# 2. Assign the model to the MPS device
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model.to(device)

# 3. Assign input tensors to the same device during inference
# inputs = inputs.to(device)
