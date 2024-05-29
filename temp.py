import torch
import torch_directml
from PIL import Image
from torchvision import transforms

# Initialize the device
device_name = torch_directml.device_name(0)
device = torch.device(torch_directml.device(0))

# Load the model
model = torch.load('model1.pt')
model.to(device)

# Load and preprocess the image
image_path = '98_100.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
image_tensor = image_tensor.to(device)

# Make prediction
model.eval()
with torch.no_grad():
    out = model(image_tensor)

print(out)