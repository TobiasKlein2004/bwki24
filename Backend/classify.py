import torch
import torch_directml
from PIL import Image
from torchvision import transforms

device_name = torch_directml.device_name(0)
device = torch.device(torch_directml.device(0))

print(f'Running on {device_name}')

# Load the model
model = torch.load('models/model1.pt')
model.to(device)

def classifyImage(imagePath) -> list:
    image = Image.open(imagePath)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        out = model(image_tensor)

    # print(out.tolist()[0])
    return out.tolist()[0]