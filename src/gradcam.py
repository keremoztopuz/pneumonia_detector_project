import os
import torch
import numpy as np 
from PIL import Image 

import matplotlib.pyplot as plt
from torchvision import transforms

from src.config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES, IMAGE_SIZE, MEAN, STD
from src.model import create_model

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        #forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        #backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        #CAM calculation
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1,2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class
    
def visualize_gradcam(image_path, save_path=None):
    #model loading
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    #last conv layer
    target_layer = model.stages[-1].blocks[-1]

    #gradcam
    gradcam = GradCAM(model, target_layer)
    
    #load image
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    #generate cam
    cam, pred_class = gradcam.generate_cam(input_tensor)

    #resizing
    cam_resized = np.uint8(255 * cam)
    cam_resized = Image.fromarray(cam_resized).resize((IMAGE_SIZE, IMAGE_SIZE))
    cam_resized = np.array(cam_resized)

    #heatmap
    heatmap = plt.cm.jet(cam_resized)[:,:,:3]
    heatmap = np.uint8(255 * heatmap)

    #original image
    original = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original = np.array(original)    
    
    #overlay
    overlay = (0.5 * original + 0.5 * heatmap).astype(np.uint8)

    #visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("original")
    
    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"overlay - pred: {CLASS_NAMES[pred_class]}")
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grad-CAM saved to {save_path}")
    else:
        plt.show()
    
    return cam, pred_class

if __name__ == "__main__":
    test_image = "data/test/PNEUMONIA/person1_virus_6.jpeg"
    visualize_gradcam(test_image, save_path="gradcam_result.png")