import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import ModifiedCassavaNet

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]


def load_model(weights_path, device):
    model = ModifiedCassavaNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)


class TTA:
    def __init__(self, model, device, n_aug=5):
        self.model  = model
        self.device = device
        self.n_aug  = n_aug

        self.base_pipeline = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.tta_pipeline = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.SafeRotate(limit=10, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _preprocess(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        return image  # HWC uint8 numpy RGB

    def _forward(self, tensor):
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out, _, _ = self.model(tensor)
        return torch.softmax(out, dim=1).squeeze(0)

    def predict(self, image):
        image = self._preprocess(image)

        probs = [self._forward(self.base_pipeline(image=image)["image"])]
        for _ in range(self.n_aug):
            probs.append(self._forward(self.tta_pipeline(image=image)["image"]))

        mean_probs = torch.stack(probs).mean(dim=0).cpu()
        class_idx  = mean_probs.argmax().item()
        confidence = mean_probs[class_idx].item()

        return {
            "class_idx":  class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4),
            "all_probs":  {
                name: round(mean_probs[i].item(), 4)
                for i, name in enumerate(CLASS_NAMES)
            }
        }

    def predict_batch(self, images):
        return [self.predict(img) for img in images]