import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Try importing transformers & timm; fall back gracefully
try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False


class ModelHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = (224,224)
        self.model = None
        self.extractor = None
        self.label_map = None
        self._load()

    def _load(self):
        if HF_AVAILABLE:
            try:
                # Try to load with AutoFeatureExtractor first
                try:
                    self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                except Exception:
                    # Fallback for models that might use different preprocessors
                    from transformers import AutoImageProcessor
                    self.extractor = AutoImageProcessor.from_pretrained(self.model_name)
                
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # build medical-specific label map
                if hasattr(self.model.config, 'id2label'):
                    self.label_map = self.model.config.id2label
                else:
                    # Create medical-specific label maps based on model type
                    if "bone-fracture" in self.model_name:
                        self.label_map = {0: 'No Fracture', 1: 'Fracture Detected'}
                    elif "hemorrhage" in self.model_name:
                        self.label_map = {0: 'No Hemorrhage', 1: 'Hemorrhage Detected'}
                    elif "tumor" in self.model_name.lower():
                        self.label_map = {0: 'No Tumor', 1: 'Tumor Detected'}
                    else:
                        self.label_map = {0: 'Normal', 1: 'Abnormal'}
                
                print(f"Successfully loaded medical model: {self.model_name}")
                return
            except Exception as e:
                print('HF load failed:', e)
        
        # fallback to timm for standard backbone
        if TIMM_AVAILABLE:
            try:
                model_name = self.model_name.split('/')[-1]
                if 'resnet' in model_name.lower():
                    model_name = 'resnet50'
                elif 'vit' in model_name.lower():
                    model_name = 'vit_base_patch16_224'
                self.model = timm.create_model(model_name, pretrained=True, num_classes=1000)
                self.model.eval()
                self.model.to(self.device)
                # Create ImageNet label map for demo
                self.label_map = {i: f'Class_{i}' for i in range(1000)}
                return
            except Exception as e:
                print('timm load failed:', e)
        
        # Create a dummy model for testing
        print('Creating dummy medical model for testing...')
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 2),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # Medical-specific dummy labels
        if "bone-fracture" in self.model_name:
            self.label_map = {0: 'No Fracture', 1: 'Fracture Detected'}
        elif "hemorrhage" in self.model_name:
            self.label_map = {0: 'No Hemorrhage', 1: 'Hemorrhage Detected'}
        elif "tumor" in self.model_name.lower():
            self.label_map = {0: 'No Tumor', 1: 'Tumor Detected'}
        else:
            self.label_map = {0: 'Normal', 1: 'Abnormal'}

    def predict(self, input_tensor: np.ndarray):
        """Accepts preprocessed numpy CHW batch (1,C,H,W). Returns (label, confidence, probs).
        """
        x = torch.tensor(input_tensor, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = self.label_map[idx] if self.label_map and idx in self.label_map else str(idx)
            # interpret label heuristically into trauma categories
            trauma, trauma_conf = self.interpret_label(label, probs)
            return trauma, conf, probs

    def gradcam(self, input_tensor: np.ndarray):
        """Return an RGB image with heatmap overlay. If Grad-CAM unavailable, return original image scaled.
        """
        # Make a displayable image (HWC 0-1)
        img = input_tensor[0].transpose(1,2,0)
        # un-normalize assuming imagenet stats
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img_disp = (img * std) + mean
        img_disp = np.clip(img_disp, 0, 1)

        if not GRADCAM_AVAILABLE:
            # return resized numpy image
            return (img_disp*255).astype('uint8')

        try:
            # prepare model & target layer: try common conv blocks, otherwise pick the last conv-like module
            target_layers = []
            if hasattr(self.model, 'layer4'):
                target_layers = [self.model.layer4[-1]]
            elif hasattr(self.model, 'blocks'):
                target_layers = [self.model.blocks[-1]]
            else:
                # search for conv layers in reverse order
                convs = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
                if len(convs) > 0:
                    target_layers = [convs[-1]]
                else:
                    # fallback to last child
                    children = list(self.model.children())
                    if len(children) > 0:
                        target_layers = [children[-1]]
                    else:
                        # No suitable layer found, return original image
                        return (img_disp*255).astype('uint8')

            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=(self.device.type=='cuda'))
            # predict to get target
            _, conf, probs = self.predict(input_tensor)
            idx = int(np.argmax(probs))
            targets = [ClassifierOutputTarget(idx)]
            input_tensor_t = torch.tensor(input_tensor, dtype=torch.float32).to(self.device)
            grayscale_cam = cam(input_tensor_t, targets=targets)[0]
            visualization = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
            return visualization
        except Exception as e:
            print(f'Grad-CAM failed: {e}, returning original image')
            return (img_disp*255).astype('uint8')

    def interpret_label(self, label: str, probs: np.ndarray):
        """Enhanced medical interpretation for specialized trauma detection models."""
        l = label.lower()
        
        # Model-specific interpretations
        if "bone-fracture" in self.model_name:
            if any(k in l for k in ['fracture', 'break', 'crack', '1']):
                severity = "High" if probs.max() > 0.8 else "Moderate" if probs.max() > 0.6 else "Low"
                return f'Bone Fracture Detected ({severity} Confidence)', probs.max()
            else:
                return 'No Fracture Detected', probs.max()
        
        elif "hemorrhage" in self.model_name:
            if any(k in l for k in ['hemorrhage', 'bleeding', 'bleed', '1']):
                severity = "Critical" if probs.max() > 0.8 else "Moderate" if probs.max() > 0.6 else "Mild"
                return f'Intracranial Hemorrhage ({severity} Risk)', probs.max()
            else:
                return 'No Hemorrhage Detected', probs.max()
        
        elif "tumor" in self.model_name.lower():
            if any(k in l for k in ['tumor', 'tumour', 'mass', 'lesion', '1']):
                severity = "High Suspicion" if probs.max() > 0.8 else "Moderate Suspicion" if probs.max() > 0.6 else "Low Suspicion"
                return f'Brain Tumor Detected ({severity})', probs.max()
            else:
                return 'No Tumor Detected', probs.max()
        
        # General medical classification fallback
        if any(k in l for k in ['abnormal', 'positive', 'detected', '1']):
            return f'Abnormality Detected: {label}', probs.max()
        else:
            return f'Normal: {label}', probs.max()
