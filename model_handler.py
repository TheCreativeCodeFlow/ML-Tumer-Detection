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
        self.input_size = (224, 224)
        self.model = None
        self.extractor = None
        self.label_map = None
        self._load()

    def _load(self):
        """Load model with improved error handling and fallback strategies."""
        if HF_AVAILABLE:
            try:
                # Try to load with AutoFeatureExtractor first
                try:
                    self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    print(f"✅ Loaded feature extractor for {self.model_name}")
                except Exception as extractor_error:
                    # Fallback for models that might use different preprocessors
                    try:
                        from transformers import AutoImageProcessor
                        self.extractor = AutoImageProcessor.from_pretrained(self.model_name)
                        print(f"✅ Loaded image processor for {self.model_name}")
                    except Exception as processor_error:
                        print(f"⚠️ Could not load preprocessor: {processor_error}")
                
                # Load the model
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True  # Added for models with custom code
                )
                self.model.to(self.device)
                self.model.eval()
                
                # Build medical-specific label map
                if hasattr(self.model.config, 'id2label'):
                    # Ensure all labels are strings
                    self.label_map = {k: str(v) for k, v in self.model.config.id2label.items()}
                else:
                    # Create medical-specific label maps based on model type
                    self.label_map = self._create_default_label_map()
                
                print(f"✅ Successfully loaded medical model: {self.model_name}")
                print(f"   Device: {self.device}")
                print(f"   Label map: {self.label_map}")
                return
            except Exception as e:
                print(f'⚠️ HuggingFace load failed: {e}')
        
        # Fallback to timm for standard backbone
        if TIMM_AVAILABLE:
            try:
                model_name = self.model_name.split('/')[-1]
                if 'resnet' in model_name.lower():
                    model_name = 'resnet50'
                elif 'vit' in model_name.lower():
                    model_name = 'vit_base_patch16_224'
                elif 'efficientnet' in model_name.lower():
                    model_name = 'efficientnet_b0'
                else:
                    model_name = 'resnet50'  # Default fallback
                
                self.model = timm.create_model(model_name, pretrained=True, num_classes=1000)
                self.model.eval()
                self.model.to(self.device)
                # Create ImageNet label map for demo
                self.label_map = {i: f'Class_{i}' for i in range(1000)}
                print(f"✅ Loaded timm model as fallback: {model_name}")
                return
            except Exception as e:
                print(f'⚠️ timm load failed: {e}')
        
        # Create a dummy model for testing
        print('⚠️ Creating dummy medical model for testing...')
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 2),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        self.label_map = self._create_default_label_map()
    
    def _create_default_label_map(self):
        """Create medical-specific default labels based on model name."""
        if "bone-fracture" in self.model_name.lower():
            return {0: 'No Fracture (Normal)', 1: 'Fracture Detected'}
        elif "hemorrhage" in self.model_name.lower():
            return {0: 'No Hemorrhage (Normal)', 1: 'Hemorrhage Detected'}
        elif "tumor" in self.model_name.lower() or "tumour" in self.model_name.lower():
            return {0: 'No Tumor (Normal)', 1: 'Tumor Detected'}
        else:
            return {0: 'Normal', 1: 'Abnormal'}

    def predict(self, input_tensor: np.ndarray):
        """Accepts preprocessed numpy CHW batch (1,C,H,W). Returns (label, confidence, probs).
        """
        try:
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
                
                # Get label and ensure it's a string
                if self.label_map and idx in self.label_map:
                    label = str(self.label_map[idx])
                else:
                    label = str(idx)
                
                # Interpret label heuristically into medical categories
                medical_label, medical_conf = self.interpret_label(label, probs)
                return medical_label, conf, probs
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback
            return "Error in prediction", 0.0, np.array([0.0])

    def gradcam(self, input_tensor: np.ndarray):
        """Return an RGB image with heatmap overlay. If Grad-CAM unavailable, return original image scaled.
        """
        try:
            # Make a displayable image (HWC 0-1)
            img = input_tensor[0].transpose(1, 2, 0)
            # Un-normalize assuming imagenet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_disp = (img * std) + mean
            img_disp = np.clip(img_disp, 0, 1)

            if not GRADCAM_AVAILABLE:
                # Return resized numpy image
                return (img_disp * 255).astype('uint8')

            # Prepare model & target layer: try common conv blocks, otherwise pick the last conv-like module
            target_layers = []
            if hasattr(self.model, 'layer4'):
                target_layers = [self.model.layer4[-1]]
            elif hasattr(self.model, 'blocks'):
                target_layers = [self.model.blocks[-1]]
            elif hasattr(self.model, 'stages'):
                target_layers = [self.model.stages[-1]]
            else:
                # Search for conv layers in reverse order
                convs = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
                if len(convs) > 0:
                    target_layers = [convs[-1]]
                else:
                    # Fallback to last child
                    children = list(self.model.children())
                    if len(children) > 0:
                        target_layers = [children[-1]]
                    else:
                        # No suitable layer found, return original image
                        print("⚠️ No suitable layer for Grad-CAM, returning original image")
                        return (img_disp * 255).astype('uint8')

            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=(self.device.type == 'cuda'))
            # Predict to get target
            _, conf, probs = self.predict(input_tensor)
            idx = int(np.argmax(probs))
            targets = [ClassifierOutputTarget(idx)]
            input_tensor_t = torch.tensor(input_tensor, dtype=torch.float32).to(self.device)
            grayscale_cam = cam(input_tensor_t, targets=targets)[0]
            visualization = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
            return visualization
        except Exception as e:
            print(f'⚠️ Grad-CAM failed: {e}, returning original image')
            # Return original image as fallback
            img = input_tensor[0].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_disp = (img * std) + mean
            img_disp = np.clip(img_disp, 0, 1)
            return (img_disp * 255).astype('uint8')

    def interpret_label(self, label: str, probs: np.ndarray):
        """Enhanced medical interpretation for specialized tumor detection models."""
        # Ensure label is a string
        label = str(label)
        l = label.lower()
        
        # Get the prediction index (0 or 1)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs.max())
        
        # Model-specific interpretations
        if "bone-fracture" in self.model_name.lower():
            # For binary classification: typically 0=normal, 1=fracture
            if pred_idx == 1 or any(k in l for k in ['fracture', 'break', 'crack', '1']):
                severity = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
                return f'Bone Fracture Detected ({severity} Confidence)', confidence
            else:
                return 'No Fracture Detected (Normal)', confidence
        
        elif "hemorrhage" in self.model_name.lower():
            # For binary classification: typically 0=normal, 1=hemorrhage
            if pred_idx == 1 or any(k in l for k in ['hemorrhage', 'bleeding', 'bleed', '1']):
                severity = "Critical" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Mild"
                return f'Intracranial Hemorrhage ({severity} Risk)', confidence
            else:
                return 'No Hemorrhage Detected (Normal)', confidence
        
        elif "tumor" in self.model_name.lower() or "tumour" in self.model_name.lower():
            # For binary classification: typically 0=normal, 1=tumor
            if pred_idx == 1 or any(k in l for k in ['tumor', 'tumour', 'mass', 'lesion', '1', 'positive']):
                severity = "High Suspicion" if confidence > 0.8 else "Moderate Suspicion" if confidence > 0.6 else "Low Suspicion"
                return f'Brain Tumor Detected ({severity})', confidence
            else:
                return 'No Tumor Detected (Normal)', confidence
        
        # General medical classification fallback
        if pred_idx == 1 or any(k in l for k in ['abnormal', 'positive', 'detected', '1']):
            return f'Abnormality Detected: Class {pred_idx}', confidence
        else:
            return f'Normal: Class {pred_idx}', confidence
