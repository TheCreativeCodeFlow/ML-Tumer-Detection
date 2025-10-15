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

# Grad-CAM and visualization
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    # Optional ViT reshape transform for token -> spatial map
    try:
        from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform as VIT_RESHAPE
    except Exception:
        VIT_RESHAPE = None
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False

# OpenCV for enhanced heatmap visualization
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# TorchCAM fallback support
try:
    from torchcam.methods import GradCAM as TorchCAMGradCAM
    TORCHCAM_AVAILABLE = True
except Exception:
    TORCHCAM_AVAILABLE = False


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

    def gradcam(self, input_tensor: np.ndarray, use_enhanced=True):
        """Generate professional medical imaging heatmap with enhanced visualization.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            use_enhanced: If True, uses advanced OpenCV-based heatmap visualization
        
        Returns:
            RGB numpy array with heatmap overlay
        """
        try:
            # Make a displayable image (HWC 0-1)
            img = input_tensor[0].transpose(1, 2, 0)
            # Un-normalize assuming imagenet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_disp = (img * std) + mean
            img_disp = np.clip(img_disp, 0, 1)

            if not GRADCAM_AVAILABLE and not TORCHCAM_AVAILABLE:
                # No CAM libs available -> vanilla saliency
                return self._saliency_fallback(input_tensor, img_disp)

            # Prepare model & target layer: try common conv blocks, otherwise pick the last conv-like module
            target_layers = []
            reshape_transform = None

            # CNN backbones
            if hasattr(self.model, 'layer4'):
                target_layers = [self.model.layer4[-1]]
            # Vision Transformer patterns (timm / HF)
            elif hasattr(self.model, 'blocks') or hasattr(self.model, 'vit'):
                # Try to find the last transformer block or a LayerNorm inside
                if hasattr(self.model, 'blocks'):
                    target_layers = [self.model.blocks[-1]]
                elif hasattr(self.model, 'vit'):
                    # Search for the last LayerNorm inside ViT
                    lns = [m for m in self.model.vit.modules() if isinstance(m, torch.nn.LayerNorm)]
                    if lns:
                        target_layers = [lns[-1]]
                    else:
                        # Fallback to last encoder layer
                        enc_layers = list(self.model.vit.modules())
                        target_layers = [enc_layers[-1]] if enc_layers else []
                # Set reshape transform for ViT tokens -> spatial map
                if VIT_RESHAPE is not None:
                    reshape_transform = VIT_RESHAPE
                else:
                    # Minimal ViT reshape transform: (B, N, C) -> (B, C, H, W) with N-1 = H*W
                    def minimal_vit_reshape(tensor):
                        # tensor expected as (B, N, C)
                        B, N, C = tensor.shape
                        S = int((N - 1) ** 0.5)
                        t = tensor[:, 1:, :].reshape(B, S, S, C).permute(0, 3, 1, 2)
                        return t
                    reshape_transform = minimal_vit_reshape
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
                        print("⚠️ No suitable layer for Grad-CAM, returning saliency fallback")
                        return self._saliency_fallback(input_tensor, img_disp)

            # Predict to get target index
            _, conf, probs = self.predict(input_tensor)
            idx = int(np.argmax(probs))

            input_tensor_t = torch.tensor(input_tensor, dtype=torch.float32).to(self.device)

            if GRADCAM_AVAILABLE:
                cam = GradCAM(
                    model=self.model,
                    target_layers=target_layers,
                    use_cuda=(self.device.type == 'cuda'),
                    reshape_transform=reshape_transform
                )
                targets = [ClassifierOutputTarget(idx)]
                grayscale_cam = cam(input_tensor_t, targets=targets)[0]
            elif TORCHCAM_AVAILABLE:
                # TorchCAM path (CNN-friendly). For ViT this may not work; if it fails, fallback to saliency
                try:
                    cam_extractor = TorchCAMGradCAM(self.model, target_layers=target_layers)
                    # Run model forward WITH gradients (no torch.no_grad here)
                    input_tensor_t.requires_grad_(True)
                    outputs = self.model(input_tensor_t)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    activation_maps = cam_extractor(idx, logits)
                    # Take first map
                    cam_map = activation_maps[0].detach().cpu().numpy()
                    # Normalize 0-1
                    if cam_map.max() > cam_map.min():
                        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())
                    # Resize to image size
                    H, W = img_disp.shape[:2]
                    if CV2_AVAILABLE:
                        cam_resized = cv2.resize(cam_map, (W, H))
                    else:
                        from PIL import Image as _PILImage
                        cam_resized = np.array(_PILImage.fromarray((cam_map * 255).astype(np.uint8)).resize((W, H), _PILImage.BILINEAR)) / 255.0
                    grayscale_cam = cam_resized
                except Exception:
                    return self._saliency_fallback(input_tensor, img_disp)
            
            # Use enhanced visualization if available and requested
            if use_enhanced and CV2_AVAILABLE:
                visualization = self._create_enhanced_heatmap(img_disp, grayscale_cam)
            else:
                visualization = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
            
            return visualization
        except Exception as e:
            print(f'⚠️ Grad-CAM failed: {e}, returning original image')
            import traceback
            traceback.print_exc()
            # Return original image as fallback
            return self._saliency_fallback(input_tensor, None)
    
    def _create_enhanced_heatmap(self, img_rgb, heatmap, alpha=0.7):
        """Create professional medical imaging heatmap using OpenCV.
        
        Args:
            img_rgb: Original image in RGB format (0-1 range)
            heatmap: Grayscale attention map (0-1 range)
            alpha: Blending factor (0=original, 1=full heatmap)
        
        Returns:
            Enhanced heatmap visualization as RGB numpy array
        """
        # Convert original image to uint8
        img_uint8 = (img_rgb * 255).astype(np.uint8)
        
        # Enhance heatmap contrast SIGNIFICANTLY for better visibility
        # Use power transform to amplify important regions
        heatmap_enhanced = np.power(heatmap, 0.5)  # More aggressive enhancement
        
        # Normalize to full 0-1 range
        heatmap_min = heatmap_enhanced.min()
        heatmap_max = heatmap_enhanced.max()
        if heatmap_max > heatmap_min:
            heatmap_enhanced = (heatmap_enhanced - heatmap_min) / (heatmap_max - heatmap_min)
        
        # Apply histogram equalization for better contrast
        heatmap_uint8 = (heatmap_enhanced * 255).astype(np.uint8)
        heatmap_eq = cv2.equalizeHist(heatmap_uint8)
        
        # Apply Gaussian blur for smoother, more professional appearance
        heatmap_smooth = cv2.GaussianBlur(heatmap_eq, (21, 21), 0)
        
        # Create VIBRANT colormap (JET is most vibrant for medical imaging)
        # JET: blue (low) -> cyan -> green -> yellow -> orange -> red (high)
        heatmap_color = cv2.applyColorMap(heatmap_smooth, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Create mask - use LOWER threshold to show more activation
        threshold = 0.15  # Show regions with >15% activation
        mask = (heatmap_enhanced > threshold).astype(np.float32)

        # Smooth the mask edges for professional appearance
        mask_smooth = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_smooth = np.expand_dims(mask_smooth, axis=2)

        # Determine if the activation mask is too small; if so, force a full-overlay
        mask_area = mask_smooth.mean()

        # If mask_area is very small, do a full-image overlay so colors are visible
        if mask_area < 0.02:
            # Full overlay using addWeighted to ensure vibrant colors appear
            result_full = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_color, alpha, 0)
            result = result_full
        else:
            # Blend with STRONGER alpha for visible localized colors
            result = img_uint8.astype(np.float32) * (1 - alpha * mask_smooth) + \
                     heatmap_color.astype(np.float32) * (alpha * mask_smooth)
            result = np.clip(result, 0, 255).astype(np.uint8)

        # Optional: Boost saturation of the final result for even more vibrant colors
        try:
            result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            result_hsv[:, :, 1] = np.clip(result_hsv[:, :, 1] * 1.3, 0, 255)  # Increase saturation by 30%
            result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except Exception:
            # If HSV conversion fails for any reason, continue with the current result
            pass

        return result

    def _saliency_fallback(self, input_tensor: np.ndarray, img_disp: np.ndarray | None):
        """Vanilla saliency fallback when CAM libraries are unavailable or fail.
        Returns an RGB uint8 overlay similar to Grad-CAM output.
        """
        try:
            x = torch.tensor(input_tensor, dtype=torch.float32, requires_grad=True).to(self.device)
            out = self.model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            idx = int(torch.argmax(logits, dim=-1).item())
            score = logits[:, idx].sum()
            self.model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            score.backward()
            grads = x.grad.detach().abs().mean(dim=1)[0]  # (H, W)
            grads = grads.cpu().numpy()

            # Normalize to 0-1
            if grads.max() > grads.min():
                grads = (grads - grads.min()) / (grads.max() - grads.min())

            # Build display image if not provided
            if img_disp is None:
                img = input_tensor[0].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_disp = (img * std) + mean
                img_disp = np.clip(img_disp, 0, 1)

            if CV2_AVAILABLE:
                return self._create_enhanced_heatmap(img_disp, grads, alpha=0.7)
            else:
                # Simple numpy blend as last resort
                heatmap_color = np.stack([grads, np.zeros_like(grads), 1 - grads], axis=-1)  # rudimentary blue->red
                overlay = (0.3 * (img_disp) + 0.7 * heatmap_color)
                overlay = np.clip(overlay, 0, 1)
                return (overlay * 255).astype(np.uint8)
        except Exception:
            # As absolute last fallback, return the original image
            if img_disp is None:
                img = input_tensor[0].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_disp = np.clip((img * std) + mean, 0, 1)
            return (img_disp * 255).astype(np.uint8)

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
