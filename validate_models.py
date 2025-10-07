#!/usr/bin/env python3
"""
Model Validation Script for Medical Tumor Detection Models
Tests the four specialized Hugging Face models for medical image analysis
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🩺 Medical Tumor Detection Model Validation")
print("=" * 50)
print(f"Validation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Define medical models to validate
MEDICAL_MODELS = {
    "Bone Fracture Detection": "Hemgg/bone-fracture-detection-using-xray",
    "Intracranial Hemorrhage Detection": "DifeiT/rsna-intracranial-hemorrhage-detection",
    "Brain Tumor Detection": "ShimaGh/Brain-Tumor-Detection",
    "MRI Brain Abnormality Classification": "hugginglearners/brain-tumor-detection-mri"
}

# Test each model
for model_display, model_id in MEDICAL_MODELS.items():
    print(f"🔬 Testing: {model_display}")
    print(f"Model ID: {model_id}")
    
    try:
        # Import and test model handler
        from model_handler import ModelHandler
        
        # Create handler instance
        handler = ModelHandler(model_id)
        
        # Test basic functionality
        print(f"  ✅ Model loaded successfully")
        print(f"  🎨 Device: {handler.device}")
        print(f"  📊 Input size: {handler.input_size}")
        print(f"  🏷️ Label map: {handler.label_map}")
        
        # Test with dummy input
        import numpy as np
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        try:
            pred, conf, probs = handler.predict(dummy_input)
            print(f"  🧠 Prediction test: {pred} (confidence: {conf:.2%})")
            
            # Test Grad-CAM
            try:
                cam_result = handler.gradcam(dummy_input)
                print(f"  🔍 Grad-CAM test: Generated {cam_result.shape if hasattr(cam_result, 'shape') else 'visualization'}")
            except Exception as cam_error:
                print(f"  ⚠️ Grad-CAM test failed: {cam_error}")
            
            print(f"  ✅ Model validation: PASSED")
            
        except Exception as pred_error:
            print(f"  ❌ Prediction test failed: {pred_error}")
            print(f"  💡 This may indicate the model needs special configuration or has compatibility issues")
            
    except Exception as load_error:
        print(f"  ❌ Model loading failed: {load_error}")
        print(f"  💻 Trying alternative loading method...")
        
        # Try direct transformers import
        try:
            from transformers import AutoModelForImageClassification, AutoFeatureExtractor
            model = AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
            try:
                extractor = AutoFeatureExtractor.from_pretrained(model_id)
            except:
                from transformers import AutoImageProcessor
                extractor = AutoImageProcessor.from_pretrained(model_id)
            print(f"  ✅ Direct transformers loading: SUCCESS")
            print(f"  📊 Model config: {model.config.architectures if hasattr(model.config, 'architectures') else 'Unknown'}")
        except Exception as direct_error:
            print(f"  ❌ Direct transformers loading failed: {direct_error}")
            print(f"  💡 Model may require specific dependencies or authentication")
    
    print("-" * 50)
    print()

print("📊 Model Validation Summary")
print("=" * 30)
print("Validation completed. Check individual model results above.")
print()
print("📄 Next Steps:")
print("1. Review any failed model loads")
print("2. Test with actual medical images")
print("3. Validate clinical accuracy with ground truth")
print("4. Run full Streamlit application test")
print()
print(f"Validation finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")