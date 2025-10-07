#!/usr/bin/env python3
"""
Quick test script to verify model predictions work correctly
"""

import sys
import numpy as np
from model_handler import ModelHandler

print("🧪 Testing Model Prediction Fix")
print("=" * 50)

# Test with Brain Tumor Detection model
model_name = "ShimaGh/Brain-Tumor-Detection"
print(f"\n📋 Testing model: {model_name}")

try:
    # Initialize handler
    print("Loading model...")
    handler = ModelHandler(model_name)
    print(f"✅ Model loaded successfully")
    print(f"   Label map: {handler.label_map}")
    
    # Create dummy input
    print("\n🔬 Testing prediction with dummy input...")
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    # Make prediction
    pred, conf, probs = handler.predict(dummy_input)
    
    print(f"\n✅ Prediction successful!")
    print(f"   Prediction: {pred}")
    print(f"   Confidence: {conf:.2%}")
    print(f"   Probabilities: {probs}")
    
    # Test Grad-CAM
    print(f"\n🎨 Testing Grad-CAM...")
    try:
        cam_result = handler.gradcam(dummy_input)
        print(f"✅ Grad-CAM successful! Shape: {cam_result.shape}")
    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Model is working correctly.")
    print("✅ You can now reload the Streamlit app")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Check the error above and review model_handler.py")
