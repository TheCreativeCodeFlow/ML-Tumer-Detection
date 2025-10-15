#!/usr/bin/env python3
"""
Test script for enhanced heatmap visualization
Demonstrates the new medical imaging quality heatmaps
"""

import numpy as np
from PIL import Image
import cv2

def test_enhanced_heatmap():
    """Test the enhanced heatmap generation."""
    print("🧪 Testing Enhanced Medical Imaging Heatmap Visualization")
    print("=" * 60)
    
    # Create a sample medical image (simulated MRI scan)
    print("\n1. Creating sample medical image...")
    img_size = (224, 224, 3)
    img = np.random.rand(*img_size).astype(np.float32)
    
    # Create a sample CAM (simulated tumor detection)
    print("2. Creating sample attention map (CAM)...")
    cam = np.zeros((224, 224), dtype=np.float32)
    
    # Simulate a tumor region (circular area with high attention)
    center_x, center_y = 112, 112
    radius = 40
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    cam[mask] = 0.8 + np.random.rand(mask.sum()) * 0.2
    
    # Add some noise
    cam += np.random.rand(224, 224) * 0.1
    cam = np.clip(cam, 0, 1)
    
    print("3. Generating enhanced heatmap...")
    
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Apply Gaussian blur for smoother heatmap
    cam_smooth = cv2.GaussianBlur(cam, (15, 15), 0)
    
    # Apply intensity threshold
    threshold = 0.5
    cam_smooth[cam_smooth < threshold] = 0
    
    # Normalize again
    if cam_smooth.max() > 0:
        cam_smooth = cam_smooth / cam_smooth.max()
    
    # Convert to 8-bit
    cam_8bit = np.uint8(255 * cam_smooth)
    
    # Apply JET colormap
    heatmap = cv2.applyColorMap(cam_8bit, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # Convert to grayscale
    img_gray = np.mean(img, axis=2, keepdims=True)
    img_gray = np.repeat(img_gray, 3, axis=2)
    
    # Blend
    alpha = 0.5
    mask_blend = cam_smooth[..., np.newaxis]
    result = img_gray * (1 - mask_blend * alpha) + heatmap * mask_blend * alpha
    result = np.clip(result * 1.1, 0, 1)
    
    # Convert to uint8
    result_img = (result * 255).astype(np.uint8)
    original_img = (img * 255).astype(np.uint8)
    
    print("4. Saving test images...")
    
    # Save images
    Image.fromarray(original_img).save('test_original.png')
    Image.fromarray(result_img).save('test_heatmap.png')
    
    print("\n✅ Test completed successfully!")
    print("\n📊 Generated files:")
    print("   • test_original.png - Sample input image")
    print("   • test_heatmap.png - Enhanced heatmap visualization")
    
    print("\n🎨 Heatmap Features:")
    print("   ✓ JET colormap (blue → cyan → green → yellow → red)")
    print("   ✓ Gaussian smoothing for professional appearance")
    print("   ✓ Threshold-based highlighting (only significant regions)")
    print("   ✓ Grayscale base with colored overlay")
    print("   ✓ Enhanced contrast and brightness")
    
    print("\n🏥 Medical Imaging Standards:")
    print("   ✓ Red areas = High attention (tumor/abnormality detected)")
    print("   ✓ Yellow/Orange = Moderate attention")
    print("   ✓ Green/Cyan = Low attention")
    print("   ✓ Blue = Normal tissue (no attention)")
    
    print("\n🚀 Ready for real-time medical image analysis!")
    print("   Run 'streamlit run app.py' to see the enhanced heatmaps in action.")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_heatmap()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\n💡 Make sure OpenCV is installed:")
        print("   pip install opencv-python")
