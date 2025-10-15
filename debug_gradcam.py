import numpy as np
from model_handler import ModelHandler
from PIL import Image
import os

MODEL = 'ShimaGh/Brain-Tumor-Detection'
OUT_FILE = 'debug_heatmap.png'

h = ModelHandler(MODEL)

# Try to find any sample image in repo (common names), else use random tensor
candidates = ['test_image.png', 'sample.jpg', 'test_original.png']
img = None
for c in candidates:
    if os.path.exists(c):
        img = Image.open(c).convert('RGB')
        break

if img is None:
    # create a dummy RGB image (random)
    arr = (np.random.rand(224,224,3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)

# Convert to preprocess expected tensor if handler has preprocess helper
try:
    prep = h.preprocess_image(img)
except Exception:
    # fallback: create normalized tensor (1,3,224,224)
    arr = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    # transpose to CHW
    prep = np.transpose(arr, (2,0,1))[None,:,:,:]

print('Calling gradcam...')
cam = h.gradcam(prep, use_enhanced=True)

print('Returned type:', type(cam))
print('Shape:', getattr(cam, 'shape', None))
if hasattr(cam, 'dtype'):
    print('Dtype:', cam.dtype)
try:
    print('Min/Max:', cam.min(), cam.max())
except Exception:
    pass

# Save result if array-like
try:
    from PIL import Image
    if hasattr(cam, 'dtype'):
        out = cam
        if out.dtype != 'uint8':
            # try to scale
            mi = float(out.min())
            ma = float(out.max())
            if ma - mi > 0:
                out = (255.0 * (out - mi) / (ma - mi)).astype(np.uint8)
            else:
                out = (out * 255).astype(np.uint8)
        # ensure 3 channels
        if out.ndim == 2:
            out = np.stack([out]*3, axis=-1)
        if out.shape[2] == 4:
            out = out[:,:,:3]
        Image.fromarray(out).save(OUT_FILE)
        print('Saved', OUT_FILE)
except Exception as e:
    print('Could not save output image:', e)
