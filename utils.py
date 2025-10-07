from PIL import Image, ImageOps
import numpy as np
import io

try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

ALLOWED_EXT = ('.png','.jpg','.jpeg','.dcm')


def allowed_file(filename: str) -> bool:
    fn = filename.lower()
    return any(fn.endswith(ext) for ext in ALLOWED_EXT)


def load_image(file_buffer: io.BytesIO, filename: str):
    """Load image from bytes. Supports JPEG/PNG and DICOM if pydicom installed."""
    fn = filename.lower()
    if fn.endswith('.dcm'):
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom not installed; cannot read DICOM files")
        ds = pydicom.dcmread(file_buffer)
        arr = ds.pixel_array
        # Normalize to 0-255
        arr = arr.astype('float32')
        arr -= arr.min()
        if arr.max()>0:
            arr /= arr.max()
        arr = (arr*255).astype('uint8')
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert('RGB')
        else:
            img = Image.fromarray(arr)
    else:
        img = Image.open(file_buffer)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    return img


def preprocess_image(img: Image.Image, target_size=(224,224)):
    """Resize, center-crop, normalize to [0,1], return numpy tensor CHW."""
    # Resize with aspect and center crop
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype('float32')/255.0
    # Normalize with ImageNet stats (common for pretrained models)
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    arr = (arr - mean)/std
    # HWC to CHW
    arr = np.transpose(arr, (2,0,1))
    return arr[np.newaxis,...]
