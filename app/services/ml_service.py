from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import io, os

# Import ML scripts (they may rely on optional deps/APIs)

def predict_irradiance(cloud_cover_pct: float, weather: Optional[Dict[str, Any]] = None) -> float:
    """Simple wrapper using provided ML logic; falls back to heuristic if model or deps unavailable."""
    try:
        from app.ml.solar_irradiance_prediction import SolarIrradiancePredictor
        predictor = SolarIrradiancePredictor()
        return float(predictor.predict_from_features(cloud_cover_pct=cloud_cover_pct, weather=weather or {}))
    except Exception as e:
        # Heuristic: clear sky ~ 1000 W/m2, scale by cloud cover
        clear_sky = 1000.0
        return max(0.0, clear_sky * (1.0 - (cloud_cover_pct/100.0)))

def predict_energy_output(irradiance_wm2: float, panel_area_m2: float = 1.6, panel_efficiency: float = 0.20) -> float:
    """Estimate DC output in kW for a PV panel/array."""
    watts = irradiance_wm2 * panel_area_m2 * panel_efficiency
    return round(watts / 1000.0, 4)

def detect_cloud_cover(image_bytes: bytes) -> float:
    """Return estimated cloud cover percentage from an uploaded RGB image."""
    try:
        from app.ml.cloud_detection import detect_cloud_cover as cd
        
        # Optimize memory usage by resizing large images
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize if image is too large to prevent memory issues
        max_size = 1024  # Maximum dimension
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        arr = np.array(img, dtype=np.uint8)  # Use uint8 to save memory
        return float(cd(arr))
        
    except Exception as e:
        print(f"Cloud detection error: {e}")
        # Fallback: naive grayscale threshold with memory optimization
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Resize for memory efficiency
            max_size = 512
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            arr = np.array(img, dtype=np.float32)  # Use float32 instead of float64
            gray = arr.mean(axis=2)
            thr = np.percentile(gray, 70)
            mask = (gray > thr).mean()
            return float(mask * 100.0)
        except Exception as fallback_error:
            print(f"Fallback cloud detection error: {fallback_error}")
            return 50.0  # Default moderate cloud cover