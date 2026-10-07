import cv2
import numpy as np

class VisionEnhancer:
    def __init__(self, mode='clahe', tint_green=False, thermal=False):
        """
        Initializes the vision enhancement module.
        mode: 'clahe' (low light enhancement) or 'gamma'
        tint_green: applies a green night-vision effect
        thermal: applies a heat radar (thermal) effect
        """
        self.mode = mode
        self.tint_green = tint_green
        self.thermal = thermal
        if self.mode == 'clahe':
            # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance(self, frame):
        """Applies night vision / low-light enhancement to the frame."""
        # Low light enhancement
        if self.mode == 'clahe':
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            cl = self.clahe.apply(l)
            
            # Merge the CLAHE enhanced L-channel back
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            enhanced = frame.copy()
            
        # Optional thermal radar or night-vision green tint
        if getattr(self, 'thermal', False):
            # Apply a heat radar (thermal) effect
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            # Invert so brighter areas are 'hotter' in thermal colormap
            enhanced = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        elif self.tint_green:
            # Convert to grayscale, then create a green-tinted BGR image
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            green_tint = np.zeros_like(enhanced)
            # Set the Green channel to the grayscale values
            green_tint[:, :, 1] = gray
            enhanced = green_tint

        return enhanced
