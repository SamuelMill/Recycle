# src/tracking.py
import cv2

def create_csrt():
    """Return a CSRT tracker across OpenCV builds; fall back if needed."""
    # legacy/new API
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    # fallbacks
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    raise RuntimeError("No compatible OpenCV tracker found. Install opencv-contrib-python.")

def auto_init_bbox(frame, backsub, min_area=3000):
    """
    Find a moving blob and return (x,y,w,h), else None.
    """
    mask = backsub.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

