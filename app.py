# app.py — CSRT + HOG-SVM demo
import sys, os
sys.path.append(os.path.dirname(__file__))  
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.classifier import HogSvmClassifier


import cv2
import numpy as np
import streamlit as st
from collections import deque


from src.tracking import create_csrt, auto_init_bbox

def st_image_safe(ph, img_rgb):
    try:
        ph.image(img_rgb, channels="RGB", use_container_width=True)
    except TypeError:
        ph.image(img_rgb, channels="RGB", use_column_width=True)

def smart_crop(frame, box, margin=0.15):
    x, y, w, h = map(int, box)
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)
    return frame[y1:y2, x1:x2]

st.set_page_config(page_title="Recycle Object Recognition (CSRT + HOG-SVM)", layout="wide")
st.title("♻️ Recycle Object Recognition — CSRT Tracker + HOG-SVM Classifier")

with st.sidebar:
    st.header("Settings")
    model_path    = st.text_input("Model path", "models/recycle_hog_svm.joblib")
    use_webcam    = st.toggle("Use webcam", value=False)
    min_area      = st.slider("Auto-ROI min area (px)", 500, 20000, 3000, step=500)
    warmup_frames = st.slider("BG warmup frames", 0, 200, 30, step=10)
    smooth_win    = st.slider("Label smoothing (frames)", 1, 21, 7, step=2)
    margin_pct    = st.slider("BBox margin (%)", 0, 40, 20, step=5)
    min_conf      = st.slider("Min confidence", 0.0, 1.0, 0.35, step=0.05)
    st.caption("Tip: lower min_area for small items; margin adds context; smoothing stabilizes labels.")
    run_btn       = st.button("Run")

video_file = None
if not use_webcam:
    video_file = st.file_uploader("Upload a short video (mp4/avi/mov/mkv)", type=["mp4", "avi", "mov", "mkv"])

frame_placeholder  = st.empty()
status_placeholder = st.empty()


if run_btn:
    try:
        clf = HogSvmClassifier(model_path=model_path)
        st.info(f"Model loaded. Classes: {clf.classes} | feature_mode={clf.feature_mode} | img_size={clf.img_size}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if use_webcam:
        cap = cv2.VideoCapture(0)
        src_name = "Webcam"
    else:
        if video_file is None:
            st.warning("Please upload a video or enable webcam.")
            st.stop()
        tpath = "temp_input.mp4"
        with open(tpath, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(tpath)
        src_name = "Uploaded Video"

    if not cap.isOpened():
        st.error("Could not open video source.")
        st.stop()

    backsub     = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    tracker     = None
    initialized = False
    labels_hist = deque(maxlen=smooth_win)
    track_id    = 1
    warm        = warmup_frames

    status_placeholder.info(f"Source: {src_name} — warming up background model…")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_disp = frame.copy()

        if warm > 0:
            backsub.apply(frame)
            warm -= 1
            cv2.putText(frame_disp, f"Warming up BG… {warm}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        else:
            if not initialized:
                bbox = auto_init_bbox(frame, backsub, min_area=min_area)
                if bbox is not None:
                    tracker = create_csrt()
                    tracker.init(frame, tuple(bbox))
                    initialized = True
                    labels_hist.clear()
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_disp, "CSRT initialized", (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame_disp, "Searching for moving object…", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            else:
                ok_trk, box = tracker.update(frame)
                if not ok_trk:
                    initialized = False
                    cv2.putText(frame_disp, "Tracking lost — reinitializing…", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    x, y, w, h = map(int, box)
                    x2, y2 = x + w, y + h
                    cv2.rectangle(frame_disp, (x, y), (x2, y2), (0, 255, 0), 2)

                    crop = smart_crop(frame, box, margin=margin_pct / 100.0)
                    if crop.size > 0:

                        crop = cv2.GaussianBlur(crop, (3, 3), 0)
                        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                        L, A, B = cv2.split(lab)
                        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
                        crop = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

                        preds = clf.topk(crop, k=3)  
                        label, conf = preds[0]
                        shown = f"{label} ({conf:.2f})" if conf >= min_conf else f"unsure ({conf:.2f})"

                        labels_hist.append(label if conf >= min_conf else "unsure")

                        cv2.putText(frame_disp, f"ID {track_id}: {shown}",
                                    (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)

                        if len(preds) > 1:
                            alt1, c1 = preds[1]
                            cv2.putText(frame_disp, f"alt1: {alt1} ({c1:.2f})",
                                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (200, 200, 200), 2)
                        if len(preds) > 2:
                            alt2, c2 = preds[2]
                            cv2.putText(frame_disp, f"alt2: {alt2} ({c2:.2f})",
                                        (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (150, 150, 150), 2)
                    else:
                        cv2.putText(frame_disp, "Empty crop", (x, max(20, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        st_image_safe(frame_placeholder, frame_rgb)

    cap.release()
    status_placeholder.success("Done.")
