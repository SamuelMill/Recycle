# src/classifier.py
from pathlib import Path
import json
import numpy as np
import cv2
from joblib import load
from skimage.feature import hog

class HogSvmClassifier:
    def __init__(self, model_path="models/recycle_hog_svm.joblib"):
        self.model_path = str(model_path)
        bundle = load(self.model_path)
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self.label_encoder = bundle["label_encoder"]
        self.classes = np.array(bundle.get("classes", []))

        self.img_size = tuple(bundle.get("img_size", (128, 128)))
        self.hog_params = bundle.get("hog_params", dict(
            orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            transform_sqrt=True, feature_vector=True
        ))
        self.feature_mode = bundle.get("feature_mode", None)


        meta_path = Path(self.model_path).with_name("model_meta.json")
        if (self.feature_mode is None or not self.feature_mode) and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.feature_mode = meta.get("feature_mode", self.feature_mode)
                if "img_size" in meta:
                    self.img_size = tuple(meta["img_size"])
            except Exception:
                pass
        if not self.feature_mode:
            self.feature_mode = "color_edge_hist"  


    def _resize(self, bgr):
        return cv2.resize(bgr, self.img_size, interpolation=cv2.INTER_AREA)

    def _prep(self, bgr):
        bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    def _hog_color(self, bgr):
        img = self._resize(bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats = []
        for ch in cv2.split(rgb):
            feats.extend(hog(ch, **self.hog_params))
        return np.asarray(feats, np.float32)

    def _hog_edge(self, bgr):
        img = self._resize(bgr)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ed = cv2.Canny(gray, 80, 160)
        return hog(ed, **self.hog_params).astype(np.float32)

    def _hsv_hist(self, bgr, bins=(16, 16, 16)):
        img = self._resize(bgr)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180, 0,256, 0,256])
        hist = cv2.normalize(hist, None).flatten().astype(np.float32)
        return hist

    def _features(self, bgr):
        if bgr is None or bgr.size == 0:
            raise ValueError("Empty image")
        mode = (self.feature_mode or "color_edge_hist").lower()
        if mode == "color_edge_hist":
            bgr = self._prep(bgr)
            f = np.concatenate([self._hog_color(bgr), self._hog_edge(bgr), self._hsv_hist(bgr)], axis=0)
            return f.reshape(1, -1)
        if mode == "color":
            bgr = self._prep(bgr)
            f = self._hog_color(bgr)
            return f.reshape(1, -1)

        img = self._resize(bgr)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = hog(gray, **self.hog_params).astype(np.float32)
        return f.reshape(1, -1)

    def predict_proba_vec(self, bgr):
        Xs = self._features(bgr)
        Xs = self.scaler.transform(Xs)
        return self.svm.predict_proba(Xs)[0]

    def topk(self, bgr, k=3):
        probs = self.predict_proba_vec(bgr)
        order = np.argsort(-probs)[:k]
        return [(self.classes[i], float(probs[i])) for i in order]

