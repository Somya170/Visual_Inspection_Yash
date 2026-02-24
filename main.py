import cv2
import numpy as np
import json
import os
import time

# New MediaPipe Tasks API (0.10.30+ compatible)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ZONE_FILE = "zone_config.json"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_detector.tflite"


# ─── ZONE MANAGER ─────────────────────────────────────────
class ZoneManager:
    def __init__(self):
        self.zone_points = []
        self.zone_defined = False
        self.load_zone()

    def load_zone(self):
        if os.path.exists(ZONE_FILE):
            with open(ZONE_FILE, "r") as f:
                data = json.load(f)
                self.zone_points = [tuple(p) for p in data.get("zone", [])]
                if len(self.zone_points) >= 3:
                    self.zone_defined = True
                    print("✅ Saved zone loaded!")

    def save_zone(self):
        with open(ZONE_FILE, "w") as f:
            json.dump({"zone": self.zone_points}, f)
        print("💾 Zone saved!")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.zone_points.append((x, y))
            print(f"📍 Point: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.zone_points) >= 3:
                self.zone_defined = True
                self.save_zone()
                print("✅ Zone confirmed!")
            else:
                print("⚠️ 3 points Minimum")

    def draw_zone(self, frame):
        if not self.zone_points:
            return
        pts = np.array(self.zone_points, np.int32)
        if self.zone_defined:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            for pt in self.zone_points:
                cv2.circle(frame, pt, 6, (0, 255, 0), -1)

    def in_zone(self, point):
        if not self.zone_defined or len(self.zone_points) < 3:
            return False
        pts = np.array(self.zone_points, np.float32)
        return cv2.pointPolygonTest(
            pts, (float(point[0]), float(point[1])), False) >= 0

    def rect_in_zone(self, x, y, w, h):
        checks = [
            (x + w//2, y + h//2),
            (x + w//2, y),
            (x + w//2, y + h),
            (x,        y + h//2),
            (x + w,    y + h//2),
        ]
        return any(self.in_zone(p) for p in checks)

    def reset_zone(self):
        self.zone_points = []
        self.zone_defined = False
        if os.path.exists(ZONE_FILE):
            os.remove(ZONE_FILE)
        print("🔄 Zone reset!")


# ─── DRAW BOX ─────────────────────────────────────────────
def draw_labeled_box(frame, x, y, w, h, label, color):
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), color, -1)
    cv2.putText(frame, label, (x+4, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)


# ─── ALERT ────────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self.last_print = 0

    def trigger(self, frame, parts):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 8)
        cv2.rectangle(frame, (0,0), (w,115), (0,0,0), -1)
        cv2.rectangle(frame, (0,h-40), (w,h), (0,0,0), -1)
        cv2.putText(frame, "!! HUMAN IN DANGER ZONE !!",
                    (10,48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, (0,0,255), 3)
        cv2.putText(frame, "MACHINE TURNED OFF",
                    (10,95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0,165,255), 2)
        parts_str = "Detected: " + " | ".join(parts)
        cv2.putText(frame, parts_str, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0,255,255), 2)
        now = time.time()
        if now - self.last_print > 2.5:
            self.last_print = now
            print("\n" + "🚨"*10)
            print(f"  HUMAN IN HAZARDOUS ZONE!")
            print(f"  Parts → {', '.join(parts)}")
            print(f"  ⚡ MACHINE TURNED OFF")
            print("🚨"*10)

    def safe(self, frame):
        w = frame.shape[1]
        cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
        cv2.putText(frame, "Zone Clear - SAFE",
                    (10,36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (0,220,0), 2)


# ─── MAIN DETECTOR ────────────────────────────────────────
class HazardDetector:
    def __init__(self):
        self.zone  = ZoneManager()
        self.alert = AlertSystem()

        # ── Hand Landmarker (New API) ──
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=HAND_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5)
        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            hand_opts)

        # ── Face Detector (New API) ──
        face_opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.6)
        self.face_det = mp_vision.FaceDetector.create_from_options(
            face_opts)

        # ── HOG Body ──
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

        print("✅ All detectors ready!")

    def run(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Camera 1 nahi mila, 0 try kar raha...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("Hazard Zone Detector")
        cv2.setMouseCallback("Hazard Zone Detector",
                             self.zone.mouse_callback)

        print("\n" + "="*45)
        print("  Left Click  → Add zone point")
        print("  Right Click → Zone confirm")
        print("  R           → Zone reset")
        print("  Q           → Quit")
        print("="*45 + "\n")

        frame_count   = 0
        cached_bodies = []

        # Hand connections for drawing
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fh, fw = frame.shape[:2]
            self.zone.draw_zone(frame)
            parts_in_zone = []

            if self.zone.zone_defined:

                # MediaPipe Image 
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # ── HAND DETECTION ──────────────────────
                hand_result = self.hand_det.detect(mp_image)
                if hand_result.hand_landmarks:
                    for hand_lms in hand_result.hand_landmarks:

                        # Pixel coordinates
                        pts = [(int(lm.x*fw), int(lm.y*fh))
                               for lm in hand_lms]

                        # Skeleton draw
                        for (a, b) in HAND_CONNECTIONS:
                            cv2.line(frame, pts[a], pts[b],
                                     (0,200,200), 2)
                        for pt in pts:
                            cv2.circle(frame, pt, 4,
                                       (0,255,255), -1)

                        # Bounding box
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        hx = max(0, min(xs)-15)
                        hy = max(0, min(ys)-15)
                        hw = min(fw, max(xs)+15) - hx
                        hh = min(fh, max(ys)+15) - hy

                        in_z = self.zone.rect_in_zone(hx,hy,hw,hh)
                        # Finger points directly check
                        for pt in pts:
                            if self.zone.in_zone(pt):
                                in_z = True
                                break

                        col = (0,200,255) if in_z else (0,130,130)
                        draw_labeled_box(
                            frame, hx, hy, hw, hh, "HAND", col)
                        if in_z and "Hand" not in parts_in_zone:
                            parts_in_zone.append("Hand")

                # ── FACE DETECTION ──────────────────────
                face_result = self.face_det.detect(mp_image)
                if face_result.detections:
                    for det in face_result.detections:
                        bbox = det.bounding_box
                        fx = bbox.origin_x
                        fy = bbox.origin_y
                        fw2 = bbox.width
                        fh2 = bbox.height

                        in_z = self.zone.rect_in_zone(
                            fx, fy, fw2, fh2)
                        col = (0,80,255) if in_z else (0,200,255)
                        draw_labeled_box(
                            frame, fx, fy, fw2, fh2, "FACE", col)
                        if in_z and "Face" not in parts_in_zone:
                            parts_in_zone.append("Face")

                # ── BODY (HOG) ──────────────────────────
                if frame_count % 4 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bodies, _ = self.hog.detectMultiScale(
                        gray, winStride=(10,10),
                        padding=(4,4), scale=1.05)
                    cached_bodies = list(bodies) \
                        if len(bodies) > 0 else []

                for (bx, by, bw, bh) in cached_bodies:
                    in_z = self.zone.rect_in_zone(bx,by,bw,bh)
                    col  = (0,50,255) if in_z else (0,200,0)
                    draw_labeled_box(
                        frame, bx, by, bw, bh, "BODY", col)
                    if in_z and "Body" not in parts_in_zone:
                        parts_in_zone.append("Body")

                # ── Alert ───────────────────────────────
                if parts_in_zone:
                    self.alert.trigger(frame, parts_in_zone)
                else:
                    self.alert.safe(frame)

            # Bottom bar
            hf2 = frame.shape[0]
            cv2.rectangle(frame,(0,hf2-28),(640,hf2),(20,20,20),-1)
            if not self.zone.zone_defined:
                cv2.putText(frame,
                    "LEFT CLICK=Point | RIGHT CLICK=Confirm zone",
                    (8,hf2-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0,255,255), 1)
            else:
                cv2.putText(frame,
                    "HAND=Cyan  FACE=Blue  BODY=Green | R=Reset Q=Quit",
                    (8,hf2-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180,180,180), 1)

            if self.zone.zone_defined:
                cv2.putText(frame, "DANGER ZONE",
                    (frame.shape[1]-155, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 2)

            cv2.imshow("Hazard Zone Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.zone.reset_zone()

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Stopped")


if __name__ == "__main__":
    detector = HazardDetector()
    detector.run()
