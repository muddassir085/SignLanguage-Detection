"""Run real-time webcam inference with the trained landmark MLP."""
# .\.venv\Scripts\Activate.ps1
# python src/webcam.py --threshold 0.70 --show-landmarks

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import tensorflow as tf

from landmarks import create_holistic, extract_features_from_rgb


def draw_landmarks(frame, result) -> None:
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    if result.face_landmarks:
        drawing.draw_landmarks(frame, result.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=styles.get_default_face_mesh_contours_style())
    for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
        if hand:
            drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum confidence before showing a class.")
    parser.add_argument("--show-landmarks", action="store_true", help="Overlay detected face/hand landmarks.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    classes = json.loads((args.models_dir / "classes.json").read_text(encoding="utf-8"))
    scaler = joblib.load(args.models_dir / "scaler.pkl")
    model = tf.keras.models.load_model(args.models_dir / "best_model.keras")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        with create_holistic(static_image_mode=False) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extraction = extract_features_from_rgb(rgb, holistic)
                hand_detected = bool(
                    extraction.result.left_hand_landmarks or extraction.result.right_hand_landmarks
                )
                # A classifier must always choose a class. Do not let an all-missing
                # landmark vector become a misleading high-confidence gesture label.
                if hand_detected:
                    probabilities = model.predict(scaler.transform([extraction.features]), verbose=0)[0]
                    index = int(probabilities.argmax())
                    confidence = float(probabilities[index])
                    gesture = classes[index] if confidence >= args.threshold else "Unknown / Uncertain"
                    confidence_text = f"Confidence: {confidence:.0%}"
                else:
                    gesture = "Unknown / Uncertain"
                    confidence_text = "Confidence: no hand detected"
                cv2.putText(frame, f"Gesture: {gesture}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, confidence_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                if args.show_landmarks:
                    draw_landmarks(frame, extraction.result)
                cv2.imshow("Sign Language Recognition (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
