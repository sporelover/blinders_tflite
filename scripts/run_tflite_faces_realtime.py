#!/usr/bin/env python3
"""
Run TFLite model on webcam per-face and display bounding boxes + top-4 percentages.

How it works:
 - Detect faces with OpenCV Haar cascade
 - For each detected face crop and preprocess to model input size
 - Run TFLite interpreter and get output probabilities (handles uint8 quantized outputs)
 - Draw rectangle and display top-4 labels with percentages next to each face

Usage:
  python scripts/run_tflite_faces_realtime.py

Edit constants below if needed (MODEL_PATH, LABELS_PATH, IMG_SIZE, CAMERA_INDEX)
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import tensorflow as tf


# -------------------- Configuration (edit here) --------------------
OUT_DIR = Path('scripts/output')
MODEL_PATH = OUT_DIR / 'model.tflite'
LABELS_PATH = OUT_DIR / 'labels.json'
IMG_SIZE = 224  # must match training img size
CAMERA_INDEX = 0
TOP_K = 4
DEBUG = False  # set True to print per-face debug info to terminal
# -------------------------------------------------------------------


def load_labels(p: Path):
    with open(p, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    # labels might be {"0": "name"} or {0: "name"}
    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out


def preprocess_image(bgr, input_dtype, input_scale=None, input_zero_point=None):
    # bgr: cropped face in BGR
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.issubdtype(input_dtype, np.floating):
        # The exported TFLite model already contains the MobileNetV2 preprocessing step
        # (preprocess_input) because we applied it inside the Keras model before export.
        # Therefore we should pass raw pixel values in range [0,255] as float32.
        img = img.astype('float32')
        return np.expand_dims(img, axis=0)
    else:
        # quantized uint8
        img = img.astype('uint8')
        if input_scale is not None and input_zero_point is not None:
            # convert float -> quantized if needed, but here we assume model expects uint8 images
            return np.expand_dims(img, axis=0)
        return np.expand_dims(img, axis=0)


def dequantize_output(out, scale, zero_point):
    if scale is None:
        return out
    return (out.astype(np.float32) - zero_point) * scale


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f'Model not found: {MODEL_PATH}. Run training script first.')
    if not LABELS_PATH.exists():
        raise SystemExit(f'Labels not found: {LABELS_PATH}. Run training script first.')

    labels = load_labels(LABELS_PATH)
    # Display/update settings
    UPDATE_INTERVAL = 10.0  # seconds between updating displayed prediction
    tracked_center = None
    last_update_time = 0.0
    num_classes = max(labels.keys()) + 1 if labels else 0
    displayed_probs = np.zeros(num_classes, dtype=np.float32)

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details['dtype']
    # handle quantized input params (if any)
    input_scale, input_zero_point = None, None
    if 'quantization' in input_details and input_details['quantization'] is not None:
        q = input_details['quantization']
        if isinstance(q, tuple) and len(q) >= 2:
            input_scale, input_zero_point = q[0], q[1]

    output_scale, output_zero_point = None, None
    if 'quantization' in output_details and output_details['quantization'] is not None:
        q = output_details['quantization']
        if isinstance(q, tuple) and len(q) >= 2:
            output_scale, output_zero_point = q[0], q[1]

    # face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit('Cannot load Haar cascade for face detection')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit('Cannot open camera')

    fps_ts = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # choose the largest detected face to track/display (ignore others)
        if len(faces) == 0:
            # nothing detected; just show frame
            cv2.imshow('TFLite Face Realtime (top-4)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # select largest face by area
        areas = [(w*h, (x, y, w, h)) for (x, y, w, h) in faces]
        areas.sort(reverse=True, key=lambda t: t[0])
        _, (x, y, w, h) = areas[0]
        # enlarge box slightly for better crop
        pad = int(0.2 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)
        face_img = frame[y0:y1, x0:x1]

        # compute center for simple tracking across frames
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center = (cx, cy)

        # decide whether to update displayed prediction now
        now = time.time()
        # if tracked_center is None, force update; also update immediately if person moved far
        should_update = False
        if tracked_center is None:
            should_update = True
        else:
            # distance threshold relative to box size
            max_dim = max(w, h)
            dx = tracked_center[0] - cx
            dy = tracked_center[1] - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > 0.6 * max_dim:
                # new person or big movement -> update immediately
                should_update = True
        if (now - last_update_time) >= UPDATE_INTERVAL:
            should_update = True

        if should_update:
            # run interpreter and update displayed_probs
            inp = preprocess_image(face_img, input_dtype, input_scale, input_zero_point)
            # Ensure dtype matches exactly what interpreter expects and contiguous memory
            expected_dtype = np.dtype(input_dtype)
            try:
                inp = inp.astype(expected_dtype, copy=False)
            except TypeError:
                inp = inp.astype(expected_dtype)
            inp = np.ascontiguousarray(inp)

            interpreter.set_tensor(input_details['index'], inp)
            try:
                interpreter.invoke()
                out = interpreter.get_tensor(output_details['index'])
            except Exception as e:
                if DEBUG:
                    print('Interpreter.invoke() failed:', e)
                out = np.zeros((1, output_details['shape'][1]), dtype=np.float32)
            if output_scale is not None:
                out = dequantize_output(out, output_scale, output_zero_point)

            probs = out.reshape(-1)
            # Softmax if logits
            if probs.min() < 0 or probs.max() > 1.001:
                ex = np.exp(probs - np.max(probs))
                probs = ex / ex.sum()

            # fresh-interpreter fallback when zeros
            if np.allclose(probs, 0):
                if DEBUG:
                    print('Zero output detected, attempting fresh interpreter...')
                try:
                    fresh = tf.lite.Interpreter(model_path=str(MODEL_PATH))
                    fresh.allocate_tensors()
                    fin = fresh.get_input_details()[0]
                    fout = fresh.get_output_details()[0]
                    t = inp.astype(fin['dtype'], copy=False)
                    t = np.ascontiguousarray(t)
                    fresh.set_tensor(fin['index'], t)
                    fresh.invoke()
                    out2 = fresh.get_tensor(fout['index'])
                    if DEBUG:
                        print('fresh output:', out2, 'sum:', float(out2.sum()))
                    if float(out2.sum()) > 0.0:
                        interpreter = fresh
                        input_details = fin
                        output_details = fout
                        input_dtype = input_details['dtype']
                        # update quant params
                        input_scale, input_zero_point = None, None
                        if 'quantization' in input_details and input_details['quantization'] is not None:
                            q = input_details['quantization']
                            if isinstance(q, tuple) and len(q) >= 2:
                                input_scale, input_zero_point = q[0], q[1]
                        output_scale, output_zero_point = None, None
                        if 'quantization' in output_details and output_details['quantization'] is not None:
                            q = output_details['quantization']
                            if isinstance(q, tuple) and len(q) >= 2:
                                output_scale, output_zero_point = q[0], q[1]
                        out = out2
                        probs = out.reshape(-1)
                        if probs.min() < 0 or probs.max() > 1.001:
                            ex = np.exp(probs - np.max(probs))
                            probs = ex / ex.sum()
                except Exception as e:
                    if DEBUG:
                        print('Fresh interpreter run failed:', e)

            # update displayed values and tracking state
            displayed_probs = probs.copy()
            last_update_time = now
            tracked_center = center
            topk_idx = displayed_probs.argsort()[-TOP_K:][::-1]
        else:
            # do not run inference; use last displayed probs
            probs = displayed_probs
            topk_idx = probs.argsort()[-TOP_K:][::-1]

        # draw rectangle
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # compose text lines with top-k (use displayed probs)
        lines = []
        for idx in topk_idx:
            name = labels.get(int(idx), f'id:{idx}')
            pct = float(probs[int(idx)]) * 100.0
            lines.append(f'{name}: {pct:.1f}%')

        # draw semi-transparent background for text
        line_h = 20
        box_w = max(200, int(w))
        box_h = line_h * len(lines) + 4
        tx = x0
        ty = y1 + 5
        # ensure inside frame vertically
        if ty + box_h > frame.shape[0]:
            ty = y0 - 5 - box_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (tx + 5, ty + (i + 1) * line_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frames += 1
        if time.time() - fps_ts >= 1.0:
            fps = frames / (time.time() - fps_ts)
            fps_ts = time.time()
            frames = 0
        # show
        cv2.imshow('TFLite Face Realtime (top-4)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
