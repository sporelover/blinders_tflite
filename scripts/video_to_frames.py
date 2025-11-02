#!/usr/bin/env python3
"""
video_to_frames.py

Simple script: given a mapping person->video (or a single video), extract one frame every N frames
and save as images under datas/<person>/. This is intentionally minimal (no face detection) so you can
quickly produce images to inspect and label.

Usage examples:
  python scripts/video_to_frames.py --mapping mapping.json --output dataset --frame-step 5 --resize 224 224
  python scripts/video_to_frames.py --single videos/jairo.mp4 --name jairo --output dataset --frame-step 10

"""
import argparse
import os
from pathlib import Path
import cv2

# --------------------------- Convenience defaults ---------------------------
# If you prefer to hardcode a video path into the script, set DEFAULT_SINGLE_PATH
# and DEFAULT_SINGLE_NAME below. If these are non-empty and you run the script
# without CLI args, it will use them automatically.
# Example:
#   DEFAULT_SINGLE_PATH = r'C:\Users\jairo\Downloads\AI\videos\jairo.mp4'
#   DEFAULT_SINGLE_NAME = 'jairo'
DEFAULT_SINGLE_PATH = r'C:\Users\jairo\Downloads\AI\videos\luis.mp4'
DEFAULT_SINGLE_NAME = 'luis'



def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_frames_for_video(video_path, out_dir, name, frame_step=5, resize=None, max_images=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0
    ensure_dir(os.path.join(out_dir, name))
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            img = frame
            if resize:
                img = cv2.resize(img, (resize[0], resize[1]))
            out_path = os.path.join(out_dir, name, f"img_{saved+1:04d}.jpg")
            cv2.imwrite(out_path, img)
            saved += 1
            if max_images and saved >= max_images:
                break
        idx += 1
    cap.release()
    return saved


def main(args):
    # Determine single video input: priority CLI --single, then DEFAULT_SINGLE_PATH
    if args.single:
        video_path = args.single
        name = args.name or Path(video_path).stem
    elif DEFAULT_SINGLE_PATH:
        video_path = DEFAULT_SINGLE_PATH
        name = DEFAULT_SINGLE_NAME or Path(video_path).stem
    else:
        raise SystemExit('Provide --single path --name PERSON or set DEFAULT_SINGLE_PATH/DEFAULT_SINGLE_NAME inside the script')

    print(f"Processing {name} -> {video_path}")
    c = extract_frames_for_video(video_path, args.output, name, frame_step=args.frame_step,
                                 resize=(args.width, args.height) if args.resize else None,
                                 max_images=args.max_images)
    print(f"Saved {c} images for {name}")
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', help='Single video path', default=None)
    parser.add_argument('--name', help='If using --single, the person name (optional)', default=None)
    parser.add_argument('--output', default='dataset', help='Output dataset dir')
    parser.add_argument('--frame-step', type=int, default=5, help='Extract one frame every N frames')
    parser.add_argument('--resize', action='store_true', help='Resize extracted frames')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()
    main(args)
