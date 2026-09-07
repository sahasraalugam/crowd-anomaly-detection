"""
extract_frames.py
------------------
Run this ONCE before FinalModel_with_metrics.py.

It reads video clips from:
    Data/video/violent/*.mp4
    Data/video/non-violent/*.mp4

...and saves individual frames as .jpg images into:
    Data/Frames/violent/
    Data/Frames/non-violent/

This is the folder structure your training script (flow_from_directory)
actually needs, since it trains on images, not raw video files.
"""

import os
import cv2

FRAME_SIZE = (150, 150)
FRAME_SKIP = 5  # save every 5th frame (keeps dataset size manageable + reduces near-duplicate frames)

SOURCE_ROOT = "Data/video"
DEST_ROOT = "Data/Frames"

CLASSES = ["violent", "non-violent"]


def extract_frames_from_video(video_path, dest_folder, frame_skip=FRAME_SKIP, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame_resized = cv2.resize(frame, frame_size)
            out_path = os.path.join(dest_folder, f"{video_name}_frame{count}.jpg")
            cv2.imwrite(out_path, frame_resized)
            saved += 1
        count += 1

    cap.release()
    return saved


def main():
    total_saved = 0
    for cls in CLASSES:
        src_dir = os.path.join(SOURCE_ROOT, cls)
        dest_dir = os.path.join(DEST_ROOT, cls)
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"WARNING: {src_dir} does not exist — skipping. "
                  f"Make sure your videos are placed there.")
            continue

        video_files = [f for f in os.listdir(src_dir)
                        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        if not video_files:
            print(f"WARNING: no video files found in {src_dir}")
            continue

        print(f"\nProcessing class '{cls}' — {len(video_files)} video(s) found...")
        class_saved = 0
        for vf in video_files:
            video_path = os.path.join(src_dir, vf)
            saved = extract_frames_from_video(video_path, dest_dir)
            class_saved += saved
            print(f"  {vf}: saved {saved} frames")

        print(f"Class '{cls}' total: {class_saved} frames saved to {dest_dir}")
        total_saved += class_saved

    print(f"\nDONE. Total frames saved across all classes: {total_saved}")
    print("You can now run FinalModel_with_metrics.py")


if __name__ == "__main__":
    main()
