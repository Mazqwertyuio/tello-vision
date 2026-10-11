"""Extract key visual sample frames from the DA2 evaluation video.

Saves annotated JPEG frames illustrating different navigation situations:
1. FLAT_WALL_BRAKE - Monochrome wall safety brake
2. CLEAR_AISLE - Forward corridor navigation
3. CENTERING - Lateral alignment between obstacles
4. CLOSED_DOOR - Doorway aperture validation & braking
"""

from pathlib import Path
import cv2
import json

def extract_samples():
    base_dir = Path("debug/20261009_154542")
    video_path = base_dir / "da2_geometry_eval.mp4"
    json_path = base_dir / "da2_geometry_summary.json"
    out_dir = base_dir / "visual_samples"
    out_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick representative frames
    target_frames = {
        5: "01_flat_wall_brake.jpg",
        45: "02_clear_aisle_forward.jpg",
        75: "03_corridor_centering.jpg",
        105: "04_closed_door_brake.jpg",
        180: "05_frame_180_flight_window.jpg"
    }

    extracted = []
    for f_idx, filename in target_frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            save_path = out_dir / filename
            cv2.imwrite(str(save_path), frame)
            extracted.append((f_idx, str(save_path)))
            print(f"Saved: {save_path.name}")

    cap.release()
    print("Done extracting sample frames!")

if __name__ == "__main__":
    extract_samples()
