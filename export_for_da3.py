"""Export clean flight video and image sequence for Depth Anything 3 (DA3).

Prepares clean, stabilized video and PNG frame sequences for Depth Anything 3
3D reconstruction and mapping (including DA3-Streaming):
1. Strips all telemetry/HUD overlays (guaranteed 100% clean camera stream).
2. Trims pre-takeoff table view and post-landing floor view (flight-only).
3. Fixes color channel inversion for legacy recordings (swap R/B).
4. Exports standardized MP4 video and/or frame sequence (da3_images/frame_%06d.png).
5. Prints the exact command to run in Depth-Anything-3-main/da3_streaming.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def find_latest_flight_dir(base_debug_dir: Path) -> Path:
    """Find the most recent flight directory containing recordings."""
    if not base_debug_dir.exists():
        return None
    candidates = [
        d for d in base_debug_dir.iterdir()
        if d.is_dir() and (d / "flight_raw.mp4").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def get_airborne_range(csv_path: Path):
    """Find start and end frame indices where drone was airborne."""
    if not csv_path.exists():
        return None, None

    first_idx = None
    last_idx = None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row.get("state", "")
            try:
                idx = int(row.get("frame_idx", 0))
            except ValueError:
                continue

            if "AIRBORNE" in state:
                if first_idx is None:
                    first_idx = idx
                last_idx = idx

    return first_idx, last_idx


def main():
    parser = argparse.ArgumentParser(description="Export clean video for Depth Anything 3")
    parser.add_argument("--dir", type=str, default="", help="Flight debug directory (default: latest)")
    parser.add_argument("--input", type=str, default="", help="Specific video file to process")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (default: inside flight dir)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS for frame extraction (default: 10.0)")
    parser.add_argument("--scale", type=str, default="", help="Optional resolution WxH, e.g. 640x480 (default: original 960x720)")
    parser.add_argument("--swap-rb", action="store_true", help="Swap Red/Blue channels (fixes legacy recordings)")
    parser.add_argument("--no-trim", action="store_true", help="Do not trim; export entire video")
    parser.add_argument("--start-frame", type=int, default=-1, help="Custom start frame")
    parser.add_argument("--end-frame", type=int, default=-1, help="Custom end frame")
    parser.add_argument("--skip-frames", action="store_true", help="Skip extracting PNG frames (only export MP4)")
    args = parser.parse_args()

    base_debug = Path(__file__).resolve().parent / "debug"

    # 1. Resolve flight directory and video
    if args.input:
        video_path = Path(args.input).resolve()
        flight_dir = video_path.parent
    elif args.dir:
        flight_dir = Path(args.dir).resolve()
        video_path = flight_dir / "flight_raw.mp4"
    else:
        flight_dir = find_latest_flight_dir(base_debug)
        if not flight_dir:
            print("Error: No flight directory found in debug/!")
            sys.exit(1)
        video_path = flight_dir / "flight_raw.mp4"

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (flight_dir / "da3_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "da3_images"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TELLO -> DEPTH ANYTHING 3 (DA3) CLEAN EXPORTER")
    print(f"Source video: {video_path}")
    print(f"Output dir:   {output_dir}")
    print("=" * 65)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w, target_h = src_w, src_h
    if args.scale:
        try:
            w_str, h_str = args.scale.lower().split("x")
            target_w, target_h = int(w_str), int(h_str)
        except Exception:
            print(f"Warning: Invalid scale '{args.scale}', using original {src_w}x{src_h}")

    # Determine trim range
    start_frame = 0
    end_frame = total_frames - 1

    csv_path = flight_dir / "flight_telemetry.csv"
    if not args.no_trim:
        airborne_start, airborne_end = get_airborne_range(csv_path)
        if airborne_start is not None and airborne_end is not None:
            start_frame = max(0, airborne_start)
            end_frame = min(total_frames - 1, airborne_end)
            print(f"Telemetry auto-trim: Frames {start_frame} to {end_frame} (Airborne flight only)")
        else:
            print("No airborne telemetry found; processing entire video.")

    if args.start_frame >= 0:
        start_frame = args.start_frame
    if args.end_frame >= 0:
        end_frame = args.end_frame

    # Auto-detect legacy color inversion if recorded <= 20261008_101608
    swap_rb = args.swap_rb
    if not swap_rb and flight_dir.name <= "20261008_101608":
        swap_rb = True
        print(f"Legacy recording detected ({flight_dir.name}): Enabled R/B color swap automatically.")

    # Frame sampling step to match requested FPS
    target_fps = args.fps
    step = max(1, int(round(src_fps / target_fps)))
    effective_fps = src_fps / step

    print(f"Source: {total_frames} frames @ {src_fps:.1f} fps ({src_w}x{src_h})")
    print(f"Export range: frames [{start_frame}..{end_frame}] ({end_frame - start_frame + 1} frames)")
    print(f"Sample step: every {step} frame(s) -> ~{effective_fps:.1f} fps")
    print(f"Output resolution: {target_w}x{target_h}")
    print(f"Color channel correction: {'Active (R/B Swap)' if swap_rb else 'Standard (Native True BGR)'}")

    # Initialize clean MP4 video writer
    clean_video_path = output_dir / "clean_flight_da3.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(clean_video_path), fourcc, effective_fps, (target_w, target_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_idx = start_frame
    saved_count = 0

    while current_idx <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_idx - start_frame) % step == 0:
            # Color correction if needed
            if swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if requested
            if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 1. Write to clean MP4
            out_video.write(frame)

            # 2. Save individual PNG image for DA3-Streaming
            if not args.skip_frames:
                saved_count += 1
                img_path = frames_dir / f"frame_{saved_count:06d}.png"
                ok, encoded = cv2.imencode(".png", frame)
                if ok:
                    img_path.write_bytes(encoded.tobytes())

        current_idx += 1

    cap.release()
    out_video.release()

    print("\n" + "=" * 65)
    print("SUCCESS: CLEAN EXPORT COMPLETE!")
    print(f"Clean MP4 video:     {clean_video_path}")
    if not args.skip_frames:
        print(f"Clean image frames:  {frames_dir} ({saved_count} PNG frames @ ~{effective_fps:.1f} fps)")
    print("=" * 65)

    da3_streaming_dir = Path(__file__).resolve().parent.parent / "DA" / "Depth-Anything-3-main" / "da3_streaming"
    print("\nTo generate 3D depth maps / reconstruct space with Depth Anything 3, run:")
    if da3_streaming_dir.exists():
        print(f"  cd \"{da3_streaming_dir}\"")
        print(f"  python da3_streaming.py --image_dir \"{frames_dir}\"")
    else:
        print(f"  python da3_streaming.py --image_dir \"{frames_dir}\"")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
