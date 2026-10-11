"""Offline Flight Test & Room Geometry Benchmarking Harness with Depth Anything V2 (DA2).

Runs offline testing on recorded DJI Tello flight video datasets (debug/<session>),
evaluates 3D room geometry, detects passages & doors with DA2, generates flight commands,
compares them against recorded drone telemetry, and exports an annotated evaluation video and report.
"""

import argparse
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np

from da2_depth import DA2DepthEstimator
from room_geometry_da2 import RoomGeometryEvaluator
from trajectory_planner import FlightTrajectoryPlanner, render_bev_minimap, draw_ar_perspective_path
from room_mapper import RoomOccupancyMapper


def find_latest_flight_dir(base_debug_dir: Path) -> Path:
    """Find the most recent flight directory with video recordings."""
    if not base_debug_dir.exists():
        return None
    candidates = []
    for p in base_debug_dir.iterdir():
        if p.is_dir() and ((p / "flight_raw.mp4").exists() or (p / "flight_airborne_clean.mp4").exists()):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_telemetry(csv_path: Path) -> dict:
    """Load flight telemetry indexed by frame_idx."""
    telemetry = {}
    if not csv_path.exists():
        return telemetry
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("frame_idx", 0))
                telemetry[idx] = row
            except ValueError:
                continue
    return telemetry


def render_evaluation_hud(frame, norm_depth, depth_colormap, eval_res, telem_row, frame_idx, total_frames, fps, infer_ms, planner=None, mapper=None):
    """Render side-by-side annotated visualization of camera feed, DA2 depth, and trajectory planner."""
    fh, fw = frame.shape[:2]
    vis_frame = frame.copy()
    vis_depth = depth_colormap.copy()

    state = eval_res["state"]
    cmds = eval_res["commands"]
    dl, dc, dr = eval_res["sectors"]
    dc_max = eval_res["center_max"]
    bias = eval_res["bias"]
    door = eval_res["door_info"]
    passages = eval_res["passages"]
    is_flat = eval_res["is_flat_wall"]

    # State color coding
    STATE_COLORS = {
        "OPEN_DOORWAY": (0, 255, 0),
        "OPEN_CORRIDOR": (0, 220, 100),
        "CLEAR_AISLE": (0, 255, 0),
        "CAUTION": (0, 200, 255),
        "BLOCKED_AHEAD": (0, 0, 255),
        "CLOSED_DOOR": (0, 0, 255),
        "DOORWAY_BLOCKED": (0, 140, 255),
        "FLAT_WALL_BRAKE": (0, 50, 255),
    }
    state_color = STATE_COLORS.get(state, (200, 200, 200))

    # 0. Flight Trajectory Comparison (AR 3D Path Corridor + Top-Down Radar Mini-Map)
    if planner is not None:
        old_path, new_path, polar_obs = planner.plan_trajectories(norm_depth, (dl, dc, dr), eval_res, telem_row)
        # AR Perspective 3D Path Ribbon on camera frame
        draw_ar_perspective_path(vis_frame, new_path, old_path)
        # Top-down BEV Mini-Map (Radar)
        bev_map = render_bev_minimap(old_path, new_path, polar_obs, state, size=(250, 250))
        bh, bw = bev_map.shape[:2]
        bx, by = fw - bw - 12, 68
        if by + bh < fh and bx >= 0:
            vis_frame[by:by + bh, bx:bx + bw] = bev_map

    # Top overlay bar on camera feed
    cv2.rectangle(vis_frame, (0, 0), (fw, 60), (20, 20, 20), -1)
    cv2.putText(vis_frame, f"DA2 EVAL [Frame {frame_idx}/{total_frames}]", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(vis_frame, f"State: {state}", (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

    # Telemetry comparison if available
    t_state = telem_row.get("state", "N/A") if telem_row else "N/A"
    t_fb = telem_row.get("fb", "0") if telem_row else "0"
    t_yaw = telem_row.get("yaw", "0") if telem_row else "0"
    cv2.putText(vis_frame, f"Drone Rec: {t_state[:28]}", (280, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(vis_frame, f"Rec FB: {t_fb} | Yaw: {t_yaw}", (280, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Draw geometric passage on frame if detected
    if passages:
        p = passages[0]
        px1, px2 = int(p["x1"] * (fw / norm_depth.shape[1])), int(p["x2"] * (fw / norm_depth.shape[1]))
        cv2.line(vis_frame, (px1, int(fh * 0.25)), (px1, int(fh * 0.75)), (0, 255, 255), 2)
        cv2.line(vis_frame, (px2, int(fh * 0.25)), (px2, int(fh * 0.75)), (0, 255, 255), 2)
        pcx = (px1 + px2) // 2
        pcy = int(fh * 0.45)
        cv2.drawMarker(vis_frame, (pcx, pcy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis_frame, f"Aperture Step:{p['depth_step']:.0f}", (px1, int(fh * 0.23)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Draw door box if detected
    if door and door.get("box"):
        bx, by, bw, bh = door["box"]
        d_color = (0, 255, 0) if door["passable"] else (0, 0, 255)
        cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), d_color, 2)
        tag = f"{door['status']} (conf {door.get('conf', 0):.2f}, step {door.get('depth_step', 0):.0f})"
        cv2.putText(vis_frame, tag, (bx, max(20, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, d_color, 2)

    # Bottom command bar on camera feed
    banner_y = fh - 65
    cv2.rectangle(vis_frame, (0, banner_y), (fw, fh), (20, 20, 20), -1)
    cmd_str = f"DA2 COMMANDS -> Fwd: {cmds['fb']:+d} | Strafe: {cmds['lr']:+d} | Yaw: {cmds['yaw']:+d} | Up: {cmds['ud']:+d}"
    cv2.putText(vis_frame, cmd_str, (15, banner_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 2)
    tex = eval_res.get("tex_metrics", {})
    sub_str = f"Sectors: L={dl:.0f} C={dc:.0f} R={dr:.0f} (Max={dc_max:.0f}) | Bias={bias:+.0f} | Edge={tex.get('edge', 0):.1f}%"
    cv2.putText(vis_frame, sub_str, (15, banner_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Annotate Depth Map
    dh, dw = vis_depth.shape[:2]
    sw = dw // 3
    # Vertical sector guidelines
    cv2.line(vis_depth, (sw, 0), (sw, dh), (80, 80, 80), 1)
    cv2.line(vis_depth, (2 * sw, 0), (2 * sw, dh), (80, 80, 80), 1)

    # Forward Flight Window from Calibration Config (Eye-level, Floor Excluded)
    from flight_window_config import load_flight_window_config, get_pixel_bounds
    cfg = getattr(planner, "window_cfg", None) or load_flight_window_config()
    box_x1, box_y1, box_x2, box_y2 = get_pixel_bounds(cfg, dw, dh)
    c_box_color = (0, 0, 255) if dc >= 165 else ((0, 200, 255) if dc >= 135 else (0, 255, 0))

    # Dim floor area below flight window to visually confirm floor exclusion
    if box_y2 < dh:
        floor_roi = vis_depth[box_y2:dh, 0:dw]
        dimmed = cv2.addWeighted(np.zeros_like(floor_roi), 0.55, floor_roi, 0.45, 0)
        vis_depth[box_y2:dh, 0:dw] = dimmed
        cv2.line(vis_depth, (0, box_y2), (dw, box_y2), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(vis_depth, "[FLOOR EXCLUDED]", (dw // 2 - 62, dh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # Highlight forward flight window
    cv2.rectangle(vis_depth, (box_x1, box_y1), (box_x2, box_y2), c_box_color, 2)
    win_label = f"FLIGHT WINDOW [{cfg['y_min_pct']}%..{cfg['y_max_pct']}%]"
    cv2.putText(vis_depth, win_label, (box_x1 + 4, box_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_box_color, 1)

    # Top overlay on depth map
    cv2.rectangle(vis_depth, (0, 0), (dw, 55), (15, 15, 15), -1)
    cv2.putText(vis_depth, f"DEPTH ANYTHING V2 ({infer_ms:.1f}ms)", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(vis_depth, f"L:{dl:.0f}  C:{dc:.0f} (max={dc_max:.0f})  R:{dr:.0f}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, c_box_color, 2)

    # Resize depth to match frame height for side-by-side view
    vis_depth_resized = cv2.resize(vis_depth, (fw, fh))

    # 1. Global Room Occupancy Grid & Max-Depth Frontier Map overlay
    if mapper is not None:
        dt = 1.0 / max(5.0, fps)
        mapper.update_from_telemetry(telem_row, dt=dt)
        rays = mapper.extract_polar_rays(norm_depth, win_cfg=cfg)
        mapper.update_map(rays)
        mapper.find_deepest_frontier_corridor(rays)
        map_hud = mapper.render_map_hud(hud_width=250, hud_height=250)
        mh, mw = map_hud.shape[:2]
        mx, my = fw - mw - 12, 68
        if my + mh < fh and mx >= 0:
            vis_depth_resized[my:my + mh, mx:mx + mw] = map_hud

    combined = np.hstack([vis_frame, vis_depth_resized])
    return combined


def run_offline_flight_test(
    flight_dir: Path,
    frame_step: int = 1,
    max_frames: int = None,
    export_video: bool = True,
    use_yolo: bool = True,
    video_filename: str = None,
):
    """Run complete offline geometry & navigation evaluation on a flight session."""
    flight_dir = Path(flight_dir).resolve()
    print("=" * 70)
    print("TELLO INDOOR GEOMETRY & NAVIGATION BENCHMARK (DEPTH ANYTHING V2)")
    print(f"Flight Directory: {flight_dir}")

    # Find video
    if video_filename:
        video_path = flight_dir / video_filename
    else:
        video_path = flight_dir / "flight_raw.mp4"
        if not video_path.exists():
            video_path = flight_dir / "flight_airborne_clean.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found in {flight_dir}")

    csv_path = flight_dir / "flight_telemetry.csv"
    if not csv_path.exists():
        csv_path = flight_dir / "commands.csv"

    print(f"Video Source:     {video_path.name}")
    print(f"Telemetry File:   {csv_path.name if csv_path.exists() else 'None'}")
    print("=" * 70)

    telemetry = load_telemetry(csv_path)

    # Initialize Depth Anything V2
    da2 = DA2DepthEstimator(input_size=252)

    # Initialize YOLO if requested
    yolo_model = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            yolo_weights = Path("yolov8s-worldv2.pt")
            if yolo_weights.exists():
                print("Loading YOLO-World v2 for semantic doorways...")
                yolo_model = YOLO(str(yolo_weights))
                yolo_model.set_classes(["door", "door frame", "doorway", "person", "wall"])
                print("YOLO-World ready!")
        except Exception as e:
            print(f"YOLO loading skipped ({e}), proceeding with pure geometry mode.")

    evaluator = RoomGeometryEvaluator(yolo_detector=yolo_model)
    planner = FlightTrajectoryPlanner()
    mapper = RoomOccupancyMapper()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = flight_dir / "da2_geometry_eval.mp4"
    writer = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps / max(1, frame_step), (fw * 2, fh))

    frame_idx = 0
    processed_count = 0
    state_counts = {}
    latencies = []
    evaluation_log = []

    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % frame_step != 0:
            continue

        if max_frames and processed_count >= max_frames:
            break

        # DA2 depth prediction
        norm_depth, depth_colormap, sectors, center_max, infer_ms = da2.predict(frame)
        latencies.append(infer_ms)

        # Room geometry & command evaluation
        t_eval0 = time.perf_counter()
        eval_res = evaluator.evaluate(frame, norm_depth, sectors, center_max)
        eval_ms = (time.perf_counter() - t_eval0) * 1000

        state = eval_res["state"]
        state_counts[state] = state_counts.get(state, 0) + 1

        telem_row = telemetry.get(frame_idx, {})

        # Record event
        record = {
            "frame_idx": frame_idx,
            "state": state,
            "commands": eval_res["commands"],
            "sectors": [round(s, 1) for s in sectors],
            "center_max": round(center_max, 1),
            "bias": round(eval_res["bias"], 1),
            "door_status": eval_res["door_info"].get("status"),
            "door_passable": eval_res["door_info"].get("passable", False),
            "passages_count": len(eval_res["passages"]),
            "infer_ms": round(infer_ms, 2),
            "eval_ms": round(eval_ms, 2),
        }
        if telem_row:
            record["drone_recorded_state"] = telem_row.get("state")
            record["drone_recorded_fb"] = telem_row.get("fb")
            record["drone_recorded_yaw"] = telem_row.get("yaw")
        evaluation_log.append(record)

        # Render and write video frame
        if writer is not None:
            combined = render_evaluation_hud(
                frame, norm_depth, depth_colormap, eval_res, telem_row,
                frame_idx, total_frames, fps, infer_ms, planner=planner, mapper=mapper
            )
            writer.write(combined)

        processed_count += 1
        if processed_count % 30 == 0 or processed_count == total_frames:
            print(f"  Processed {processed_count}/{total_frames} frames ({processed_count / total_frames * 100:.1f}%) | "
                  f"DA2: {infer_ms:.1f}ms | State: {state}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nAnnotated evaluation video exported to: {out_video_path}")

    total_time = time.time() - t_start
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    # Summary Report
    summary = {
        "flight_dir": str(flight_dir),
        "video_file": video_path.name,
        "total_frames_processed": processed_count,
        "elapsed_seconds": round(total_time, 2),
        "avg_da2_latency_ms": round(avg_latency, 2),
        "avg_fps": round(processed_count / max(0.001, total_time), 1),
        "state_distribution": state_counts,
        "output_video": str(out_video_path) if export_video else None,
    }

    report_path = flight_dir / "da2_geometry_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": evaluation_log}, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY:")
    print(f"  Frames Processed:   {processed_count}")
    print(f"  Average DA2 Speed:  {avg_latency:.1f} ms ({summary['avg_fps']} FPS)")
    print(f"  State Breakdown:")
    for st, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / max(1, processed_count)) * 100
        print(f"    - {st:<18}: {count:>4} frames ({pct:5.1f}%)")
    print(f"  Report saved to:    {report_path}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline Room Geometry & Flight Test with Depth Anything V2")
    parser.add_argument("--flight", type=str, default="latest", help="Flight folder name in debug/ or full path")
    parser.add_argument("--step", type=int, default=1, help="Frame step/stride (default 1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process")
    parser.add_argument("--no-video", action="store_true", help="Skip exporting evaluation video")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO semantic doors (pure geometry only)")
    args = parser.parse_args()

    base_debug = Path(__file__).parent / "debug"
    if args.flight == "latest":
        flight_dir = find_latest_flight_dir(base_debug)
    else:
        cand = Path(args.flight)
        if cand.exists():
            flight_dir = cand
        else:
            flight_dir = base_debug / args.flight

    if not flight_dir or not flight_dir.exists():
        print(f"Error: Flight directory '{args.flight}' not found.")
        return

    run_offline_flight_test(
        flight_dir=flight_dir,
        frame_step=args.step,
        max_frames=args.max_frames,
        export_video=not args.no_video,
        use_yolo=not args.no_yolo,
    )


if __name__ == "__main__":
    main()
