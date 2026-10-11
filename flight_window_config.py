"""Flight Window Configuration & Calibration Manager.

Stores and loads the user-calibrated forward eye-level flight corridor window,
allowing interactive runtime adjustment of its size, position, and floor cutoff.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "flight_window_config.json"

DEFAULT_CONFIG = {
    "y_min_pct": 18,     # Top boundary (cuts off ceiling lights / top wall)
    "y_max_pct": 54,     # Bottom boundary (strictly cuts off the floor plane)
    "x_center_pct": 50,  # Center corridor X alignment (percentage of frame width)
    "width_pct": 34,     # Width of central corridor flight window (%)
}


def load_flight_window_config():
    """Load calibration config from disk or return defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "y_min_pct": int(data.get("y_min_pct", DEFAULT_CONFIG["y_min_pct"])),
                    "y_max_pct": int(data.get("y_max_pct", DEFAULT_CONFIG["y_max_pct"])),
                    "x_center_pct": int(data.get("x_center_pct", DEFAULT_CONFIG["x_center_pct"])),
                    "width_pct": int(data.get("width_pct", DEFAULT_CONFIG["width_pct"])),
                }
        except Exception as e:
            print(f"Warning: Failed to read {CONFIG_PATH} ({e}), using defaults.")
    return DEFAULT_CONFIG.copy()


def save_flight_window_config(cfg):
    """Save calibration config to disk."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved flight window calibration to {CONFIG_PATH}: {cfg}")
        return True
    except Exception as e:
        print(f"Error saving flight window config: {e}")
        return False


def get_pixel_bounds(cfg, width, height):
    """Convert percentage config to exact pixel rectangle bounds (x1, y1, x2, y2)."""
    y1 = int(height * (cfg["y_min_pct"] / 100.0))
    y2 = int(height * (cfg["y_max_pct"] / 100.0))

    cx = width * (cfg["x_center_pct"] / 100.0)
    w_px = width * (cfg["width_pct"] / 100.0)

    x1 = int(max(0, cx - w_px / 2.0))
    x2 = int(min(width - 1, cx + w_px / 2.0))
    y1 = int(max(0, y1))
    y2 = int(min(height - 1, max(y1 + 4, y2)))

    return x1, y1, x2, y2
