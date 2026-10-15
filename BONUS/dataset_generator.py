"""
BONUS CHALLENGE — Step 1
Robotic Arm Joint Health Monitoring System
Dataset Generator

Generates a synthetic historical dataset of labelled joint readings
for the shoulder, elbow, and wrist joints of a 6-axis robotic arm.

Sensor columns:
  - torque      : joint torque in Newton-metres (Nm)
  - temperature : joint temperature in Celsius
  - vibration   : vibration amplitude in mm/s

Labels (4 classes):
  - SAFE                  : all readings within normal operating range
  - MONITOR               : slight deviation — keep watching
  - SCHEDULE_MAINTENANCE  : degradation detected — plan service
  - EMERGENCY_STOP        : critical threshold crossed — stop immediately

Run:
  python BONUS/dataset_generator.py
"""

import csv
import random
from pathlib import Path
from datetime import datetime, timedelta

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_FILE = Path(__file__).parent / "joint_health_dataset.csv"

# ── Joint names (6-axis arm has 3 key joints) ─────────────────────────────────
JOINTS = ["shoulder", "elbow", "wrist"]

# ── Sensor value ranges per class ─────────────────────────────────────────────
#    Thresholds chosen to reflect real servo-motor physics:
#      - Torque    : safe < 60 Nm  →  emergency > 85 Nm
#      - Temp      : safe < 70 °C  →  emergency > 90 °C
#      - Vibration : safe < 4 mm/s →  emergency > 8.5 mm/s

CLASS_RANGES = {
    "SAFE": {
        "torque":      (10.0, 60.0),
        "temperature": (25.0, 70.0),
        "vibration":   (0.2,  4.0),
    },
    "MONITOR": {
        "torque":      (60.0, 75.0),
        "temperature": (70.0, 82.0),
        "vibration":   (4.0,  6.5),
    },
    "SCHEDULE_MAINTENANCE": {
        "torque":      (75.0, 85.0),
        "temperature": (82.0, 90.0),
        "vibration":   (6.5,  8.5),
    },
    "EMERGENCY_STOP": {
        "torque":      (85.0, 130.0),
        "temperature": (90.0, 120.0),
        "vibration":   (8.5,  15.0),
    },
}

# ── Class distribution weights ─────────────────────────────────────────────────
CLASS_WEIGHTS = {
    "SAFE":                 42,
    "MONITOR":              25,
    "SCHEDULE_MAINTENANCE": 23,
    "EMERGENCY_STOP":       10,
}

TOTAL_RECORDS = 100  # ≥ 50 as required; using 100 for better model training


def generate_reading(label: str) -> dict:
    """Generate a single sensor reading consistent with the given label."""
    ranges = CLASS_RANGES[label]
    return {
        "torque":      round(random.uniform(*ranges["torque"]),      2),
        "temperature": round(random.uniform(*ranges["temperature"]), 2),
        "vibration":   round(random.uniform(*ranges["vibration"]),   2),
    }


def main():
    records = []
    base_time = datetime(2026, 10, 11, 6, 0, 0)

    # Build weighted class list
    class_pool = []
    for label, weight in CLASS_WEIGHTS.items():
        class_pool.extend([label] * weight)

    for i in range(TOTAL_RECORDS):
        timestamp = base_time + timedelta(minutes=i * 5)
        joint     = random.choice(JOINTS)
        label     = random.choice(class_pool)
        reading   = generate_reading(label)

        records.append({
            "timestamp":   timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "joint":       joint,
            "torque":      reading["torque"],
            "temperature": reading["temperature"],
            "vibration":   reading["vibration"],
            "label":       label,
        })

    # Guarantee at least 3 examples of EMERGENCY_STOP for stratified split
    for j in JOINTS:
        records.append({
            "timestamp":   (base_time + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "joint":       j,
            "torque":      round(random.uniform(85.0, 130.0), 2),
            "temperature": round(random.uniform(90.0, 120.0), 2),
            "vibration":   round(random.uniform(8.5, 15.0),   2),
            "label":       "EMERGENCY_STOP",
        })

    random.shuffle(records)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "joint", "torque",
                           "temperature", "vibration", "label"]
        )
        writer.writeheader()
        writer.writerows(records)

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import Counter
    label_counts = Counter(r["label"] for r in records)

    print("=" * 55)
    print("  DATASET GENERATOR — Complete")
    print("=" * 55)
    print(f"  Total records : {len(records)}")
    print(f"  Saved to      : {OUTPUT_FILE.resolve()}")
    print()
    print("  Label distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        bar = "#" * cnt
        print(f"    {lbl:<25} {cnt:>3}  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    main()
