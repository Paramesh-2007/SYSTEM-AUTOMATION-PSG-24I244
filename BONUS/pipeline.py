"""
BONUS CHALLENGE — Step 4
Robotic Arm Joint Health Monitoring System
Main Pipeline — All 5 Modules Integrated

Modules:
  1. safety_stop()      — Rule-based immediate threshold check (no ML)
  2. predictive_wear()  — ML model prediction (RandomForest)
  3. log_signal()       — NLP keyword scanner on maintenance log
  4. spoof_defense()    — Anomaly / spoofed-sensor detection
  5. final_decision()   — Combines all signals → one of 4 outcomes

Final decisions:
  SAFE                  — all clear
  MONITOR               — watch closely, no action yet
  SCHEDULE_MAINTENANCE  — plan a service window
  EMERGENCY_STOP        — halt immediately

Run:
  python BONUS/pipeline.py
"""

import re
import warnings
import joblib
from pathlib import Path

# Suppress sklearn's harmless feature-name warning (we pass lists, not DataFrames)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent
MODEL_FILE = BASE_DIR / "joint_model.pkl"
LOG_FILE   = BASE_DIR / "maintenance_log.txt"

# ── Module 1: Safety threshold values ─────────────────────────────────────────
#    These are HARD limits derived from the arm's servo motor spec sheets.
#    They must fire instantly, with NO dependency on any ML model.
#    Chosen rationale:
#      - Torque    > 85 Nm  : exceeds rated peak torque; risks gear strip
#      - Temperature > 90°C : motor winding insulation degrades rapidly above 90°C
#      - Vibration  > 8.5   : ISO 10816 "danger zone" for rotating machinery
SAFETY_THRESHOLDS = {
    "torque":      85.0,    # Nm
    "temperature": 90.0,    # °C
    "vibration":   8.5,     # mm/s
}

# ── Module 3: Maintenance log concern keywords ────────────────────────────────
CONCERN_KEYWORDS = [
    "grinding", "stiff", "stuck", "overheating", "hot", "loud", "noise",
    "loose", "vibrating", "shaking", "leaking", "error", "fault",
    "unusual", "weird", "strange", "worn", "damage", "rattling",
    "flickering", "shaky", "high-pitched", "imbalance", "strip-down",
]

# ── Module 4: Physical plausibility limits ────────────────────────────────────
#    Values outside these ranges are physically impossible for this arm.
PHYSICS_LIMITS = {
    "torque":      (0.0,   200.0),   # Nm
    "temperature": (0.0,   150.0),   # °C
    "vibration":   (0.0,   25.0),    # mm/s
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — IMMEDIATE SAFETY STOP (rule-based, no ML)
# ═══════════════════════════════════════════════════════════════════════════════

def safety_stop(reading: dict) -> tuple[bool, str]:
    """
    Check if ANY single sensor value crosses a hard safety threshold.

    Returns
    -------
    (triggered: bool, reason: str)
    """
    for sensor, limit in SAFETY_THRESHOLDS.items():
        value = reading.get(sensor, 0)
        if value > limit:
            return True, (
                f"{sensor.upper()} = {value} "
                f"exceeds limit {limit}"
            )
    return False, "all within safe thresholds"


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — PREDICTIVE WEAR DETECTION (ML model)
# ═══════════════════════════════════════════════════════════════════════════════

_model = None  # lazy-loaded singleton

def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_FILE)
    return _model


def predictive_wear(reading: dict) -> str:
    """
    Use the trained RandomForest to predict the health label.

    Returns one of: SAFE / MONITOR / SCHEDULE_MAINTENANCE / EMERGENCY_STOP
    """
    model = _load_model()
    features = [[
        reading["torque"],
        reading["temperature"],
        reading["vibration"],
    ]]
    return model.predict(features)[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — MAINTENANCE LOG NLP SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def log_signal(arm_id: str, joint: str) -> tuple[str, list]:
    """
    Scan the maintenance log for concern keywords related to this arm+joint.

    Returns
    -------
    (signal: "concern" | "clear", matched_keywords: list)
    """
    log_text = LOG_FILE.read_text(encoding="utf-8")
    pattern  = re.compile(
        rf"\[.*?\]\s+{re.escape(arm_id)}\s+{re.escape(joint)}\s*::(.*)",
        re.IGNORECASE
    )
    matches       = pattern.findall(log_text)
    found_keywords = []

    for entry_text in matches:
        entry_lower = entry_text.lower()
        for kw in CONCERN_KEYWORDS:
            if kw in entry_lower and kw not in found_keywords:
                found_keywords.append(kw)

    if found_keywords:
        return "concern", found_keywords
    return "clear", []


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — SPOOFED-SENSOR DEFENSE
# ═══════════════════════════════════════════════════════════════════════════════

def spoof_defense(reading: dict, previous_reading: dict | None) -> tuple[bool, str]:
    """
    Three-layer spoofed-sensor detection:

    Layer A — Out-of-physics range:
      Any value outside absolute physical limits is impossible →
      sensor hardware failure or spoofed feed.

    Layer B — Cross-sensor inconsistency:
      A motor under high torque load MUST generate heat.
      If torque > 70 Nm but temperature < 30°C, the reading is suspicious
      (a cold motor physically cannot produce that torque sustainably).

    Layer C — Frozen sensor:
      If ALL three values are identical to the previous tick,
      the sensor feed is likely frozen/repeated — classic replay attack.

    Returns
    -------
    (detected: bool, reason: str)
    """
    # Layer A: physics range
    for sensor, (lo, hi) in PHYSICS_LIMITS.items():
        val = reading.get(sensor, 0)
        if not (lo <= val <= hi):
            return True, (
                f"LAYER-A: {sensor} = {val} outside physical range "
                f"[{lo}, {hi}]"
            )

    # Layer B: cross-sensor consistency
    if reading["torque"] > 70.0 and reading["temperature"] < 30.0:
        return True, (
            f"LAYER-B: torque={reading['torque']} Nm but "
            f"temperature={reading['temperature']}°C — "
            f"physically inconsistent (cold motor, max torque)"
        )

    # Layer C: frozen / replay sensor
    if previous_reading is not None:
        if (reading["torque"]      == previous_reading["torque"] and
            reading["temperature"] == previous_reading["temperature"] and
            reading["vibration"]   == previous_reading["vibration"]):
            return True, (
                "LAYER-C: all sensor values identical to previous tick "
                "— possible frozen/replayed sensor feed"
            )

    return False, "no spoof detected"


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — FINAL DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def final_decision(
    safety_triggered: bool,
    spoof_detected:   bool,
    ml_prediction:    str,
    log_sig:          str,
) -> str:
    """
    Combine all four signals into ONE final decision per arm.

    Priority order (highest first):
      1. Safety stop triggered          → EMERGENCY_STOP (non-negotiable)
      2. Spoof detected                 → EMERGENCY_STOP (untrusted data)
      3. ML predicts EMERGENCY_STOP     → EMERGENCY_STOP
      4. ML predicts SCHEDULE_MAINTENANCE
         OR log has concern keyword     → SCHEDULE_MAINTENANCE
      5. ML predicts MONITOR            → MONITOR
      6. All clear                      → SAFE
    """
    if safety_triggered:
        return "EMERGENCY_STOP"
    if spoof_detected:
        return "EMERGENCY_STOP"
    if ml_prediction == "EMERGENCY_STOP":
        return "EMERGENCY_STOP"
    if ml_prediction == "SCHEDULE_MAINTENANCE" or log_sig == "concern":
        return "SCHEDULE_MAINTENANCE"
    if ml_prediction == "MONITOR":
        return "MONITOR"
    return "SAFE"


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO — Run pipeline on 6 arms
# ═══════════════════════════════════════════════════════════════════════════════

DECISION_COLORS = {
    "SAFE":                  "[ SAFE                 ]",
    "MONITOR":               "[ MONITOR              ]",
    "SCHEDULE_MAINTENANCE":  "[ SCHEDULE_MAINTENANCE ]",
    "EMERGENCY_STOP":        "[ !!! EMERGENCY_STOP ! ]",
}

# Each test case is designed to exercise one of the 4 final outcomes
DEMO_ARMS = [
    # ── Normal operation — should produce SAFE ────────────────────────────────
    {
        "arm_id":   "ARM-05",
        "joint":    "shoulder",
        "reading":  {"torque": 28.5, "temperature": 52.0, "vibration": 1.8},
        "prev":     {"torque": 27.0, "temperature": 51.0, "vibration": 1.9},
        "scenario": "Normal operation",
    },
    # ── Borderline values — should produce MONITOR ────────────────────────────
    #    ARM-99 has NO maintenance log entries, so log_signal = "clear"
    {
        "arm_id":   "ARM-99",
        "joint":    "elbow",
        "reading":  {"torque": 67.0, "temperature": 75.0, "vibration": 5.2},
        "prev":     {"torque": 64.0, "temperature": 73.0, "vibration": 4.9},
        "scenario": "Borderline / elevated readings — no log concern",
    },
    # ── Log has concern + moderate sensors — SCHEDULE_MAINTENANCE ─────────────
    {
        "arm_id":   "ARM-01",
        "joint":    "shoulder",
        "reading":  {"torque": 55.0, "temperature": 65.0, "vibration": 3.5},
        "prev":     {"torque": 53.0, "temperature": 63.0, "vibration": 3.2},
        "scenario": "Log concern (grinding note) — moderate sensors",
    },
    # ── ML predicts SCHEDULE_MAINTENANCE (high wear range) ────────────────────
    {
        "arm_id":   "ARM-04",
        "joint":    "wrist",
        "reading":  {"torque": 80.0, "temperature": 86.0, "vibration": 7.8},
        "prev":     {"torque": 78.0, "temperature": 84.0, "vibration": 7.5},
        "scenario": "High wear zone — ML schedules maintenance",
    },
    # ── Safety threshold breached — EMERGENCY_STOP ────────────────────────────
    {
        "arm_id":   "ARM-02",
        "joint":    "wrist",
        "reading":  {"torque": 112.0, "temperature": 108.0, "vibration": 13.5},
        "prev":     {"torque": 88.0,  "temperature": 95.0,  "vibration": 9.0},
        "scenario": "Critical threshold crossed — immediate stop",
    },
    # ── Frozen sensor spoof — EMERGENCY_STOP ─────────────────────────────────
    {
        "arm_id":   "ARM-06",
        "joint":    "shoulder",
        "reading":  {"torque": 42.0, "temperature": 68.0, "vibration": 3.1},
        "prev":     {"torque": 42.0, "temperature": 68.0, "vibration": 3.1},
        "scenario": "Frozen/replayed sensor feed — spoof detected",
    },
]


def run_pipeline():
    divider    = "=" * 65
    subdiv     = "-" * 65
    all_results = []

    print(f"\n{divider}")
    print("  ROBOTIC ARM JOINT HEALTH MONITORING PIPELINE")
    print(f"{divider}\n")

    for i, arm in enumerate(DEMO_ARMS, 1):
        arm_id   = arm["arm_id"]
        joint    = arm["joint"]
        reading  = arm["reading"]
        prev     = arm["prev"]

        print(f"  Arm {i}/{len(DEMO_ARMS)} — {arm_id} [{joint.upper()}]")
        print(f"  Scenario : {arm['scenario']}")
        print(subdiv)
        print(f"  Sensors  : torque={reading['torque']} Nm  "
              f"temp={reading['temperature']}°C  "
              f"vibration={reading['vibration']} mm/s")
        print()

        # Module 1
        s_triggered, s_reason = safety_stop(reading)
        print(f"  [Module 1] Safety Stop  : {'TRIGGERED' if s_triggered else 'OK'}")
        print(f"             Detail       : {s_reason}")

        # Module 2
        ml_pred = predictive_wear(reading)
        print(f"  [Module 2] ML Prediction: {ml_pred}")

        # Module 3
        log_sig, keywords = log_signal(arm_id, joint)
        kw_str = ", ".join(keywords) if keywords else "none"
        print(f"  [Module 3] Log Signal   : {log_sig.upper()}  keywords=[{kw_str}]")

        # Module 4
        spoof, spoof_reason = spoof_defense(reading, prev)
        print(f"  [Module 4] Spoof Check  : {'DETECTED' if spoof else 'CLEAN'}")
        print(f"             Detail       : {spoof_reason}")

        # Module 5
        decision = final_decision(s_triggered, spoof, ml_pred, log_sig)
        banner   = DECISION_COLORS[decision]
        print()
        print(f"  {banner}")
        print(f"  Final Decision: {decision}")
        print()
        print(divider + "\n")

        all_results.append(decision)

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(all_results)

    print("  PIPELINE SUMMARY")
    print(subdiv)
    for decision in ["SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"]:
        mark = "YES" if counts[decision] > 0 else "NO "
        print(f"  [{mark}]  {decision}")
    print()

    all_4 = all(
        d in all_results
        for d in ["SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"]
    )
    if all_4:
        print("  All 4 required outcomes were demonstrated successfully.")
    else:
        missing = [d for d in
                   ["SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"]
                   if d not in all_results]
        print(f"  WARNING: missing outcomes: {missing}")
    print(divider)


if __name__ == "__main__":
    run_pipeline()
