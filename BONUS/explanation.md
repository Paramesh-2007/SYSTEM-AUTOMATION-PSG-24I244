# Robotic Arm Joint Health Monitoring System
## Written Explanation — Bonus Challenge Submission

---

## 1. Safety Thresholds (Module 1)

Module 1 is a **hard rule-based safety stop** that fires instantly, before any
ML model is consulted. This is by design: the challenge requires the safety
stop to work even if the model is completely wrong or unavailable.

### Chosen Thresholds

| Sensor      | Limit  | Unit  | Rationale |
|-------------|--------|-------|-----------|
| Torque      | > 85   | Nm    | Exceeds the rated peak torque of a typical 6-axis industrial servo arm joint. Operating above rated torque strips gears and causes catastrophic mechanical failure. |
| Temperature | > 90   | °C    | Motor winding insulation (class B/F rated) degrades rapidly above 90°C. Sustained operation above this point causes irreversible winding damage and fire risk. |
| Vibration   | > 8.5  | mm/s  | Per ISO 10816-3 (vibration severity for industrial machines), values above 7.1 mm/s indicate "danger" — 8.5 mm/s is a conservative upper bound before catastrophic bearing failure. |

### Why No Model Dependency

If a sensor is spoofed to report a safe value while the real torque is
destructively high, the model may be fooled. The safety check is a deterministic
rule applied directly to raw sensor values — it cannot be overridden by Module 2,
Module 3, or Module 4 results. EMERGENCY_STOP from Module 1 is always final.

---

## 2. How ML Prediction + Maintenance Log Combine (Modules 2 & 3)

### The ML Model (Module 2)

A **RandomForestClassifier** was trained on 103 labelled historical joint
readings. The features are `torque`, `temperature`, and `vibration`. The model
predicts one of four classes:

- `SAFE` — normal operation
- `MONITOR` — slight deviation, keep watching
- `SCHEDULE_MAINTENANCE` — wear detected, plan a service
- `EMERGENCY_STOP` — critical degradation predicted

The model captures **non-linear patterns and feature interactions** that a
rule-based system would miss. For example, moderate torque combined with
moderately elevated temperature AND slightly high vibration may predict
`SCHEDULE_MAINTENANCE` even if no single value crosses a hard threshold.

### The Maintenance Log (Module 3)

Technicians write free-form inspection notes. Module 3 scans the log for
**concern keywords** (e.g., "grinding", "stiff", "overheating", "loose",
"rattling", "shaking") matching the specific arm and joint being evaluated.

This is an **independent second signal** — it uses human observations, not
sensor numbers. A technician may notice a grinding sound *before* any sensor
value exceeds thresholds.

### How They Combine (Module 5)

```
if safety_stop_triggered:       → EMERGENCY_STOP   (always wins)
elif spoof_detected:            → EMERGENCY_STOP   (untrusted data)
elif ml_pred == EMERGENCY_STOP: → EMERGENCY_STOP
elif ml_pred == SCHEDULE_MAINTENANCE
     OR log_signal == "concern":→ SCHEDULE_MAINTENANCE
elif ml_pred == MONITOR:        → MONITOR
else:                           → SAFE
```

Key design decision: **the log signal can ESCALATE but cannot DE-ESCALATE**.
If the ML predicts `SAFE` but the technician log has a concern keyword, the
outcome becomes `SCHEDULE_MAINTENANCE`. The log cannot make a `MONITOR`
prediction become `SAFE`.

---

## 3. Spoofed-Sensor Defense (Module 4)

A spoofed-sensor attack feeds the system a fabricated sensor value — typically
a safe-looking value — to mask a real problem (e.g., make a dangerously hot
joint look cool).

Module 4 implements **three detection layers**:

### Layer A — Out-of-Physics Range Check

```
torque       must be in [0,   200] Nm
temperature  must be in [0,   150] °C
vibration    must be in [0,    25] mm/s
```

Any reading outside these absolute physical limits is **impossible** for this
hardware. A torque of -5 Nm or 999 Nm cannot be real — it indicates hardware
failure or an injected fake reading.

**Catches:** Simple spoofs that inject obviously out-of-range values.  
**Does NOT catch:** Carefully crafted spoofs within physical range.

### Layer B — Cross-Sensor Consistency Check

```
if torque > 70 Nm AND temperature < 30°C → suspicious
```

A motor producing 70+ Nm of torque is working hard. Working hard generates
heat. A cold motor (< 30°C) physically cannot sustain that torque for more
than a fraction of a second. If both conditions are true simultaneously, either
the temperature sensor is spoofed (reporting cold when the motor is hot) or the
torque sensor is spoofed (reporting high torque when the motor is idle).

**Catches:** Spoofs that replace temperature with a safe-looking cold value
while real torque is high.  
**Does NOT catch:** Spoofs that consistently adjust all sensors together in a
physically plausible but misleading way.

### Layer C — Frozen / Replay Attack Detection

```
if ALL values identical to previous tick → frozen sensor detected
```

A replay attack feeds the same safe reading repeatedly (e.g., recordings from
a quiet period). Real sensors always have micro-variation due to noise and
mechanical movement. If all three sensors report bit-for-bit identical values
across two consecutive ticks, the feed is almost certainly frozen or replayed.

**Catches:** Replay attacks and stuck sensor hardware.  
**Does NOT catch:** Spoofs that add tiny random noise to each replay to avoid
identical-value detection (a more sophisticated attack would require statistical
drift analysis across longer time windows, beyond this system's scope).

---

## 4. Four Final Outcomes — Example Runs

| Outcome | Trigger Condition | Example |
|---|---|---|
| `SAFE` | All modules clear | torque=28, temp=52, vibration=1.8, no log concern |
| `MONITOR` | ML predicts slight elevation | torque=67, temp=75, vibration=5.2 |
| `SCHEDULE_MAINTENANCE` | ML or log concern | torque=55, temp=65, vibration=3.5 + log has "grinding" note |
| `EMERGENCY_STOP` | Threshold crossed or spoof | torque=112, temp=108 OR frozen sensor values |
