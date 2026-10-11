import random
import time
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "DAY-1" / "day1_log.txt"
CSV_FILE = BASE_DIR / "day1_sample.csv"

def generate_sensor_data():
    return {
        "temperature": round(random.uniform(55, 98), 2),
        "pressure": round(random.uniform(1, 6.5), 2),
        "vibration": round(random.uniform(1, 10), 2),
        "current": round(random.uniform(8, 29), 2),
        "rpm": random.randint(1400, 2200)
    }

RULES = [
    {"name": "OVERHEATING", "priority": 1,
     "condition": lambda r: r["temperature"] > 75,
     "action": "STOP_MACHINE"},

    {"name": "LOW_PRESSURE", "priority": 2,
     "condition": lambda r: r["pressure"] < 3.5,
     "action": "CHECK_PRESSURE"},

    {"name": "HIGH_VIBRATION", "priority": 3,
     "condition": lambda r: r["vibration"] > 6,
     "action": "CHECK_MACHINE"},

    {"name": "MOTOR_OVERLOAD", "priority": 4,
     "condition": lambda r: r["current"] > 20,
     "action": "REDUCE_LOAD"}
]

def evaluate_rules(reading):
    for rule in sorted(RULES, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NORMAL", "CONTINUE_NORMAL"

def take_action(action):
    actions = {
        "STOP_MACHINE": "Machine stopped due to high temperature.",
        "CHECK_PRESSURE": "Low pressure detected.",
        "CHECK_MACHINE": "Machine inspection required.",
        "REDUCE_LOAD": "Machine load reduced.",
        "CONTINUE_NORMAL": "Machine running normally."
    }
    return actions[action]

def log_decision(reading, state, action):
    with open(LOG_FILE, "a") as file:
        file.write(
            f"{time.ctime()} | {reading} | {state} | {action}\n"
        )

def save_csv(reading, state):
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writerow([
                "temperature", "pressure", "vibration",
                "current", "rpm", "state"
            ])

        writer.writerow([
            reading["temperature"],
            reading["pressure"],
            reading["vibration"],
            reading["current"],
            reading["rpm"],
            state
        ])

def main():
    for tick in range(1, 251):
        reading = generate_sensor_data()
        state, action = evaluate_rules(reading)
        outcome = take_action(action)

        print(f"[Tick {tick:03d}] SENSE -> {reading}")
        print(f"            STATE -> {state}")
        print(f"            ACT   -> {outcome}\n")

        log_decision(reading, state, action)
        save_csv(reading, state)

        time.sleep(0.2)

    print("CSV saved to:", CSV_FILE)
    print("Log saved to:", LOG_FILE)

if __name__ == "__main__":
    main()