
import random
import time

LOG_FILE = "day1_log.txt"

# STEP 1: Generate sensor data
def generate_sensor_data():
    return {
        "temperature": random.randint(50, 95),
        "pressure": random.randint(20, 100),
        "vibration": round(random.uniform(0, 10), 2),
    }

# STEP 2, 7, 8: Define automation rules
RULES = [
    {"name": "HIGH_TEMP", "priority": 1,
     "condition": lambda r: r["temperature"] > 80,
     "action": "STOP_MACHINE"},

    {"name": "HIGH_PRESSURE", "priority": 2,
     "condition": lambda r: r["pressure"] > 85,
     "action": "RELEASE_PRESSURE"},

    {"name": "HIGH_VIBRATION", "priority": 3,
     "condition": lambda r: r["vibration"] > 8,
     "action": "CHECK_MACHINE"},

]

# STEP 3: Evaluate rules
def evaluate_rules(reading, rules):
    for rule in sorted(rules, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"

# STEP 4: Take action
def take_action(action):
    actions = {
        "STOP_MACHINE": "Machine stopped due to high temperature.",
        "RELEASE_PRESSURE": "Pressure released.",
        "CHECK_MACHINE": "Machine inspection required.",
        "REDUCE_LOAD": "Machine load reduced.",
        "CONTINUE_NORMAL": "Machine running normally."
    }
    return actions.get(action, "Unknown action")

# STEP 5: Log decisions
def log_decision(reading, rule_name, action):
    with open(LOG_FILE, "a") as file:
        file.write(
            f"{time.ctime()} | {reading} | "
            f"{rule_name} | {action}\n"
        )

# STEP 6: Added current sensor in generate_sensor_data()

def main():
    for tick in range(1, 21):
        reading = generate_sensor_data()
        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)
        time.sleep(0.2)

if __name__ == "__main__":
    main()