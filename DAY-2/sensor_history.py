import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Save the CSV inside the DAY-2 folder
csv_file = Path(__file__).parent / "Sensor_history.csv"

# Generate 500 historical sensor readings
num_records = 500
current_time = datetime.now()

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # CSV column headings
    writer.writerow([
        "timestamp",
        "temperature",
        "pressure",
        "vibration",
        "rpm",
        "current",
        "state"
    ])

    for i in range(num_records):
        timestamp = current_time - timedelta(
            minutes=num_records - 1 - i
        )

        # Simulate machine operating states
        state = random.choices(
            ["RUNNING", "IDLE", "FAULT"],
            weights=[80, 15, 5],
            k=1
        )[0]

        if state == "RUNNING":
            temperature = round(random.uniform(40, 85), 2)
            pressure = round(random.uniform(2.0, 5.0), 2)
            vibration = round(random.uniform(0.5, 6.0), 2)
            rpm = random.randint(1000, 3000)
            current = round(random.uniform(5, 20), 2)

        elif state == "IDLE":
            temperature = round(random.uniform(25, 45), 2)
            pressure = round(random.uniform(0.5, 2.0), 2)
            vibration = round(random.uniform(0.0, 1.0), 2)
            rpm = random.randint(0, 200)
            current = round(random.uniform(0.2, 3.0), 2)

        else:  # FAULT
            temperature = round(random.uniform(85, 110), 2)
            pressure = round(random.uniform(0.0, 6.0), 2)
            vibration = round(random.uniform(6.0, 15.0), 2)
            rpm = random.randint(0, 3500)
            current = round(random.uniform(15, 30), 2)

        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            temperature,
            pressure,
            vibration,
            rpm,
            current,
            state
        ])

print(f"Generated {num_records} sensor records.")
print(f"CSV saved to: {csv_file.resolve()}")