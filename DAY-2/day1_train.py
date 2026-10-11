import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "day1_sample.csv"
MODEL_FILE = Path(__file__).resolve().parent / "svm_sensor_model.pkl"

data = pd.read_csv(CSV_FILE)

X = data[["temperature", "pressure", "vibration", "current", "rpm"]]
y = data["state"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", random_state=42)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_FILE)
print("Model saved to:", MODEL_FILE)

while True:
    print("\nEnter sensor readings (or type exit):")
    values = input("temperature pressure vibration current rpm: ")

    if values.lower() == "exit":
        break

    try:
        readings = list(map(float, values.split()))

        if len(readings) != 5:
            print("Enter exactly 5 values.")
            continue

        result = model.predict([readings])[0]
        print("Predicted state:", result)

    except ValueError:
        print("Invalid input. Enter numeric values.")
