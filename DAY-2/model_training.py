import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("day2_sample.csv")
df.columns = df.columns.str.strip()

x = df[["temperature", "pressure", "vibration", "current", "rpm"]]
y = df["state"].str.strip()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale")
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("SVM Model Training Completed")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "svm_sensor_model.pkl")
print("Model saved successfully")

print("\nEnter sensor readings:")

temperature = float(input("Temperature: "))
pressure = float(input("Pressure: "))
vibration = float(input("Vibration: "))
current = float(input("Current: "))
rpm = float(input("RPM: "))

data = pd.DataFrame([{
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration,
    "current": current,
    "rpm": rpm
}])

print("Predicted Machine State:", model.predict(data)[0])