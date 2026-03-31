import pandas as pd
import numpy as np
import heapq
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def generate_data(size=10000):
    np.random.seed(42) 
    temps = np.random.uniform(97.0, 105.0, size)
    pain = np.random.randint(1, 11, size)
    age = np.random.randint(1, 95, size)
    priority = []
    for t, p in zip(temps, pain):
        if t > 103 or (t > 101 and p > 8):
            priority.append(2)  # Emergency
        elif t > 100 or p > 5:
            priority.append(1)  # Urgent
        else:
            priority.append(0)  # Routine
    return pd.DataFrame({
        'Temp': temps,
        'Pain': pain,
        'Age': age,
        'Priority': priority
    })


print("Generating 10,000 patient records...")
df = generate_data()

X = df[['Temp', 'Pain', 'Age']]
y = df['Priority']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"ML Model Trained. Accuracy: {accuracy * 100:.2f}%")
patient_queue = []

def triage_patient(name, temp, pain, age):
    """Predicts priority and adds to the search queue."""
    # ML Prediction
    input_data = pd.DataFrame([[temp, pain, age]], columns=['Temp', 'Pain', 'Age'])
    pred_priority = int(model.predict(input_data)[0])
    timestamp = time.time()
    heapq.heappush(patient_queue, (-pred_priority, timestamp, name, pred_priority))
    print(f"[Check-in] {name}: Temp {temp}, Pain {pain}. Predicted Priority: {pred_priority}")

def see_next_patient():
    if not patient_queue:
        print("No patients waiting.")
        return
    _, _, name, level = heapq.heappop(patient_queue)
    status = ["ROUTINE", "URGENT", "EMERGENCY"][level]
    print(f"\n>>> DOCTOR IS SEEING: {name} | STATUS: {status}")

triage_patient("John Doe", 98.4, 2, 30)  # Routine
triage_patient("Matt Murdock", 99.6, 6, 35)  # Routine
triage_patient("Jane Smith", 104.2, 10, 25)  # Emergency
triage_patient("Mark Brown", 101.5, 6, 50)  # Urgent

see_next_patient()
see_next_patient()
see_next_patient()
see_next_patient()
