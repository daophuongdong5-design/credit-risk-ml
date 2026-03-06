import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Generate transaction dataset
n = 20000

data = pd.DataFrame({
    "transaction_amount": np.random.randint(1,5000,n),
    "transactions_per_day": np.random.randint(1,30,n),
    "location_change": np.random.randint(0,2,n),
    "device_change": np.random.randint(0,2,n)
})

# Train anomaly detection
model = IsolationForest(contamination=0.02)

model.fit(data)

data["fraud_prediction"] = model.predict(data)

print(data.head())
