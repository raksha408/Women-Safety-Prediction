import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
df = pd.read_csv("safety_data.csv")

# ✅ Ensure these match the original training columns
categorical_columns = [
    "Area", "Zone", "Time", 
    "People.Frequency", "Is.Police_Station", "Is.Bar", 
    "Tier", "Residence.Level"
]

# ✅ Fit encoder with ALL categorical columns
encoder = OrdinalEncoder()
encoder.fit(df[categorical_columns])

# ✅ Save encoder
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("✅ Encoder successfully saved as 'label_encoders.pkl'.")
