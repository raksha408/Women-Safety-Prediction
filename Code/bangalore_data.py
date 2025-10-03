import pandas as pd

# Load the Bangalore dataset
dataset_path = r"D:\Society\Programs\MachineLearning-WomenSafety-master\Banglore.csv"
df = pd.read_csv(dataset_path)

# Mapping the columns to match safety.csv format
df_transformed = pd.DataFrame()

# Extracting 'Area' and 'Zone' from 'Place'
df_transformed["Area"] = df["Place"].apply(lambda x: x.split(",")[-1].strip() if pd.notna(x) else "Unknown")
df_transformed["Zone"] = df["Police Station"]

# Converting 'Time' into categories
def categorize_time(time):
    if pd.isna(time) or not isinstance(time, str):
        return "Afternoon"  # Default category if time is missing or invalid
    
    parts = time.split(":")
    if len(parts) < 2 or not parts[0].isdigit():
        return "Afternoon"  # Default if time format is incorrect
    
    hour = int(parts[0])
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df_transformed["Time"] = df["Time"].apply(categorize_time)

# Mapping 'People.Frequency' based on crime occurrence
df_transformed["People.Frequency"] = "Medium"

# Assigning 'Yes' if a Police Station exists in the given zone
df_transformed["Is.Police_Station"] = "Yes"

# Checking if crime type relates to bars (Assuming crimes at night near bars are risky)
df_transformed["Is.Bar"] = df["Time"].apply(lambda x: "Yes" if x == "Night" else "No")

# Assigning 'Tier' based on Area (Assumption: You may refine this based on real classifications)
df_transformed["Tier"] = "Middle"

# Assigning 'Residence.Level' (Assumption: Higher crime areas might have different residence levels)
df_transformed["Residence.Level"] = "Medium"

# Mapping crimes to safety status
crime_risk = {
    "Murder": "Unsafe",
    "Robbery": "Unsafe",
    "Assault": "Unsafe",
    "Theft": "Safe",
}

df_transformed["Class"] = df["Type"].apply(lambda x: crime_risk.get(x, "Safe"))

# Save the transformed dataset
output_path_transformed = r"D:\Society\Programs\MachineLearning-WomenSafety-master\bangalore_safety.csv"
df_transformed.to_csv(output_path_transformed, index=False)
print(f"✅ Saved transformed dataset: {output_path_transformed}")

# Encode categorical values into numbers for numeric.csv
df_numeric = df_transformed.copy()

encoding_map = {
    "Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3,
    "Low": 0, "Medium": 1, "High": 2,
    "Yes": 1, "No": 0,
    "Middle": 1, "Outer": 2,
    "Safe": 1, "Unsafe": 0
}

for col in ["Time", "People.Frequency", "Is.Police_Station", "Is.Bar", "Tier", "Residence.Level", "Class"]:
    df_numeric[col] = df_numeric[col].map(encoding_map)

# Save numeric dataset
output_path_numeric = r"D:\Society\Programs\MachineLearning-WomenSafety-master\bangalore_numeric.csv"
df_numeric.to_csv(output_path_numeric, index=False)
print(f"✅ Saved numeric dataset: {output_path_numeric}")
