import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Load dataset filenames
datasets = {
    "Chennai": "safety_data.csv",
    "Bangalore": "bangalore_safety.csv",
    "Karnataka": "Karnataka_cleaned.csv"
}

# Load trained model
@st.cache_resource
def load_model(city):
    try:
        model_file = f"best_model_{datasets[city]}.pkl"
        with open(model_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model for {city}: {e}")
        return None

# Load encoder
@st.cache_resource
def load_encoder(city):
    try:
        encoder_file = f"ordinal_encoder_{datasets[city]}.pkl"
        with open(encoder_file, "rb") as f:
            encoder = pickle.load(f)
            encoder.handle_unknown = "use_encoded_value"
            encoder.unknown_value = -1
            return encoder
    except Exception as e:
        st.error(f"Error loading encoder for {city}: {e}")
        return None

# Load DataFrame
@st.cache_data
def load_data(city):
    return pd.read_csv(datasets[city])

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Women Safety Prediction", page_icon="👩‍🦺", layout="wide")

# Page Styling
st.markdown("""
    <style>
        header {visibility: hidden;}
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #FCE4EC, #F8BBD0);
        }
        .header-title {
            text-align: center;
            font-size: 42px;
            font-family: 'Georgia', serif;
            font-weight: bold;
            color: white;
            text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.5);
            background: linear-gradient(to right, #D81B60, #6A1B9A);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 3px 3px 20px rgba(0, 0, 0, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">Women Safety Prediction</div><hr>', unsafe_allow_html=True)

# Select City
st.subheader("🌍 Select City")
city = st.selectbox("", list(datasets.keys()))

# Load model, encoder, and data
model = load_model(city)
encoder = load_encoder(city)
df = load_data(city)

if model is None or encoder is None or df is None:
    st.stop()

# Select Area
st.subheader("📍 Select Area")
area_list = sorted(df["Area"].dropna().unique())
area = st.selectbox("Select Area", area_list)

# Select Zone
zone_list = sorted(df[df["Area"] == area]["Zone"].dropna().unique())
zone = st.selectbox("🗺 Select Zone", zone_list)

# Select Time
time_list = sorted(df[(df["Area"] == area) & (df["Zone"] == zone)]["Time"].dropna().unique())
time = st.selectbox("⏰ Select Time of Visit", time_list)

# Predict Button
if st.button("🔍 Predict Safety"):
    with st.spinner("Predicting..."):
        input_data = pd.DataFrame([[area, zone, time]], columns=["Area", "Zone", "Time"])
        details = df[(df["Area"] == area) & (df["Zone"] == zone) & (df["Time"] == time)]

        if not details.empty:
            details = details.iloc[0]
            safety_status = details["Class"]
        else:
            input_data_encoded = encoder.transform(input_data)
            prediction = model.predict(input_data_encoded)[0]
            safety_status = "Safe" if prediction == 1 else "Unsafe"

        people_frequency = details.get("People.Frequency", "Unknown")
        is_police_station = details.get("Is.Police_Station", "Unknown")
        is_bar = details.get("Is.Bar", "Unknown")
        tier = details.get("Tier", "Unknown")
        residence_level = details.get("Residence.Level", "Unknown")

        # Safety Status
        color = "🟢 SAFE" if safety_status == "Safe" else "🔴 UNSAFE"

        # Result Display
        st.markdown(f"<h2 style='text-align:center;'>{color}</h2>", unsafe_allow_html=True)

        # Prediction Details
        st.markdown("""
            <hr>
            <h4 style='color:#6A1B9A;'>Prediction Details</h4>
        """, unsafe_allow_html=True)
        st.write(f"**City:** {city}")
        st.write(f"**Area:** {area}")
        st.write(f"**Zone:** {zone}")
        st.write(f"**Time:** {time}")
        st.write(f"**People Frequency:** {people_frequency}")
        st.write(f"**Police Station Nearby:** {is_police_station}")
        st.write(f"**Bar Nearby:** {is_bar}")
        st.write(f"**Tier:** {tier}")
        st.write(f"**Residence Level:** {residence_level}")
