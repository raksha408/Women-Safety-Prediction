# Women Safety Prediction

The **Women Safety Prediction** project is a machine learning and data-driven application designed to assess the safety of locations based on multiple features such as area, time, people frequency, and nearby facilities. It leverages preprocessing pipelines, resampling techniques, and classification algorithms to predict whether a given context is *Safe* or *Unsafe*.  
The project also provides a **Streamlit-based user interface** that allows users to interactively test predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Folder Structure](#folder-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Dependencies](#dependencies)  
8. [Project Workflow](#project-workflow)  
9. [Future Work](#future-work)  
10. [Conclusion](#conclusion)  
11. [Author](#author)  
12. [Acknowledgments](#acknowledgments)  

---

## Project Overview
Women’s safety is a pressing social issue, particularly in urban environments. This project applies **machine learning models** (Logistic Regression, Random Forest, SVM, etc.) to predict whether a location is safe or unsafe based on factors like:
- Area, Zone, Tier  
- Time of day  
- Presence of police stations, bars, and residential levels  
- People frequency  

The outcome is a binary prediction (*Safe / Unsafe*) that can guide awareness, research, and potential safety planning.

---

## Features
- **End-to-end ML pipeline**: preprocessing, resampling (SMOTE/SMOTEN), training, evaluation.  
- **Multiple models** trained and compared, with confusion matrices and metrics.  
- **Streamlit Web App** for interactive prediction.  
- **Visualization support** using Matplotlib & Seaborn.  
- **Reproducible training pipeline** with serialized models.  

---

## 📂 Folder Structure
## 📂 Folder Structure & Datasets

- **data/**  
  Contains cleaned and preprocessed datasets for **Chennai, Bangalore, and Karnataka**:

  1. **Chennai** – Collected from [Kaggle](https://www.kaggle.com/) (public dataset containing safety-related features for Chennai).  
  2. **Bangalore & Karnataka** – Custom datasets created based on **research and compilation from public sources**:  
     - **Sources:**  
       - Local police station data and crime reports from official police websites  
       - Municipal and government safety portals  
       - Publicly available maps and area information  
       - Online news reports and safety surveys  
     - The data was manually cleaned, mapped, and standardized to match the structure of the Chennai dataset.  
     - Columns include:  
       - `Area` – Name of the locality  
       - `Zone` – Zone or police jurisdiction  
       - `Time` – Time of day categorized as Morning, Afternoon, Evening, Night  
       - `People.Frequency` – Low, Medium, High  
       - `Is.Police_Station` – Yes / No  
       - `Is.Bar` – Yes / No  
       - `Tier` – Middle / Outer / Other classification  
       - `Residence.Level` – Low / Medium / High  
       - `Class` – Target variable: Safe / Unsafe  

  All categorical features have been **converted to numeric values** for modeling. Sample CSVs are provided for testing and demonstration purposes.

- **data/**  
  Contains cleaned and preprocessed datasets for **Bangalore, Chennai, and Karnataka**.  
  All categorical features have been converted to numeric values for model training. Sample CSVs are provided for testing and demonstration purposes.

- **code/**  
  Contains all Python scripts used in the project:

  1. `WomenSafety.py` → Preprocesses datasets, applies SMOTE, trains **Logistic Regression, SVM, and Random Forest**, and saves the best model per dataset.  
  2. `app.py` → Streamlit web interface for user-friendly predictions with dynamic selection of city, area, zone, and time.  
  3. `bangalore_data.py` → Preprocesses raw Bangalore dataset, transforms columns, maps features, and creates numeric versions for modeling.  
  4. `generate_encoder.py` → Fits and saves `OrdinalEncoder` for all categorical columns to ensure consistent encoding during training and prediction.

- **models/**  
  Stores all trained model artifacts and encoders:  
  - `women_safety_pipeline.pkl` → Final Random Forest pipeline including preprocessing  
  - `best_model_*.pkl` → Models trained on individual datasets  
  - `label_encoders.pkl` → Saved encoder for categorical features

- **requirements.txt**  
  Lists all Python libraries needed to run the project and the Streamlit app.

  ## 📥 Installation

Follow these steps to set up the **Women Safety Prediction** project locally:

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/women-safety-prediction.git
cd women-safety-prediction

### 2. Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Verify Installation

- Ensure Python version is **>= 3.10**:
```bash
python --version

- Check that **Streamlit** and required libraries are installed:
```bash
streamlit --version
python -c "import pandas, numpy, sklearn, imblearn, folium, matplotlib, seaborn"


##  Usage ##

```bash
# 1. Run the Streamlit App
# Launch the interactive web interface for predictions
streamlit run code/app.py

# 2. Train Models
# Retrain models using the preprocessed datasets
python code/WomenSafety.py

# 3. Generate Encoder
# Ensure consistent categorical encoding for training and inference
python code/generate_encoder.py

# 4. Evaluate Models
# Generate evaluation metrics and visualizations
python code/evaluate.py

### 5. Input Example

```bash
# Example: Using the Streamlit app to predict safety
# Select City, Area, Zone, and Time in the app interface
# The app will output Safety Status and related details


