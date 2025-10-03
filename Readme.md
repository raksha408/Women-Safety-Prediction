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
6. [Dependencies](#dependencies)  
7. [Project Workflow](#project-workflow)  
8. [Future Work](#future-work)  
9. [Conclusion](#conclusion)  
10. [Acknowledgments](#acknowledgments) 
11. [Author](#author)  

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

## Folder Structure

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

  ## Installation

Follow these steps to set up the **Women Safety Prediction** project locally:

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/women-safety-prediction.git
cd women-safety-prediction

---

### 2. Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate

---

### 3. Install dependencies
pip install -r requirements.txt

---

### 4. Verify Installation

- Ensure Python version is **>= 3.10**:
```bash
python --version

- Check that **Streamlit** and required libraries are installed:
```bash
streamlit --version
python -c "import pandas, numpy, sklearn, imblearn, folium, matplotlib, seaborn"

---

##Usage

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

## 🛠 Dependencies

The Women Safety Prediction project requires the following Python libraries. All dependencies can be installed at once using the `requirements.txt` file.

```bash
# Install all required libraries
pip install -r requirements.txt

---

## Project Workflow

The Women Safety Prediction project follows a structured workflow from data collection to deployment. The steps are outlined below:

1. **Data Collection**  
   - Chennai dataset collected from Kaggle.  
   - Bangalore and Karnataka datasets created through research and manual compilation from local sources and public reports.  

2. **Data Preprocessing**  
   - Cleaning and handling missing values.  
   - Categorical features (Area, Zone, Time, People Frequency, Police Station, Bar, Tier, Residence Level) converted to numeric using encoders.  
   - Separate numeric datasets saved for model training.

3. **Model Training**  
   - Models trained: **Logistic Regression, SVM, Random Forest**.  
   - SMOTE applied to handle class imbalance.  
   - Best model for each dataset saved as `.pkl` files in `models/`.

4. **Encoder Creation**  
   - `generate_encoder.py` creates label encoders for consistent categorical feature transformation.  
   - Encoders saved as `label_encoders.pkl`.

5. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-Score.  
   - Evaluation results saved as images in `Results/`.

6. **Deployment / Web Interface**  
   - `app.py` provides a Streamlit web interface.  
   - Users can select City, Area, Zone, and Time to get safety predictions.  
   - Additional details like People Frequency, Police Station, Bar, Tier, and Residence Level are displayed.

7. **Prediction Output**  
   - Safety Status: Safe / Unsafe.  
   - Outputs generated in real-time based on user selections in the Streamlit app.

>  This workflow ensures a complete pipeline from raw data to interactive predictions and reproducible results.

## Future Work

Future enhancements for the Women Safety Prediction System could include integrating **real-time crime data**, **GPS tracking**, and **AI-driven safety recommendations** to improve prediction accuracy.  

By leveraging **live crime reports, social media trends, and emergency alerts**, the system can provide dynamic updates on safety conditions. Additionally, **advanced deep learning models** can enhance prediction capabilities by analyzing complex patterns in crime data.  

A **mobile application** with **voice assistance**, **multilingual support**, and **user feedback integration** can further improve accessibility and usability, ensuring a more effective and user-friendly experience.

---

## Conclusion

The Women Safety Prediction System has been successfully developed to assess the safety of different locations based on various factors such as **time, area type, police station availability, and population density**. By leveraging **machine learning techniques**, the system provides reliable predictions, classifying areas as **Safe** or **Unsafe**, helping users make informed decisions about their safety. The model has been trained and tested thoroughly to ensure high accuracy and consistency in predictions.  

This system aims to **empower women** by offering valuable insights into the safety levels of different locations, thereby reducing potential risks. The predictive analysis can be used by **individuals, organizations, and law enforcement agencies** to enhance security measures and safety awareness.  

The system has undergone **extensive testing and validation** to verify its reliability and usability. It efficiently processes user input and provides quick and accurate safety assessments. The model’s performance has been optimized to handle different real-world scenarios, ensuring that predictions remain precise across diverse datasets. Rigorous testing evaluated the accuracy of predictions, response time, and adaptability to different locations and conditions. The results confirmed that the system performs consistently well and delivers useful insights, making it a valuable tool for assessing location safety. Additionally, **user interface testing** ensured that the system remains intuitive and user-friendly, enabling seamless interaction for users of all backgrounds.  

Overall, the Women Safety Prediction System serves as an **effective solution** to enhance safety awareness and decision-making. By utilizing machine learning algorithms, the system offers **real-time safety assessments** based on historical and location-specific data. The predictions generated can help individuals **choose safer routes, avoid high-risk areas, and take necessary precautions**. This system not only contributes to personal security but also supports broader **community safety initiatives**, making it a valuable addition to modern safety technology.

---

## Acknowledgments

I sincerely thank all those who made this project possible:

- **Kaggle** for the Chennai dataset.  
- **Local public sources** for Bangalore and Karnataka datasets.  
- **Open-source Python libraries** such as `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `streamlit`, `folium`, `matplotlib`, and `seaborn`.  
- **Community support** from GitHub, Stack Overflow, and online documentation.  

Their invaluable resources and guidance empowered the successful development of this project.

---

## Author

**Shriraksha Kulkarni**  




