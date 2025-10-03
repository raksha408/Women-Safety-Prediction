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

## Folder Structure
