import pandas as pd
import pickle
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

dataset_files = ["safety_data.csv", "bangalore_safety.csv", "Karnataka_cleaned.csv"]

def preprocess_data(file):
    print(f"\nProcessing dataset: {file}")
    df = pd.read_csv(file)

    for col in ["Area", "Zone", "Time"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    df.fillna({
        "People.Frequency": "Medium",
        "Is.Police_Station": "No",
        "Is.Bar": "No",
        "Tier": "Middle",
        "Residence.Level": "Medium",
        "Class": "Safe"
    }, inplace=True)

    X = df[["Area", "Zone", "Time"]]
    y = df["Class"].map({"Safe": 1, "Unsafe": 0})

    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)

    with open(f"ordinal_encoder_{file}.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"Encoder saved: ordinal_encoder_{file}.pkl")

    if y.value_counts().min() / y.value_counts().max() < 0.5:
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
    else:
        X_resampled, y_resampled = X_encoded, y  

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

def train_models(X_train, X_test, y_train, y_test, dataset_name):
    print(f"\nDataset: {dataset_name}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
    }

    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_model_name = name

    if best_model:
        model_filename = f"best_model_{dataset_name}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(best_model, file)
        print(f"\nBest Model for {dataset_name}: {best_model_name} ({best_acc:.4f})")
        print(f"Model saved as: {model_filename}")

for file in dataset_files:
    train_models(*preprocess_data(file), file)
