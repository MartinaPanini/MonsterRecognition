import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, y_train_encoded, class_weight_dict):
    # Create and train the model
    model = RandomForestClassifier(class_weight=class_weight_dict)
    model.fit(X_train, y_train_encoded)
    return model

def classify_data(model, label_encoder, test_data):
    predictions = model.predict(test_data)
    return label_encoder.inverse_transform(predictions)

def evaluate_model(model, X_test, y_test, label_encoder, output_path):
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    class_report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
    
    with open(output_path, "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)