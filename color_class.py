import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, y_train_encoded, class_weight_dict):
   model = MLPClassifier(
        hidden_layer_sizes=(50,50),  # Number of neurons in each layer
        activation='relu',  # Activation function
        solver='adam',  # Optimizer
        alpha=0.01,  # Regularization term
        learning_rate='adaptive',  # Learning rate schedule
        early_stopping=True,  # Stop training when validation score doesn't improve
        max_iter=400,  # Number of iterations
        random_state=42
    )
   model.fit(X_train, y_train_encoded)
   return model

def classify_data(model, label_encoder, test_data):
    predictions = model.predict(test_data)
    prediction_probs = model.predict_proba(test_data)
    max_probs = prediction_probs.max(axis=1)  # Probabilità massima per ogni predizione
    max_probs = (max_probs * 100).round(2)
    return label_encoder.inverse_transform(predictions), max_probs, prediction_probs

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