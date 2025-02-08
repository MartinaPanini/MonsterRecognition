import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, y_train_encoded):
    """
    Trains a Multi-layer Perceptron (MLP) classifier on the provided training data.
    Parameters:
    X_train (array-like or sparse matrix): The input data to train the model. 
                                        Shape (n_samples, n_features).
    y_train_encoded (array-like): The encoded target values (labels) for the training data.
                                Shape (n_samples,).
    Returns:
    MLPClassifier: The trained MLP classifier model.
    """
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
    """
    Classifies the given test data using the provided model and label encoder.
    Args:
        model: The trained model used for making predictions.
        label_encoder: The label encoder used to decode the predicted labels.
        test_data: The data to be classified.
    Returns:
        tuple: A tuple containing:
            - An array of predicted labels (decoded using the label encoder).
            - An array of maximum probabilities for each prediction, rounded to two decimal places.
            - An array of prediction probabilities for each class.
    """
    predictions = model.predict(test_data)
    prediction_probs = model.predict_proba(test_data)
    max_probs = prediction_probs.max(axis=1)  # Max probability for each prediction
    max_probs = (max_probs * 100).round(2)
    return label_encoder.inverse_transform(predictions), max_probs, prediction_probs

def evaluate_model(model, X_test, y_test, label_encoder, output_path):
    """
    Evaluates the performance of a given model on the test dataset and writes the results to a specified output file.
    Parameters:
    model (sklearn.base.BaseEstimator): The trained model to evaluate.
    X_test (array-like): The input features of the test dataset.
    y_test (array-like): The true labels of the test dataset.
    label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used to transform labels.
    output_path (str): The file path where the evaluation results will be saved.
    Writes:
    A file at the specified output path containing the accuracy and classification report of the model.
    """
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    class_report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
    
    with open(output_path, "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)