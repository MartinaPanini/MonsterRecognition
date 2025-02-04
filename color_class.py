import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, y_train_encoded, class_weight_dict, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            best_params = json.load(f)
        #print("✅ Parametri ottimali caricati da file:", best_params)
        model = RandomForestClassifier(class_weight=class_weight_dict, **best_params)
        model.fit(X_train, y_train_encoded)
        return model
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight=class_weight_dict), 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2, 
                               scoring='accuracy')

    grid_search.fit(X_train, y_train_encoded)
    best_params = grid_search.best_params_
    with open(path, 'w') as f:
        json.dump(best_params, f)
    #print("🎯 Parametri ottimali trovati e salvati:", best_params)

    # Train model with best parameters
    model = grid_search.best_estimator_
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