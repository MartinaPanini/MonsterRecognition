import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(train_data):
    # Separare le feature (X) e le etichette (y)
    X_train = train_data.drop(columns=['Label'])
    y_train = train_data['Label']

    # Codifica le etichette in numeri
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Crea e allena il modello
    model = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=10000, random_state=42)
    model.fit(X_train, y_train_encoded)

    return model, label_encoder

def classify_data(model, label_encoder, test_data):
    # Prevedi le etichette per i dati forniti
    predictions = model.predict(test_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels

def evaluate_model(model, X_test, y_test, label_encoder, output_path):
    # Prevedi le etichette per i dati di test
    y_pred = model.predict(X_test)

    # Calcola l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Report di classificazione
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    # Salva i risultati in un file
    with open(output_path, "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    print(f"Statistiche salvate in: {output_path}")