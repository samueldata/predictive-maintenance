import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from fpdf import FPDF
import joblib
import time

def configure_paths():
    base_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    file_path = os.path.join(base_dir, 'sensor_data.csv')
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return base_dir, file_path, model_dir

def append_to_csv(file_path, new_data):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        new_df = pd.DataFrame(new_data)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame(new_data)

    updated_df.to_csv(file_path, index=False)
    print(f"Dados atualizados salvos em: {file_path}")

def generate_simulated_data(file_path):
    np.random.seed(45)
    num_samples = 1000

    temperature = np.random.normal(loc=70, scale=5, size=num_samples)
    vibration = np.random.normal(loc=30, scale=2, size=num_samples)
    pressure = np.random.normal(loc=100, scale=10, size=num_samples)
    failure = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])

    new_data = {
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'failure': failure
    }

    append_to_csv(file_path, new_data)

def load_and_process_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(df.head())
        print(df.describe())
        df.plot(subplots=True, figsize=(10, 12))
        plt.show()
        return df
    else:
        print(f"Arquivo não encontrado no caminho: {file_path}")
        return None

def process_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    df_scaled = df_scaled.fillna(df_scaled.mean())
    df_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_scaled.dropna(inplace=True)
    
    epsilon = 1e-10
    df_scaled['temperature_vibration_ratio'] = df_scaled['temperature'] / (df_scaled['vibration'] + epsilon)
    
    if df_scaled['temperature_vibration_ratio'].isna().any() or np.isinf(df_scaled['temperature_vibration_ratio']).any():
        print("Valores NaN ou infinitos encontrados na coluna 'temperature_vibration_ratio'.")
        df_scaled['temperature_vibration_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_scaled.dropna(inplace=True)
    
    return df_scaled

def train_random_forest(df_scaled, model_dir):
    X = df_scaled.drop('failure', axis=1)
    y = df_scaled['failure']
    print("Training Features Shape:", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    print("RandomForest model saved.")
    
    return model, X_train, y_train, X_test, y_test

def train_tensorflow_model(X_train, y_train, X_test, y_test, model_dir):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')
    
    # Save the model
    model.save(os.path.join(model_dir, 'tensorflow_model.h5'))
    print("TensorFlow model saved.")
    
    return history, accuracy, loss

def create_plots(history, base_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'accuracy_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'loss_plot.png'))
    plt.close()

def generate_report(base_dir, accuracy, loss):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Resultados do Modelo", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Precisão no Teste: {accuracy*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Perda no Teste: {loss:.4f}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Gráficos de Treinamento", ln=True)
    pdf.image(os.path.join(base_dir, 'accuracy_plot.png'), x=10, y=60, w=180)
    pdf.add_page()
    pdf.image(os.path.join(base_dir, 'loss_plot.png'), x=10, y=10, w=180)
    pdf.output(os.path.join(base_dir, "relatorio_resultados.pdf"))
    print("Relatório PDF gerado com sucesso.")

def simulate_sensor_data(model, file_path):
    while True:
        temperature = np.random.normal(loc=70, scale=5)
        vibration = np.random.normal(loc=30, scale=2)
        pressure = np.random.normal(loc=100, scale=10)
        failure = np.random.choice([0, 1], size=1, p=[0.95, 0.05])[0]

        sensor_data = [temperature, vibration, pressure, failure]
        df = pd.DataFrame([sensor_data], columns=['temperature', 'vibration', 'pressure', 'failure'])
        append_to_csv(file_path, df.to_dict(orient='records'))
        time.sleep(10)

if __name__ == "__main__":
    base_dir, file_path, model_dir = configure_paths()
    generate_simulated_data(file_path)
    df = load_and_process_data(file_path)
    if df is not None:
        df_scaled = process_data(df)
        model, X_train, y_train, X_test, y_test = train_random_forest(df_scaled, model_dir)
        history, accuracy, loss = train_tensorflow_model(X_train, y_train, X_test, y_test, model_dir)
        create_plots(history, base_dir)
        generate_report(base_dir, accuracy, loss)
        simulate_sensor_data(model, file_path) # Comentar esta linha se não desejar gerar dados de simulação ciclicos
