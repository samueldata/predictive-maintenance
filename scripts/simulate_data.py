import numpy as np
import pandas as pd
import os
import time

def configure_paths():
    base_dir = os.path.join(os.getcwd(), 'data')
    file_path = os.path.join(base_dir, 'sensor_data.csv')
    return file_path

def append_to_csv(file_path, new_data):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        new_df = pd.DataFrame(new_data)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame(new_data)

    updated_df.to_csv(file_path, index=False)
    print(f"Dados atualizados salvos em: {file_path}")

def simulate_sensor_data(file_path):
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
    file_path = configure_paths()
    simulate_sensor_data(file_path)
