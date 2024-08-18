# Predictive Maintenance System

## Overview

This project aims to develop a predictive maintenance system that analyzes sensor data from machinery to predict when maintenance is required. By leveraging machine learning algorithms, this system identifies patterns and anomalies in the data to forecast potential failures, thereby reducing downtime and maintenance costs.

## Technologies Used

- **Python**: Programming language used for implementing the project.
- **Pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning models and evaluation.
- **TensorFlow**: For deep learning models.
- **Matplotlib**: For data visualization.
- **FPDF**: For generating PDF reports.

## Project Structure

- **data/**: Contains the `sensor_data.csv` file used for storing and retrieving sensor data and reports.
- **models/**: Directory to store trained models (e.g., RandomForest, TensorFlow models).
- **notebooks/**: Jupyter notebooks for exploratory data analysis.
- **scripts/**: Contains utility scripts for various functions.
- **main.py**: The main script to run the entire workflow.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed and the required packages. You can install them using pip:

```bash
pip install pandas scikit-learn tensorflow matplotlib fpdf
```

### Running the Project
Initialize Data: The main.py script will initialize the CSV file if it doesn't already exist.

Load and Process Data: The script loads existing data, processes it, and performs training on a RandomForest model and a TensorFlow model.

Generate Reports: The script generates and saves performance reports as PDF files.

Simulate Sensor Data: It simulates real-time sensor data and appends it to the CSV file.

To run the project, simply execute:
```bash
python main.py
```

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- TensorFlow and scikit-learn for their powerful machine learning libraries.
- Matplotlib for creating informative visualizations.
- FPDF for generating PDF reports.