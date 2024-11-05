# ML Engine

This repository contains all of the SmartShield AI Team’s work for building, training, and deploying machine learning models, as well as the API and tools necessary for end-to-end machine learning workflows.

## Contents

- **Data Collection**
- **Feature Engineering**
- **Model Training and Evaluation**
- **API App For Ml Engine**
- **Testing**

## Folder Structure

```plaintext
├── LICENSE                         # License for the project
├── README.md                       # Project documentation
├── __pycache__/                    # Python bytecode cache
│   └── app.cpython-311.pyc
├── api_src/                        # Source code for the API application
│   ├── config                      # Configuration files for the API
│   ├── logger                      # Logging configuration and utilities
│   ├── routers                     # API route definitions
│   ├── schemas                     # API data schemas
│   ├── services                    # Services used by the API
│   └── tests                       # API-specific tests
├── app.py                          # Main application entry point
├── config/                         # Configuration files for model training
│   ├── __init__.py
│   └── train                       # Model training configuration
├── data/                           # Directory for datasets
│   ├── UNSW_NB15_data              # Dataset related to UNSW_NB15
│   └── __init__.py
├── models/                         # Directory for storing trained models
│   ├── UNSW_NB15_models            # Serialized UNSW_NB15 models
│   └── __init__.py
├── notebooks/                      # Jupyter notebooks for exploration and prototyping
│   ├── DetectionFeatureEngineering.ipynb  # Feature engineering notebook
│   ├── UNSW_NB15_notebooks         # Notebooks related to UNSW_NB15 data
│   └── __init__.py
├── requirements.txt                # List of dependencies
├── src/                            # Source code for data processing and model development
│   ├── __init__.py
│   ├── __pycache__
│   ├── data                        # Scripts for loading and processing data
│   ├── features                    # Scripts for creating and selecting features
│   ├── models                      # Scripts for training and evaluating models
│   ├── utils                       # Utility functions
│   └── visualization               # Scripts for generating data visualizations
├── tests/                          # Unit tests for various modules
│   ├── UNSW_NB15_tests             # Tests related to the UNSW_NB15 data and models
│   └── __init__.py
```

## Setup and Running the Application

1. **Create a Virtual Environment**  
   In the project root directory, create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**  
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install the Required Dependencies**  
   Install all required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**  
   Start the ML Engine application with Uvicorn using the following command:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8002 --workers 16
   ```

   - **`app:app`**: Specifies the app instance in the main application file.
   - **`--host 0.0.0.0`**: Makes the app accessible externally.
   - **`--port 8002`**: Runs the app on port 8002.
   - **`--workers 16`**: Starts the app with 16 worker processes, allowing concurrent request handling.
