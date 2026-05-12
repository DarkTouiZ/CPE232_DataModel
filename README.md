# CPE232 Data Model

Coursework and project repository for CPE232 Data Model. The repository contains Python notebooks, datasets, lab exercises, homework assignments, and a PM2.5 forecasting project built with Streamlit.

## Repository Contents

```text
.
├── Laboratory/     # Lab notebooks, PDFs, and sample datasets
├── HomeWork/       # Homework notebooks, reports, and datasets
└── Projects/       # PM2.5 forecasting dataset, model pipeline, and Streamlit app
```

## Highlights

- Basic Python programming exercises
- Data type exploration with CSV, JSON, XML, and HTML examples
- Data preparation and cleaning practice
- Exploratory data analysis and visualization
- Classification model exercises
- PM2.5 forecasting project using historical pollution, weather, and fire hotspot data

## PM2.5 Forecasting Project

The main project is located in `Projects/`. It includes:

- Dataset creation notebooks and supporting datasets
- Historical PM2.5, weather, forecast weather, and fire hotspot data
- Model training pipelines using Prophet and CatBoost
- Saved trained models for multiple station IDs
- A Streamlit app for PM2.5 forecasting

Key files:

- `Projects/create_dataset.ipynb` - builds combined datasets from PM2.5, weather, and fire data
- `Projects/data_model_pipeline.ipynb` - model training and experimentation pipeline
- `Projects/Data_Model_Pipeline_V2.ipynb` - updated modeling pipeline
- `Projects/app/main.py` - Streamlit forecasting application
- `Projects/requirements.txt` - Python dependencies
- `Projects/run.sh` - app launch command
- `Projects/Readme-dataCollect.md` - dataset preparation notes
- `Projects/Readme-streamlitApp.md` - Streamlit app setup notes

## Getting Started

Clone the repository:

```bash
git clone https://github.com/DarkTouiZ/CPE232_DataModel.git
cd CPE232_DataModel
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r Projects/requirements.txt
```

## Run the Streamlit App

From the `Projects/` directory, run:

```bash
cd Projects
streamlit run app/main.py
```

Or use the provided script:

```bash
cd Projects
sh run.sh
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dataset Preparation

The project dataset combines:

- PM2.5 historical records
- Weather data from Open-Meteo
- Historical weather forecast data
- Fire hotspot data
- Station metadata and coordinate files

For detailed dataset creation steps, see `Projects/Readme-dataCollect.md`.

## App Notes

The Streamlit app loads trained Prophet and CatBoost models, fetches weather forecast data, engineers time-series features, and displays PM2.5 forecast values with Plotly charts and category labels.

The app currently uses model and dataset paths under `Projects/app/` and `Projects/dataset/`, so run it from the `Projects/` directory.

## Requirements

Main libraries used in the project include:

- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- streamlit
- prophet
- catboost
- openmeteo-requests
- requests-cache
- retry-requests

See `Projects/requirements.txt` for the full dependency list.

## Notes

Some datasets and trained model files are included in the repository. Large generated artifacts such as zipped data and model archives may take additional time to clone or download.
