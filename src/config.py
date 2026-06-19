from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_GLOBAL_DATA = DATA_DIR / "global_air_quality_2014_2025.csv"

BASE_DIR = PROJECT_ROOT / "base"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"


REQUIRED_COLUMNS = [
    "Country",
    "State",
    "City",
    "Date",
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "NO (ug/m3)",
    "NO2 (ug/m3)",
    "NOx (ppb)",
    "NH3 (ug/m3)",
    "CO (mg/m3)",
    "SO2 (ug/m3)",
    "O3 (ug/m3)",
    "Benzene (ug/m3)",
    "Toluene (ug/m3)",
    "Xylene (ug/m3)",
    "AQI",
    "AQI_Bucket",
    "Wind_Speed (km/h)",
    "Humidity (%)",
    "Deforestation_Rate_%",
    "Industry_Growth_%",
    "CO2_Emission_MT",
    "Population_Density_per_SqKm",
]


NUMERIC_COLUMNS = [
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "NO (ug/m3)",
    "NO2 (ug/m3)",
    "NOx (ppb)",
    "NH3 (ug/m3)",
    "CO (mg/m3)",
    "SO2 (ug/m3)",
    "O3 (ug/m3)",
    "Benzene (ug/m3)",
    "Toluene (ug/m3)",
    "Xylene (ug/m3)",
    "AQI",
    "Wind_Speed (km/h)",
    "Humidity (%)",
    "Deforestation_Rate_%",
    "Industry_Growth_%",
    "CO2_Emission_MT",
    "Population_Density_per_SqKm",
]


def ensure_results_dirs() -> None:
    for directory in [
        RESULTS_DIR,
        PROCESSED_DATA_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
