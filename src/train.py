"""
Script principale per l'addestramento del modello finale di Age Recognition.

Questo script esegue i seguenti passaggi:
1.  Carica il dataset di sviluppo completo (`development.csv`).
2.  Definisce la pipeline end-to-end ottimale, che include:
    - L'estrazione di feature audio custom (`AudioFeatureExtractor`).
    - Tutti i passaggi di pulizia e feature engineering (per 'tempo', 'ethnicity', etc.).
    - Il pre-processore finale con lo scaler migliore (`MinMaxScaler`).
    - Il modello migliore (`HistGradientBoostingRegressor`) con gli iperparametri ottimizzati.
3.  Addestra questa pipeline finale sull'INTERO dataset di sviluppo.
4.  Salva l'oggetto pipeline addestrato in un file (`.joblib`) per l'uso successivo
    da parte dello script `predict.py`.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

from feature_extractor import AudioFeatureExtractor, TempoCleaner, RedundantFeatureDropper, RareCategoryGrouper

MODEL_PATH = Path('models/final_model.joblib')
DEVELOPMENT_PATH = Path("data/development.csv")
FEATURES_TO_DROP_PATH = Path('artifacts/features_to_drop.json')
AUDIO_DIR = 'data'
NUMERICAL_FEATURES = [
    'mean_pitch', 'max_pitch', 'min_pitch', 'jitter', 'shimmer',
    'energy', 'zcr_mean', 'spectral_centroid_mean', 'tempo', 'hnr',
    'num_words', 'num_pauses', 'silence_duration'
]
CATEGORICAL_FEATURES = ['gender', 'ethnicity']

def main():
    """Funzione principale per addestrare il modello."""
    print("Inizio del processo di addestramento...")

    print("Caricamento dei dati di valutazione...")
    try:
        dev_df = pd.read_csv(DEVELOPMENT_PATH)
        dev_df.columns = dev_df.columns.str.strip()
    except FileNotFoundError:
        print(F"ERRORE: {DEVELOPMENT_PATH} non trovato.")
        return
    X = dev_df.drop(columns=['age'])
    y = dev_df['age']

    try:
        with open(FEATURES_TO_DROP_PATH, 'r') as f:
            useless_audio_features = json.load(f)
        print(f"Caricate {len(useless_audio_features)} feature da rimuovere.")
    except FileNotFoundError:
        print(f"ERRORE: File {FEATURES_TO_DROP_PATH} non trovato. Eseguire prima il notebook pipeline_prototype.")
        return

    temp_extractor = AudioFeatureExtractor(audio_dir=AUDIO_DIR)
    new_audio_feature_names = temp_extractor._generate_feature_names()

    FINAL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + new_audio_feature_names
    FINAL_CLEANED_NUMERICAL_FEATURES = [f for f in FINAL_NUMERICAL_FEATURES if f not in useless_audio_features]
    CATEGORICAL_CLEANED_FEATURES = [f for f in CATEGORICAL_FEATURES if f not in useless_audio_features]

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    final_preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, FINAL_CLEANED_NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_CLEANED_FEATURES)
    ])

    final_pipeline_cleaned = Pipeline(steps=[
        ('tempo_cleaner', TempoCleaner(column_name='tempo')),
        ('feature_dropper', RedundantFeatureDropper(columns_to_drop=useless_audio_features)),
        ('rare_grouper', RareCategoryGrouper(columns=['ethnicity'], n_top_categories=10)), 
        ('preprocessor', final_preprocessor),
        ('model', HistGradientBoostingRegressor(
            max_iter=500
        ))
    ])

    print("Addestramento della pipeline finale su tutto il dataset di sviluppo...")
    final_pipeline_cleaned.fit(X, y)
    print("Addestramento completato.")

    model_filename = 'final_model.joblib'
    joblib.dump(final_pipeline_cleaned, model_filename)
    print(f"Modello finale salvato in '{model_filename}'")

if __name__ == '__main__':
    main()