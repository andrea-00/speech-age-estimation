"""
This script provides an example of how to use the trained model for inference.

It loads the saved pipeline from the 'models/' directory and uses it to
predict the age from a single new audio file provided via the command line.
"""

import pandas as pd
import joblib
from pathlib import Path

from feature_extractor import AudioFeatureExtractor, TempoCleaner, RedundantFeatureDropper, RareCategoryGrouper

MODEL_PATH = Path('models/final_model.joblib')
EVALUATION_PATH = Path("data/evaluation.csv")
AUDIO_DIR = 'data'
SUBMISSION_PATH = Path('data/submission.csv')

def main():
    """Carica il modello e genera le predizioni per la submission."""
    print("Caricamento del modello addestrato...")
    try:
        final_model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERRORE: Modello '{MODEL_PATH}' non trovato. Esegui prima train.py.")
        return

    print("Caricamento dei dati di valutazione...")
    try:
        eval_df = pd.read_csv(EVALUATION_PATH)
        eval_df.columns = eval_df.columns.str.strip()
    except FileNotFoundError:
        print(F"ERRORE: {EVALUATION_PATH} non trovato.")
        return

    print("Aggiornamento del percorso della cartella audio per la valutazione...")
    final_model.set_params(audio_feature_extractor__audio_dir=AUDIO_DIR)
    
    print("Esecuzione delle predizioni sul set di valutazione...")
    predictions = final_model.predict(eval_df)
    
    predictions[predictions < 0] = 0
    print("Predizioni completate.")

    submission_df = pd.DataFrame({'Id': eval_df['Id'], 'Predicted': predictions})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("\nFile 'submission.csv' creato con successo!")

if __name__ == '__main__':
    main()