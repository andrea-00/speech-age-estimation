"""
This module defines the custom scikit-learn transformer for audio feature extraction.

It contains the AudioFeatureExtractor class, which loads audio files, extracts
features (e.g., MFCCs), and returns a numerical feature vector ready to be used
in a scikit-learn pipeline.
"""
import numpy as np
import pandas as pd
import librosa

from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin

from pathlib import Path

class AudioFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Estrae un set ricco di feature audio (MFCCs, Chroma, Spectral Contrast)
    e le loro statistiche aggregate da file audio.
    Progettato per essere inserito in una pipeline di scikit-learn.
    """
    def __init__(self, audio_dir, n_mfcc=80, top_db=20):
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.top_db = top_db
        
        self.stats_funcs = [np.mean, np.std, skew, kurtosis, np.min, np.max]
        self.stats_names = ['mean', 'std', 'skew', 'kurtosis', 'min', 'max']
        
        self._feature_names = self._generate_feature_names()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Inizio estrazione nuove feature audio...")
        audio_dir_path = Path(self.audio_dir)

        features_list = []

        for file_path in X['path']:
            try:
                full_path = audio_dir_path / file_path
                y_audio, sr = librosa.load(full_path, sr=None)

                mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=self.n_mfcc)
                mfccs_delta = librosa.feature.delta(mfccs)
                mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
                chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)

                all_stats = []
                for features in [mfccs, mfccs_delta, mfccs_delta2, chroma, spectral_contrast]:
                    stats = np.array([func(features, axis=1) for func in self.stats_funcs]).flatten()
                    all_stats.append(stats)
                
                speech_segments = librosa.effects.split(y=y_audio, top_db=self.top_db)
                
                if len(speech_segments) > 0:
                    segment_durations = [(end - start) / sr for start, end in speech_segments]
                    num_segments = len(segment_durations)
                    mean_segment_duration = np.mean(segment_durations)
                    std_segment_duration = np.std(segment_durations)
                    
                    total_speech_duration = np.sum(segment_durations)
                    total_duration = librosa.get_duration(y=y_audio, sr=sr)
                    silence_ratio = (total_duration - total_speech_duration) / total_duration if total_duration > 0 else 0
                else:
                    num_segments, mean_segment_duration, std_segment_duration, silence_ratio = 0, 0, 0, 1.0

                segment_features = np.array([
                    num_segments, mean_segment_duration, std_segment_duration, silence_ratio
                ])
                
                combined_features = np.concatenate(all_stats + [segment_features])
                features_list.append(combined_features)

            except Exception as e:
                print(f"Errore nel processare {file_path}: {e}")
                features_list.append(np.full(len(self._feature_names), np.nan))
        
        new_features_df = pd.DataFrame(features_list, columns=self._feature_names, index=X.index)

        print("Estrazione completata.")
        return pd.concat([X, new_features_df], axis=1)

    def _generate_feature_names(self):
        names = []
        
        feature_groups = {
            'mfcc': self.n_mfcc,
            'mfcc_delta': self.n_mfcc,
            'mfcc_delta2': self.n_mfcc,
            'chroma': 12, # Chroma ha sempre 12 feature
            'spec_contrast': 7 # Spectral contrast ha sempre 7 feature
        }
        
        for prefix, num_features in feature_groups.items():
            for i in range(num_features):
                for stat_name in self.stats_names:
                    names.append(f"{prefix}_{stat_name}_{i+1}")
        
        segment_names = [
            'num_speech_segments',
            'mean_speech_duration',
            'std_speech_duration',
            'silence_ratio'
        ]
                    
        return names + segment_names

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out")
        
        return np.concatenate([np.array(input_features), np.array(self._feature_names)])


class TempoCleaner(BaseEstimator, TransformerMixin):
    """
    A custom transformer to clean the 'tempo' column.
    It checks if the column is of object type, and if so, it strips
    the brackets and converts it to a numeric type.
    """
    def __init__(self, column_name='tempo', verbose=False):
        self.column_name = column_name
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        if self.column_name in X_.columns and X_[self.column_name].dtype == 'object':
            if self.verbose:
                print(f"Cleaning column: '{self.column_name}'")
            X_[self.column_name] = pd.to_numeric(
                X_[self.column_name].str.strip('[]'),
                errors='coerce'
            )
        
        return X_


class RedundantFeatureDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer to drop specified redundant columns from a DataFrame.
    """
    def __init__(self, columns_to_drop, verbose=False):
        if not isinstance(columns_to_drop, list):
            raise ValueError("'columns_to_drop' must be a list of column names.")
        self.columns_to_drop = columns_to_drop
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.verbose:
            print(f"Dropping redundant columns: {self.columns_to_drop}")
        X_ = X_.drop(columns=self.columns_to_drop, errors='ignore')
        
        return X_


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    A custom transformer to group rare categorical features into a single 'Other' category.
    It identifies the top N most frequent categories during fit and applies this
    grouping during transform.
    """
    def __init__(self, columns, n_top_categories=10, verbose=False):
        self.columns = columns
        self.n_top_categories = n_top_categories
        self.top_categories_ = {}
        self.verbose = verbose

    def fit(self, X, y=None):
        for col in self.columns:
            top_cats = X[col].value_counts().nlargest(self.n_top_categories).index.tolist()
            self.top_categories_[col] = top_cats
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        
        if self.verbose:
            print(f"Grouping rare categories for columns: {self.columns}")
        
        for col in self.columns:
            # Sostituisce le categorie non principali con 'Other'
            top_cats = self.top_categories_.get(col)
            if top_cats:
                X_[col] = np.where(X_[col].isin(top_cats), X_[col], 'Other')
        
        return X_