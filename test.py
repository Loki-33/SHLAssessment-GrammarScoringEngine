import language_tool_python
import spacy
import textstat
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
from transcribe import AudioTranscribe
from analyze import AudioAnalyzer, GrammarAnalyzer 
from tqdm import tqdm 
import joblib
import warnings 
warnings.filterwarnings('ignore')

class ScorePredict:
    """
    Extract the features from teh audio files and use it to 
    predict the score for the test dataset audios 
    """
    def __init__(self, whisper_model_size = 'base'):
        self.transcriber = AudioTranscribe(whisper_model_size)
        self.grammar_analyzer = GrammarAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.model = joblib.load('grammar_scoring_model.json')['model'] 
        self.feature_names = None 

    def extract_features(self, audio_path):
        print(f"PROCESSING---------")
        print(f"TRANSCRIBING....")
        transcription = self.transcriber.transcribe(audio_path)
        text = transcription['text']
        segments = transcription['segments']

        print('GRAMMAR ANALYZE......')
        grammar_features = self.grammar_analyzer.analyze_grammar(text)

        print('AUDIO ANALYZE........')
        audio_features = self.audio_analyzer.analyze_audio(audio_path, segments)
        features = {**grammar_features, **audio_features}
        features['transcription'] = text 

        return features 

    def prepare_training_data(self, audio_paths):
        all_features = []
        with tqdm(total=len(audio_paths), desc='Processing Audio Files') as pbar:
            for audio_path in audio_paths: 
                features = self.extract_features(audio_path)
                all_features.append(features)
                pbar.update(1)
        df = pd.DataFrame(all_features)
        transcription = df['transcription'].values if 'transcription' in df.columns else None
        df = df.drop('transcription', axis=1, errors='ignore')

        self.feature_names = df.columns.tolist()

        return df, transcription

    def predict(self, X):
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 0, 5)

        return predictions 


if __name__ == '__main__':
    import os 
    test_audio_path = 'dataset/audios/test'
    test_data = pd.read_csv('dataset/csvs/test.csv')
    
    pipeline = ScorePredict()
    # TRAINING 
    audio_files = []
    for file in test_data['filename']:
        file_path = os.path.join(test_audio_path, file +'.wav')
        audio_files.append(file_path)
    #print("Extracting features from audio files...")
    #X, transcriptions = pipeline.prepare_training_data(audio_files)
    #joblib.dump(X, 'test_dataset.joblib')
    X = joblib.load('test_dataset.joblib')
    predictions = pipeline.predict(X)
    summary_df = test_data.copy()
    summary_df['label'] = np.round(predictions,4)

    summary_df.to_csv('submission.csv', index=False)
    print('DONEEE')

