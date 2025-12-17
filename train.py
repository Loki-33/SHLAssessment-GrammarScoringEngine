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
from tqdm import tqdm 
import joblib
from transcribe import AudioTranscribe
from analyze import AudioAnalyzer, GrammarAnalyzer 
import warnings 
warnings.filterwarnings('ignore')

class Scoring:
    """
    After extraction of various features we then use XGBoost model 
    to train the model on various audio files and evaluate them on the 
    different metrics like RMSE, MAE, CORRELATION, ACCURACY and then use 
    the trianed model to predict the score for the various audio files 
    ranging from 0-5 
    """
    def __init__(self, whisper_model_size = 'base'):
        self.transcriber = AudioTranscribe(whisper_model_size)
        self.grammar_analyzer = GrammarAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.scaler = StandardScaler()
        self.model = None 
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

    def prepare_training_data(self, audio_paths, labels):
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

        return df, labels, transcription


    def train(self, X_train, y_train, X_val=None, y_val=None):
        print('\nTRAINING MODEL...........')
        #X_train_scaled = self.scaler.fit_transform(X_train)
        
        params = {
            'objective': 'reg:absoluteerror',
            'max_depth':6,
            'learning_rate':0.02,
            'n_estimators': 1200,
            'min_child_weight': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'verbose': False
        }

        self.model = xgb.XGBRegressor(**params)
        if X_val is not None and y_val is not None:
            #X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            self.model.fit(X_train, y_train)
        print('TRAINING COMPLETE')
        self.print_feature_importance()
        
    def predict(self, X):
        #X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 0, 5)

        return predictions 

    def evaluate(self,X, y_true):
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Accuracy within ±0.5
        within_half = np.mean(np.abs(y_true - y_pred) <= 0.5)
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        print(f"\n{'='*50}")
        print("EVALUATION METRICS")
        print(f"{'='*50}")
        print(f"RMSE:                    {rmse:.4f}")
        print(f"MAE:                     {mae:.4f}")
        print(f"Accuracy (±0.5):         {within_half*100:.2f}%")
        print(f"Correlation:             {correlation:.4f}")
        print(f"{'='*50}\n")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'accuracy_half': within_half,
            'correlation': correlation
        }

    def print_feature_importance(self, top_n=15):
        if self.model is None:
            print("Model not trained yet!")
            return
        
        importance = self.model.get_booster().get_score(importance_type="gain")
        
        feature_importance = (pd.DataFrame(importance.items(), columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
    )
        
        print(f"\n{'='*50}")
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print(f"{'='*50}")
        for i, row in feature_importance.head(top_n).iterrows():
            print(f"{row['feature']} {row['importance']}")
        print(f"{'='*50}\n")

    def save_model(self, path='grammar_scoring_model.json'):
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        print(f"Model saved to {path}")

if __name__ == '__main__':
    import os 
    train_audio_path = 'dataset/audios/train'
    test_audio_path = 'dataset/audios/test'

    train_data = pd.read_csv('dataset/csvs/train.csv')
    test_data = pd.read_csv('dataset/csvs/test.csv')
    
    pipeline = Scoring()
    # TRAINING 
    audio_files = []
    labels = train_data['label']
    for file in train_data['filename']:
        file_path = os.path.join(train_audio_path, file +'.wav')
        audio_files.append(file_path)
    #print("Extracting features from audio files...")
    #X, y, transcriptions = pipeline.prepare_training_data(audio_files, labels)
    #joblib.dump((X, y), 'train_dataset.joblib')
    X, y = joblib.load('train_dataset.joblib')
    #X_real = X.iloc[:, [1]]
    #X_exloded = X.explode('segments')
    #X_1 = pd.json_normalize(X_exloded['segments'])
    #print(X_1.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.train(X_train, y_train, X_val, y_val)

    pipeline.evaluate(X_val, y_val)

    pipeline.save_model('grammar_scoring_model.json')
