import whisper 

'''
Using Whisper model to get transcription of the audio files along with 
the timestamps of each words 
'''
class AudioTranscribe():
    def __init__(self, model_size='base'):
        print(f"Loading Whisper {model_size} model....")
        self.model = whisper.load_model(model_size)


    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path, word_timestamps=True)
        return {
            'text': result['text'],
            'segments': result['segments'],
            'language': result['language']
        }


