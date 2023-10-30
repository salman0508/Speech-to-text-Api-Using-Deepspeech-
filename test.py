import deepspeech as ds
from DeepSpeechAudio import DeepSpeechAudio
import numpy as np
import os

audio_path = 'audio.wav'
from_sec = 0
to_sec = 20
CHANNELS = 1

def audioPro(model, audio_path, from_sec, to_sec):
    try:
        scorer_path = 'deepspeech-0.9.3-models.scorer'  # Path to your scorer file
        model.enableExternalScorer(scorer_path)
        
        ds_aud = DeepSpeechAudio(audio_path, channels=CHANNELS)
        data = ds_aud.get_portion(from_sec, to_sec).get_array_of_samples()
        input_portion = np.array(data, dtype=np.int16)
        batch_text = model.stt(input_portion)
        print(batch_text)
    except Exception as e:
        print('Exception at audioPro:', str(e))

def loadModel(model_path):
    try:
        model = ds.Model(model_path)
        return model
    except Exception as e:
        print('Exception at loadModel:', str(e))
        return None

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, 'deepspeech.tflite')
    model = loadModel(model_path)
    if model is not None:
        audioPro(model, audio_path, from_sec, to_sec)

