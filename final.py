import deepspeech as ds
from DeepSpeechAudio import DeepSpeechAudio
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

from_sec = 0
to_sec = 20
CHANNELS = 1

# Initialize the DeepSpeech model and scorer
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'deepspeech.tflite')
scorer_path = 'deepspeech-0.9.3-models.scorer'  # Path to your scorer file

model = ds.Model(model_path)
model.enableExternalScorer(scorer_path)

def audioPro(model, audio_data):
    try:
        ds_aud = DeepSpeechAudio(audio_data, channels=CHANNELS)
        data = ds_aud.get_portion(from_sec, to_sec).get_array_of_samples()
        input_portion = np.array(data, dtype=np.int16)
        batch_text = model.stt(input_portion)
        return batch_text
    except Exception as e:
        print('Exception at audioPro:', str(e))
        return str(e)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['file'].read()
    transcription = audioPro(model, audio_file)
    
    return jsonify({"transcription": transcription})

if __name__ == "__main__":
    app.run(debug=True)

