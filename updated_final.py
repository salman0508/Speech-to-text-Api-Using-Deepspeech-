import os
from flask import Flask, request, jsonify
import deepspeech
import wave
import numpy as np

app = Flask(__name__)

from_sec = 0
to_sec = 20
CHANNELS = 1

# Initialize the DeepSpeech model and scorer
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'deepspeech-0.9.3-models.tflite')
scorer_path = 'deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)

def audioPro(model, audio_data, sample_rate):
    try:
        # Open the WAV file and read it using the wave module
        with wave.open(audio_data, 'rb') as wf:
            sample_width = wf.getsampwidth()
            audio_data = wf.readframes(wf.getnframes())
        
        if sample_width == 2:
            # If the sample width is 2 (int16), proceed with the transcription
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            batch_text = model.stt(audio_data)
            return batch_text
        else:
            return "Sample width is not 2 (int16), unsupported format"
    except Exception as e:
        print('Exception at audioPro:', str(e))
        return str(e)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['file']

    sample_rate = 16000  # Adjust the sample rate if needed

    transcription = audioPro(model, audio_file, sample_rate)

    return jsonify({"transcription": transcription})

if __name__ == "__main__":
    app.run(debug=True)

