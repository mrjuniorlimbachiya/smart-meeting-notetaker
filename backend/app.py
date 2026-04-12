import sys
import os
import threading
import webbrowser
import onnxruntime_qnn

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from audio_capture import start_recording, stop_recording
from transcriber import transcribe_audio
from summariser import summarise_transcript

QNN_HTP_PATH = onnxruntime_qnn.get_qnn_htp_path()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = r'C:\Users\limba\smart-meeting-notetaker\frontend'

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/status')
def status():
    return jsonify({ 'ready': True })

@app.route('/record/start', methods=['POST'])
def record_start():
    start_recording()
    return jsonify({ 'status': 'recording' })

@app.route('/record/stop', methods=['POST'])
def record_stop():
    stop_recording()
    return jsonify({ 'status': 'saved', 'file': 'recording.wav' })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    wav_path = os.path.join(BASE_DIR, 'recording.wav')
    transcript = transcribe_audio(wav_path)
    return jsonify({ 'transcript': transcript })

@app.route('/summarise', methods=['POST'])
def summarise():
    data = request.get_json()
    transcript = data.get('transcript', '')
    result = summarise_transcript(transcript)
    return jsonify(result)

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    print('Starting NoteFlow at http://localhost:5000')
    app.run(port=5000, debug=False)