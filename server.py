from __future__ import unicode_literals
from flask import Flask
from flask import request, jsonify
import yt_dlp
import ffmpeg
import sys
import librosa
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def extract_audio_features(file_path):
    print("in")
    try:
        sr = 44100
        # Chargement du fichier audio
        y, sr = librosa.load(file_path, sr=sr)

        # spectrogram, tempo, chroma, mfccs, spectral_contrast

        spectrogram = np.abs(librosa.stft(y))
        spectrogram_min = np.min(spectrogram)
        spectrogram_max = np.max(spectrogram)
        spectrogram_mean = np.mean(spectrogram)
        spectrogram_std = np.std(spectrogram)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_min = np.min(chroma)
        chroma_max = np.max(chroma)
        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)

        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs_min = np.min(mfccs)
        mfccs_max = np.max(mfccs)
        mfccs_mean = np.mean(mfccs)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_min = np.min(spectral_contrast)
        spectral_contrast_max = np.max(spectral_contrast)
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_std = np.std(spectral_contrast)

        audio_features = {
            'spectrogram_min': spectrogram_min,
            'spectrogram_max': spectrogram_max,
            'spectrogram_mean': spectrogram_mean,
            'spectrogram_std': spectrogram_std,
            'tempo': tempo,
            'chroma_min': chroma_min,
            'chroma_max': chroma_max,
            'chroma_mean': chroma_mean,
            'chroma_std': chroma_std,
            'mfccs_min': mfccs_min,
            'mfccs_max': mfccs_max,
            'mfccs_mean': mfccs_mean,
            'spectral_contrast_min': spectral_contrast_min,
            'spectral_contrast_max': spectral_contrast_max,
            'spectral_contrast_mean': spectral_contrast_mean,
            'spectral_contrast_std': spectral_contrast_std
        }
        print("in", audio_features)
        return audio_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"

# Route post that will have an url as request body
@app.route("/download", methods=['POST'])
@cross_origin() 
def download():
    print("-------------------", request.form.get('url'))
    url = request.form.get('url')
    print("url", url)
    download_from_url(url)
    return "Downloaded"

ydl_opts = {
    'format': 'bestaudio/best',
   'outtmpl': './temp/output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}
def download_from_url(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        features = extract_audio_features('./temp/output.wav')
        if features is None:
            return None

        model = load_model('./models/best_model_0899.h5')

        label_dict = {
            0: 'blues',
            1: 'classical',
            2: 'country',
            3: 'disco',
            4: 'hiphop',
            5: 'jazz',
            6: 'metal',
            7: 'pop',
            8: 'reggae',
            9: 'rock'
        }

        # predict
        model.predict(features)

        # print predicted label
        print(label_dict[np.argmax(model.predict(features))])