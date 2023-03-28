import librosa
import opensmile
import numpy as np
from pickle import load as download
from moviepy.editor import VideoFileClip


def open_pickle(path):
    f = open(path, 'rb')
    model = download(f)
    f.close()

    return model


def make_audio(path_to_video):
    clip = VideoFileClip(path_to_video, verbose=False)
    pathto = 'data/utt.mp3'
    clip.audio.write_audiofile(pathto, verbose=False)

    return


def make_csv_openSMILE(path_audio='data/utt.mp3', s=22050):
    smile = opensmile.Smile()

    signal, sr = librosa.load(path_audio, sr=s)
    frame_audio = smile.process_signal(signal, sr)

    return np.array(np.float_(frame_audio.iloc[0].values.flatten().tolist()))


def get_weight_classifiers():
    scaler = open_pickle('models/audio_models/scaler.pickle')
    pca = open_pickle('models/audio_models/PCA.pickle')
    classifier = open_pickle('models/audio_models/SVC_model_audio_36score.pickle')

    return scaler, pca, classifier


def get_audio_prob(open_row: np.array) -> np.array:
    scaler, pca, classifier = get_weight_classifiers()

    row = np.array(open_row)
    row = row.reshape(-1, 1).T
    row = scaler.transform(row)
    row = pca.transform(row)
    probs = classifier.predict_proba(row)

    return np.array(probs[0])
