import numpy as np
from pickle import load as download
from text_transformer import RobertaClass


def open_pickle(path):
    f = open(path, 'rb')
    model = download(f)
    f.close()

    return model


def get_at_scalers(tp_scaler: str) -> tuple:

    scaler_audio = open_pickle(f'models/{tp_scaler}_models/scaler_{tp_scaler}.pickle')
    pca_audio = open_pickle(f'models/{tp_scaler}_models/PCA_{tp_scaler}.pickle')

    return scaler_audio, pca_audio


def audio_for_early_fusion(audio: np.array) -> np.array:
    audio = audio.reshape(-1, 1).T

    scaler, pca = get_at_scalers('audio')

    audio_row = scaler.transform(audio)
    audio_row = pca.transform(audio_row).tolist()[0]

    return np.array(audio_row)


def text_for_early_fusion(text: np.array) -> np.array:
    text = text.reshape(-1, 1).T

    scaler, pca = get_at_scalers('text')

    text_row = scaler.transform(text)
    text_row = pca.transform(text_row).tolist()[0]

    return np.array(text_row)


def video_for_early_fusion(video: np.array) -> np.array:
    scaler = open_pickle('models/video_models/scaler_video.pickle')

    video_row = scaler.transform(
        np.array(video).reshape(-1, 1).T).tolist()[0]

    return np.array(video_row)


def early_fusion(text: np.array, audio:np.array, video: np.array):
    targets = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'sadness', 4: 'neutral', 5: 'joy',
               6: 'surprise'}

    text_emb = text_for_early_fusion(text)
    audio_emb = audio_for_early_fusion(audio)
    video_emb = video_for_early_fusion(video)

    vector = np.concatenate([text_emb, audio_emb, video_emb], axis=None).reshape(-1, 1).T

    classifier = open_pickle('models/multimodel/SVC_model_61_57score_pooler_768.pickle')
    target = classifier.predict(vector)

    return targets[target[0]]
