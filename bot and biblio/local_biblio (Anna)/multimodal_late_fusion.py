import pickle
from using_video_model import YourModule
import pandas as pd
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch import nn
from scipy.special import softmax
import numpy as np

def rename_columns(df):
    df = df.rename(columns={0: 'anger_t', 1: 'disgust_t', 2: 'fear_t', 3: 'sadness_t', 4: 'neutral_t', 5: 'joy_t', 6: 'surprise_t',
                       7: 'anger_a', 8: 'disgust_a', 9: 'fear_a', 10: 'sadness_a', 11: 'neutral_a',
                       12: 'joy_a', 13: 'surprise_a', 14: 'anger_v', 15: 'disgust_v',
                       16: 'fear_v', 17: 'joy_v', 18: 'neutral_v', 19: 'sadness_v', 20: 'surprise_v'},
                    inplace=True)
    return df

def get_multimodal_pred(text_probs, audio_probs, video_probs):
    text_probs = np.hstack([text_probs, audio_probs])
    text_probs = np.hstack([text_probs, video_probs])
    probs = []
    probs = pd.DataFrame(probs)
    for i in range(len(text_probs)):
        probs.loc[1, i] = text_probs[i]
    rename_columns(probs)
    with open('models/multimodel/late_fusion_probs_model.pkl', 'rb') as file:
        model = pickle.load(file)


    result = model.predict(probs)
    return result[0]


## Пример
# tb = [0.0, 0.06, 0.0, 0.2, 0.04, 0.0, 0.0]
# ab = [0.0, 0.06, 0.0, 0.2, 0.04, 0.0, 0.0]
# vb = [0.0, 0.06, 0.0, 0.2, 0.04, 0.0, 0.0]
# get_multimodal_pred(np.array(tb), np.array(ab), np.array(vb))
