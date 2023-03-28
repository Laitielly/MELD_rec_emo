import pandas as pd
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch import nn
from scipy.special import softmax
import numpy as np


class YourModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(512, 7)
        self.optimizer = torch.optim.Adam(self.classifier.parameters())

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return self.classifier(features)

    def configure_optimizers(self):
        return [self.optimizer], [torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self.criterion(predictions, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self.criterion(predictions, y)
        with torch.no_grad():
            acc = ((torch.argmax(predictions, dim=1) == y).sum() / y.shape[0]).item()
        self.log("Validation Loss", loss, prog_bar=True, on_epoch=True)
        self.log("Validation Accuracy", acc, prog_bar=True, on_epoch=True)


class GetEmbs(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.model.fc = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features


def preprocess_video(path):  # Общая функция для выделения эмбеддингов и вероятностей
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)  # Инициализируем нейросеть для детектирования лиц

    path_dir = './frames_of_input'  # Создаем каталог для фреймов

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    for f in os.listdir(path_dir):
        os.remove(os.path.join(path_dir, f))  # Очищаем каталог перед работой с новым видео

    # Выдираем каждый 10-ый фрейм и сохраняем в папку
    video = cv2.VideoCapture(path)
    success = True
    i = 1
    j = 10

    while (True):
        success, frame = video.read()
        if success and i % j == 0 and i >= 10 or i == 1:
            cv2.imwrite(f'{path_dir}/{i}.jpg', frame)
        elif not success:
            break
        i += 1

    frames_without_faces = []
    # Выделяем лица и пересохраняем обрезанные по лицам картинки с измененным размером
    pointer = list(os.walk(path_dir))
    for frame in pointer[0][2]:
        photo = plt.imread(path_dir + '/' + frame)
        boxes, probs = mtcnn.detect(photo)
        if boxes is not None:
            k = 0
            m = 0
            max_p = probs[0]
            for s in probs:
                if (s > max_p):
                    m = k
                    max_p = s
                k += 1
            if max_p < 0.85:
                frames_without_faces.append(path_dir + '/' + frame)
            else:
                X_0 = boxes[m][0]
                Y_0 = boxes[m][1]
                X_1 = boxes[m][2]
                Y_1 = boxes[m][3]
                im = Image.open(path_dir + '/' + frame)
                img = im.crop((X_0, Y_0, X_1, Y_1))
                img.resize((224, 224))
                img.save(path_dir + '/' + frame)
        else:
            frames_without_faces.append(path_dir + '/' + frame)

    for each in frames_without_faces:  # Убираем фреймы без лиц из директории
        os.remove(each)
    return path_dir


def get_video_probs(path):  # Вход - дорога к видео, выход - массив вероятностей по классам
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path_dir = preprocess_video(path)

    if len(os.listdir(path_dir)) == 0:  # Если ничего не осталось, то возвращаем нулевой вектор или предсказания -1
        result_probs = [-1, -1, -1, -1, -1, -1, -1]
        return np.array(result_probs)

    model = torchvision.models.resnet18(weights='DEFAULT')
    module = YourModule(model)

    module = torch.load('models/video_models/myResnetModel.bin', map_location=torch.device(device))
    module.eval()

    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
    probs = []
    preds = []
    pointer = list(os.walk(path_dir))
    for frame in pointer[0][2]:
        img = Image.open(path_dir + '/' + frame)

        input = transform(img)
        input = torch.stack([input])

        module.eval()
        input = input.to(device)
        module.to(device)
        y_p = module(input)
        y_p = y_p.detach().numpy()
        y_p = list(softmax(y_p))
        preds.append(y_p[0])
    for i in range(7):
        sum = 0
        for one in preds:
            sum += one[i]
        probs.append(sum / len(preds))

    return np.array(probs)


def get_video_embs(path):  # Вход - дорога к видео, выход - массив из 512 значений
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path_dir = preprocess_video(path)

    if len(os.listdir(path_dir)) == 0:  # Если ничего не осталось, то возвращаем нулевой вектор или предсказания -1
        result_embs = list(np.zeros(512))
        #result_embs = vec_z
        return np.array(result_embs)

    model = torchvision.models.resnet18(weights='DEFAULT')
    module_for_embs = GetEmbs(model)

    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
    vec = []
    preds = []
    pointer = list(os.walk(path_dir))
    for frame in pointer[0][2]:
        img = Image.open(path_dir + '/' + frame)

        input = transform(img)
        input = torch.stack([input])

        input = input.to(device)
        module_for_embs.to(device)
        y_p = module_for_embs.forward(input)
        y_p = y_p.detach().numpy()
        preds.append(list(y_p[0]))
    for i in range(len(preds[0])):
        sum = 0
        for one in preds:
            sum += one[i]
        vec.append(sum / len(preds))
    return np.array(vec)
