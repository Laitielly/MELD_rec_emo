import numpy as np
import text_probs_embs as tpe
import audio_probs_embs as ape
import using_video_model as uvm
import early_fusion as ef
from text_transformer import RobertaClass
import speech_to_text as stt
import multimodal_late_fusion as mlf
from using_video_model import YourModule
import torch


def start_predict(type_of_fusion: str, path_to_video: str) -> str:
    ape.make_audio(path_to_video)

    if type_of_fusion == 'early_fusion':
        embs_audio = ape.make_csv_openSMILE()

        text = stt.get_the_text()
        print(text[0])
        embs_text = tpe.get_embs_text(text)

        prob_video = uvm.get_video_embs(path_to_video)

        return ef.early_fusion(embs_text, embs_audio, prob_video), text

    elif type_of_fusion == 'latest_fusion':
        prob_audio = ape.get_audio_prob(ape.make_csv_openSMILE())

        text = stt.get_the_text()
        prob_text = tpe.get_probs_text(text)

        prob_video = uvm.get_video_probs(path_to_video)

        return mlf.get_multimodal_pred(prob_text, prob_audio, prob_video), text