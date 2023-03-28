import text_transformer as tt
from text_transformer import RobertaClass

import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer


def get_probs_text(text: np.array) -> np.array:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = torch.load('models/text_models/pytorch_roberta_sentiment_65-32.bin',
                       map_location=torch.device(device))

    phrase = tt.SentimentData(text, tokenizer, tt.parameters['max_len'])
    phrase = DataLoader(phrase, **tt.phrase_params)

    probabilities = []
    with torch.no_grad():
        for data in phrase:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            probabilities = list(softmax(np.array(outputs.cpu()))[0])

    return probabilities


def get_embs_text(text: np.array) -> np.array:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = torch.load('models/text_models/pytorch_roberta_sentiment_65-32.bin',
                       map_location=torch.device(device))

    phrase = tt.SentimentData(text, tokenizer, tt.parameters['max_len'])
    phrase = DataLoader(phrase, **tt.phrase_params)

    emb_pooler = []
    with torch.no_grad():
        for _, data in (enumerate(phrase, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            pooler_outp = model.l1(input_ids=ids, attention_mask=mask,
                                   token_type_ids=token_type_ids).pooler_output

            emb_pooler = np.array(pooler_outp[0].tolist())

    return emb_pooler
