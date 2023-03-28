import torch
from torch.utils.data import Dataset
from transformers import RobertaModel


parameters = {'max_len': 256,
              'batch_size': 1
              }

phrase_params = {'batch_size': 1,
                 'shuffle': False,
                 'num_workers': 0
                 }


class SentimentData(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int) -> dict:
        text = self.text[index]
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 512)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(512, 7)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        pooler_output = output_1.pooler_output
        pooler_output = self.pre_classifier(pooler_output)
        pooler_output = torch.nn.ReLU()(pooler_output)
        pooler_output = self.dropout(pooler_output)
        output = self.classifier(pooler_output)

        return output