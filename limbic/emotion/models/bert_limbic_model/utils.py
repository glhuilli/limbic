import transformers

from limbic.emotion.models.bert_limbic_model.bert_limbic_model import BERTBaseUncased

from sklearn.metrics import label_ranking_average_precision_score

# import warnings
#
# warnings.filterwarnings('ignore')
#
# import logging
#
# logging.basicConfig(level=logging.ERROR)
#
# print('training....')

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = '../data/bert'
MODEL_PATH = "bert_model.bin"
TRAINING_FILE = "bert_train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

NUM_LABELS = 4


import pandas as pd


SENTENCE_EMOTIONS_TEST_FILE = '../data/sentence_emotions_test.pickle'
SENTENCE_EMOTIONS_TRAIN_FILE = '../data/sentence_emotions_train.pickle'
CONTINUES_TO_BINARY_THRESHOLD = 0.5


from limbic.limbic_constants import AFFECT_INTENSITY_EMOTIONS as EMOTIONS


def load_data_file(file_path):
    data = pd.read_pickle(file_path)

    data_sentences = data['text']  # .str.lower().apply(lambda x: utils.preprocess_sentence(x))
    y_data = data[EMOTIONS].values

    #     # This will be used throughout the notebook to compute performance
    #     y_data_labeled = utils.continuous_labels_to_binary(y_data, CONTINUES_TO_BINARY_THRESHOLD)

    #     # This representation will be needed for sklearn later in this notebook.
    #     x_data = tokenizer.texts_to_sequences(data_sentences)
    #     x_data = tf.keras.preprocessing.sequence.pad_sequences(x_data, maxlen=MAX_LEN)

    #     return data, x_data, y_data, y_data_labeled, data_sentences
    return data, y_data, data_sentences

#
# train, y_train_labeled, train_sentences = load_data_file(SENTENCE_EMOTIONS_TRAIN_FILE)
# # test, x_test, y_test, y_test_labeled, test_sentences = load_data_file(SENTENCE_EMOTIONS_TEST_FILE)
#
# print(f'train shape: {train.shape}')
# # print(f'test shape: {test.shape}')
#

import torch


class BERTDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }


import torch
import torch.nn as nn
from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter

# Tensorboard + PyTorch: https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
TB_WRITER = SummaryWriter()


def loss_fn(outputs, targets, num_labels):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, num_labels))


def train_fn(data_loader, model, optimizer, device, scheduler, num_labels, epoch):
    model.train()

    total_loss = 0
    n = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets, num_labels)
        total_loss += loss
        n += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

    TB_WRITER.add_scalar("Loss/train", total_loss / n, epoch)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            #             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def run():

    dfx, y_train_labeled, train_sentences = load_data_file(SENTENCE_EMOTIONS_TRAIN_FILE)

    df_train, df_valid = model_selection.train_test_split(
        dfx.sample(n=100, random_state=1), test_size=0.1, random_state=42,
        # stratify=dfx.sentiment.values # Consider iterative stratified
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = BERTDataset(
        text=df_train.text.values, target=df_train[EMOTIONS].values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = BERTDataset(
        text=df_valid.text.values, target=df_valid[EMOTIONS].values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in tqdm(range(EPOCHS), 'epochs'):
        train_fn(train_data_loader, model, optimizer, device, scheduler, NUM_LABELS, epoch)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        targets = np.array(np.array(targets) >= 0.1).astype(int)
        outputs = np.array(outputs)  # >= 0.5
        accuracy = label_ranking_average_precision_score(targets, outputs)
        print(f"Ranking Avg Precision Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy

    TB_WRITER.close()


def load_model(experiments_date: str):
    """
    Load the Tokenizer and the Model from disk.
    """
    model_path = os.path.join(_MODELS_PATH, f'emotions_model_{experiments_date}.h5')
    tokenizer_path = os.path.join(_MODELS_PATH, f'tokenizer_{experiments_date}.pickle')

    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    model = tf.keras.models.load_model(model_path)

    return model, tokenizer
