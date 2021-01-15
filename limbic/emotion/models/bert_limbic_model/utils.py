import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn import model_selection
from sklearn.metrics import label_ranking_average_precision_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from limbic.emotion.models.bert_limbic_model.bert_dataset import BERTDataset
from limbic.emotion.models.bert_limbic_model.bert_limbic_model import MAX_LEN, BERTBaseUncased
from limbic.limbic_constants import AFFECT_INTENSITY_EMOTIONS as EMOTIONS
from limbic.limbic_types import TrainBertParams

# TODO: Move to config file
_TARGETS_THRESHOLD = 0.1
_ADAM_LEARNING_RATE = 3e-5

# TODO: Review Tensorboard link below if there are other metrics that would be interesting to track
#   https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
TB_WRITER = SummaryWriter()


def load_data_file(file_path: str):  # TODO: Add typing
    data = pd.read_pickle(file_path)

    # TODO: Review how much processing we would like to add, considering that Bert needs words pretty much untouched.
    #   An option is to use `text.str.lower().apply(lambda x: utils.preprocess_sentence(x))`
    data_sentences = data['text']
    y_data = data[EMOTIONS].values
    return data, y_data, data_sentences


def loss_function(outputs, targets, num_labels):  # TODO: Add typing
    """
    Loss function used to re-train the BERT model.

    Using binary cross entropy logistic loss function as it's better suited for multi-label learning
    """
    return nn.BCEWithLogitsLoss()(outputs, targets.view(
        -1, num_labels))  # TODO: Verify if num_labels is used correctly


def train_model(data_loader, model, optimizer, device, scheduler, num_labels,
                epoch):  # TODO: Add typing
    """
    Train the model using batches of data from data_loader.

    Tracking results into the logs for Tensorboard, reason why I had to include the epoch.
    """
    model.train()
    total_loss = 0
    n = 0
    for d in tqdm(data_loader, total=len(data_loader), desc=f'training [{len(data_loader)}]'):
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

        loss = loss_function(outputs, targets, num_labels)
        total_loss += loss
        n += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

    TB_WRITER.add_scalar("Loss/train", total_loss / n, epoch)


def evaluate(data_loader, model, device):  # TODO: add typing
    """
    Method to evaluate the validation set after each training run.
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), desc=f'evaluate [{len(data_loader)}]'):
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
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def build_model(config: TrainBertParams):  # TODO: add typing
    """
    Runs the main loop to build the trained Bert model for multi-label emotion classification
    """

    tokenizer = transformers.BertTokenizer.from_pretrained(config.bert_path, do_lower_case=True)

    # TODO: consider adding stratify for multi-label
    dfx, _, _ = load_data_file(config.emotions_train_file)
    df_train, df_valid = model_selection.train_test_split(dfx.sample(n=config.training_sample,
                                                                     random_state=1),
                                                          test_size=config.test_size,
                                                          random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    train_dataset = BERTDataset(text=df_train.text.values,
                                target=df_train[EMOTIONS].values,
                                tokenizer=tokenizer,
                                max_len=MAX_LEN)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.train_batch_size,
                                                    num_workers=4)

    valid_dataset = BERTDataset(text=df_valid.text.values,
                                target=df_valid[EMOTIONS].values,
                                tokenizer=tokenizer,
                                max_len=MAX_LEN)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=config.valid_batch_size,
                                                    num_workers=1)

    device = torch.device(config.device)
    model = BERTBaseUncased(config.bert_path, config.bert_base_uncase_params)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = config.no_decay_components
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    num_train_steps = int(len(df_train) / config.train_batch_size * config.epochs)
    optimizer = AdamW(optimizer_parameters, lr=_ADAM_LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)

    current_date = datetime.now().date().isoformat()
    output_model_file = os.path.join(config.output_model_path, f'emo_bert_model_{current_date}.bin')
    best_accuracy = 0
    for epoch in tqdm(range(config.epochs), desc='epochs'):
        train_model(train_data_loader, model, optimizer, device, scheduler,
                    config.bert_base_uncase_params.num_labels, epoch)
        outputs, targets = evaluate(valid_data_loader, model, device)
        targets = np.array(np.array(targets) >= _TARGETS_THRESHOLD).astype(int)
        outputs = np.array(outputs)
        accuracy = label_ranking_average_precision_score(targets, outputs)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), output_model_file)
            best_accuracy = accuracy

    TB_WRITER.close()
