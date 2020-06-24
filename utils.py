from typing import Dict, List, Optional
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
from pandarallel import pandarallel
import torchtext
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
import spacy
import warnings
import re
import random
import math
import time
from janome.tokenizer import Tokenizer as JTokenizer
import seaborn as sns
import numpy as np
import pandas as pd


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len, src_tokenize):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.lower() for token in src_tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention  # torch.flip(input=attention, dims=(2,))[1: -1]


def evaluate_blue(ev_data, src_field, trg_field, model, device):
    n_gram_weights = [1 / 3] * 3

    test_len = len(ev_data)

    original_texts = []
    generated_texts = []
    macro_bleu = 0

    for example_idx in range(test_len):
        src = vars(ev_data.examples[example_idx])['src']
        trg = vars(ev_data.examples[example_idx])['trg']
        translation, _ = translate_sentence(src, src_field, trg_field, model, device)

        original_texts.append(trg)
        generated_texts.append(translation)

        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [trg[::-1]],  #
            translation[: -1],
            weights=n_gram_weights
        )
        macro_bleu += bleu_score
    macro_bleu /= test_len

    return macro_bleu


def train(model, iterator, optimizer, criterion, clip, limit: Optional[float] = 1, accumulation_steps=1):
    model.train()

    epoch_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(iterator):

        if i / len(iterator) > limit:
            break

        src = batch.src
        trg = batch.trg

        output, _ = model(src, trg[:, :-1])  #

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        epoch_loss += loss.item()

        loss /= accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()
    optimizer.step()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
