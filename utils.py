# from builtins import function
from typing import Dict, List, Optional, Union
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
import spacy
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import warnings
import time
from janome.tokenizer import Tokenizer as JTokenizer
import seaborn as sns
import numpy as np
import pandas as pd
from torchtext.data import Field


def count_parameters(model: nn.Module) -> int:
    """
    Считает количество параметров для модели model
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """
    Инициализирует веса модели
    :param model:
    :return:
    """
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)


def translate_sentence(sentence: Union[str, List[str]],
                       src_field: Field, trg_field: Field,
                       model: nn.Module, device: torch.device,
                       max_len: int, src_tokenize=None):
    """
    Функция генерирует перевод исходного предложения `sentence` моделью `model`
    :param sentence: List[str] список токено исходного предложения
    :param src_field:
    :param trg_field:
    :param model:
    :param device:
    :param max_len: максимальная длина целевого предложения
    :param src_tokenize:
    :return:
    """
    model.eval()

    if isinstance(sentence, str):
        raise NotImplemented
        # tokens = [token.lower() for token in src_tokenize(sentence)]
        # TODO убрать перевод строки
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


def evaluate_blue(ev_data, src_field, trg_field, model, device, max_len, src_tokenize):
    """

    :param ev_data:
    :param src_field:
    :param trg_field:
    :param model:
    :param device:
    :param max_len:
    :param src_tokenize:
    :return:
    """
    n_gram_weights = [1 / 3] * 3

    test_len = len(ev_data)

    original_texts = []
    generated_texts = []
    macro_bleu = 0

    for example_idx in range(test_len):
        src = vars(ev_data.examples[example_idx])['src']
        trg = vars(ev_data.examples[example_idx])['trg']
        translation, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, src_tokenize)

        original_texts.append(trg)
        generated_texts.append(translation)

        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [trg],  #
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


def display_attention(sentence: List[str],
                      translation,
                      attention,
                      n_heads,
                      n_rows,
                      n_cols,
                      fontprop_x=FontProperties(),
                      fontprop_y=FontProperties(),
                      reverse_src=False,
                      reverse_trg=False
                      ):
    """

    :param sentence:
    :param translation:
    :param attention:
    :param n_heads:
    :param n_rows:
    :param n_cols:
    :param fontprop_x:
    :param fontprop_y:
    :param reverse_src:
    :param reverse_trg:
    :return:
    """
    assert n_rows * n_cols == n_heads

    if reverse_src:
        attention = torch.flip(attention, (3,))
        sentence.reverse()
    elif reverse_trg:
        attention = torch.flip(attention, (2,))
        translation.reverse()

    fig = plt.figure(figsize=((n_cols + 1.5) * 6, (n_rows + 1) * 6))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=20)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],  #
                           rotation=45, fontproperties=fontprop_x)
        ax.set_yticklabels([''] + translation, fontproperties=fontprop_y)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def sentence_blue(original: List[str], translation: List[str], n_grams=3) -> float:
    """
    Возвращает BLUE для одного предложения
    :param original:
    :param translation:
    :param n_grams:
    :return:
    """
    blue = nltk.translate.bleu_score.sentence_bleu(
        [original],
        translation[: -1],
        weights=[1 / n_grams] * n_grams
    )
    return blue
