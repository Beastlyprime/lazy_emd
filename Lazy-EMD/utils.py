import sys
import os
import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import BertConfig, XLNetConfig, XLMConfig, RobertaConfig
from transformers import AutoModel, GPT2Tokenizer

# --------------------------------
import numpy as np
import ot
# --------------------------------

from . import __version__

__all__ = ['model_types']

SCIBERT_URL_DICT = {
    'scibert-scivocab-uncased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar', # recommend by the SciBERT authors
    'scibert-scivocab-cased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar',
    'scibert-basevocab-uncased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar',
    'scibert-basevocab-cased':  'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar',
}

model_types = list(BertConfig.pretrained_config_archive_map.keys()) + \
              list(XLNetConfig.pretrained_config_archive_map.keys()) + \
              list(RobertaConfig.pretrained_config_archive_map.keys()) + \
              list(XLMConfig.pretrained_config_archive_map.keys()) + \
              list(SCIBERT_URL_DICT.keys())

lang2model = defaultdict(lambda: 'bert-base-multilingual-cased')
lang2model.update({
    'en': 'roberta-large',
    'zh': 'bert-base-chinese',
    'en-sci': 'scibert-scivocab-uncased',
})


model2layers = {
    'bert-base-uncased': 9,
    'bert-large-uncased': 18,
    'bert-base-cased-finetuned-mrpc': 9,
    'bert-base-multilingual-cased': 9,
    'bert-base-chinese': 8,
    'roberta-base': 10,
    'roberta-large': 17,
    'roberta-large-mnli': 19,
    'xlnet-base-cased': 5, 
    'xlnet-large-cased': 7, 
    'xlm-mlm-en-2048': 7, 
    'xlm-mlm-100-1280': 11,
    'scibert-scivocab-uncased': 9,
    'scibert-scivocab-cased': 9,
    'scibert-basevocab-uncased': 9,
    'scibert-basevocab-cased':  9,
    'distilroberta-base': 5,
}

def make_normal(array):
    array_len = sum(array > 0)
    array[array_len] -= sum(array) - 1.
    return array

def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    if isinstance(tokenizer, GPT2Tokenizer):
        # for RoBERTa and GPT-2
        return tokenizer.encode(sent.strip(), add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=tokenizer.max_len)
    else:
        return tokenizer.encode(sent.strip(), add_special_tokens=True,
                                max_length=tokenizer.max_len)


def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith('scibert'):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    # drop unused layers
    if not all_layers:
        if hasattr(model, 'n_layers'): # xlm
            model.n_layers = num_layers
        elif hasattr(model, 'layer'): # xlnet
            model.layer =\
                torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, 'encoder'): # bert, roberta
            model.encoder.layer =\
                torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, 'output_hidden_states'):
            model.output_hidden_states = True
        else:
            model.encoder.output_hidden_states = True
    return model


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_idf(arr, tokenizer, idf_dict, device='cuda:0'):
    """
    Helper function that pads a list of sentences to have the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0', 
                       all_layers=False):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size],
                                          all_layers=all_layers)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def greedy_cos_idf(ref_embedding, ref_masks, ref_idf,
                   hyp_embedding, hyp_masks, hyp_idf,
                   epsilon=0.009, reg1=0.23, reg2=0.31, 
                   lemd=False,
                   all_layers=False):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
        - :param: `epsilon` (float): Entropy regularization parameter
        - :param: `reg1` (float): Margin relaxation parameter 1
        - :param: `reg2` (float): Margin relaxation parameter 2
    """

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = hyp_embedding.transpose(1, 2).transpose(0, 1)\
            .contiguous().view(L*B, hyp_embedding.size(1), D)
        ref_embedding = ref_embedding.transpose(1, 2).transpose(0, 1)\
            .contiguous().view(L*B, ref_embedding.size(1), D)
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1)\
                                  .contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1)\
                                  .contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    if lemd:
        scores_batch = []
        hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
        ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
        sim_np = sim.cpu().numpy()
        for i in range(sim.shape[0]):
            hyp = hyp_idf[i].cpu().numpy()
            ref = ref_idf[i].cpu().numpy()
            cost = 1. - 1. * sim_np[i]
            P = ot.unbalanced.sinkhorn_unbalanced(hyp, ref, cost, epsilon, reg1, reg2, method='sinkhorn', verbose=False, numItermax=200000)
            scores_batch.append(np.sum(P * cost))
        scores_batch = torch.tensor(scores_batch).float()
        return scores_batch
    else:
        sim_precision = sim.max(dim=2)
        word_precision = sim_precision.values
        sim_recall = sim.max(dim=1)
        word_recall = sim_recall.values

        hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
        precision_scale = hyp_idf.to(word_precision.device)
        ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
        recall_scale = ref_idf.to(word_recall.device)
        
        if all_layers:
            precision_scale = precision_scale.unsqueeze(0)\
                .expand(L, B, -1).contiguous().view_as(word_precision)
            recall_scale = recall_scale.unsqueeze(0)\
                .expand(L, B, -1).contiguous().view_as(word_recall)
        P = (word_precision * precision_scale).sum(dim=1)
        R = (word_recall * recall_scale).sum(dim=1)
        F = 2 * P * R / (P + R)

        hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
        ref_zero_mask = ref_masks.sum(dim=1).eq(2)

        if all_layers:
            P = P.view(L, B)
            R = R.view(L, B)
            F = F.view(L, B)

        if torch.any(hyp_zero_mask):
            print("Warning: Empty candidate sentence; Setting precision to be 0.", file=sys.stderr)
            P = P.masked_fill(hyp_zero_mask, 0.)

        if torch.any(ref_zero_mask):
            print("Warning: Empty candidate sentence; Setting recall to be 0.", file=sys.stderr)
            R = R.masked_fill(ref_zero_mask, 0.)

        F = F.masked_fill(torch.isnan(F), 0.)

        return P, R, F

def bert_cos_score_idf(model, refs, hyps, tokenizer, idf_dict,
                       epsilon, reg1, reg2,
                       verbose=False, batch_size=64, device='cuda:0',
                       lemd=False,
                       all_layers=False):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
        - :param: `lemd` (bool): use unbalanced OT
        - :param: `epsilon` (float): Entropy regularization parameter
        - :param: `reg1` (float): Margin relaxation parameter 1
        - :param: `reg2` (float): Margin relaxation parameter 2
    """
    preds = []
    def dedup_and_sort(l):
        return sorted(list(set(l)), key= lambda x : len(x.split(" ")))
    sentences = dedup_and_sort(refs+hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose: 
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start:batch_start+batch_size]
        embs, masks, padded_idf = get_bert_embedding(sen_batch, model, tokenizer, idf_dict,
                                                     device=device, all_layers=all_layers)
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)
        
    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.)
        idf_pad = pad_sequence(idf, batch_first=True)
        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long)\
                        .expand(len(lens), max_len)
            return base < lens.unsqueeze(1)
        pad_mask = length_to_mask(lens)
        return emb_pad.to(device), pad_mask.to(device), idf_pad.to(device)
        

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose: 
        if lemd:
            print("computing Lazy EMD.")
        else:
            print("computing greedy matching.")
        iter_range = tqdm(iter_range)
    for batch_start in iter_range:
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
        hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)
        if lemd:
            scores = greedy_cos_idf(*ref_stats, *hyp_stats, epsilon, reg1, reg2, lemd=lemd, all_layers=all_layers)
            preds.append(scores.cpu())
        else:
            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers=all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


def get_hash(model, num_layers, idf):
    msg = '{}_L{}{}_version={}'.format(
        model, num_layers, '_idf' if idf else '_no-idf', __version__)
    return msg


def cache_scibert(model_type, cache_folder='~/.cache/torch/transformers'):
    if not model_type.startswith('scibert'):
        return model_type

    underscore_model_type = model_type.replace('-', '_')
    cache_folder = os.path.abspath(cache_folder)
    filename = os.path.join(cache_folder, underscore_model_type)

    # download SciBERT models
    if not os.path.exists(filename):
        cmd = f'mkdir -p {cache_folder}; cd {cache_folder};'
        cmd += f'wget {SCIBERT_URL_DICT[model_type]}; tar -xvf {underscore_model_type}.tar;'
        cmd += f'rm -f {underscore_model_type}.tar ; cd {underscore_model_type}; tar -zxvf weights.tar.gz; mv weights/* .;'
        cmd += f'rm -f weights.tar.gz; rmdir weights; mv bert_config.json config.json;'
        print(cmd)
        print(f'downloading {model_type} model')
        os.system(cmd)

    # fix the missing files in scibert
    json_file = os.path.join(filename, 'special_tokens_map.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print('{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}', file=f)

    json_file = os.path.join(filename, 'added_tokens.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print('{}', file=f)

    if 'uncased' in model_type: 
        json_file = os.path.join(filename, 'tokenizer_config.json')
        if not os.path.exists(json_file):
            with open(json_file, 'w') as f:
                print('{"do_lower_case": true, "max_len": 512, "init_inputs": []}', file=f)

    return filename
