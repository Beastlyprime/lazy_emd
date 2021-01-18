# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenyimeng/anaconda3/lib/
# from overrides import overrides
from typing import List
from transformers import AutoTokenizer
import torch
import bert_score
from bert_score.utils import (
    get_model,
    get_idf_dict,
    bert_cos_score_idf,
    get_bert_embedding,
    lang2model,
    model2layers,
    get_hash,
    cache_scibert,
    sent_encode,
    bert_encode
)
import ot
import unbalanced

import numpy as np

def get_weight_mat(indexes, col):
    mat = np.zeros((len(indexes), col))
    for ind, i in enumerate(indexes):
        mat[ind, i] = 1
    return mat

def make_normal(array):
    array_len = sum(array > 0)
    array[array_len] -= sum(array) - 1.
    return array

def get_uniform_weight(length):
    w = np.zeros(length)
    w[1:-1] = 1.
    w = w / w.sum()
    return make_normal(w)


class Scorer:
    """
    - model_name: model name of bert model (e.g. bert-uncased-base, roberta-large)
    """
    def __init__(self, model_name):
        self.model_name = model_name
        num_layers = model2layers[self.model_name]
        self.bert_model = get_model(self.model_name, num_layers)
        if model_name.startswith("scibert"):
            self.tokenizer = AutoTokenizer.from_pretrained(cache_scibert(self.model_name))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def get_sim(self, ref_sent: List[str], cand_sents: List[List[str]]):
        ref_token_ids = sent_encode(self.tokenizer, ref_sent)
        cands_ids = [sent_encode(self.tokenizer, cand_sent) for cand_sent in cand_sents]
        r_tokens = [self.tokenizer.decode([i]) for i in ref_token_ids]
        c_tokens_list = []
        """ get tokens"""
        for cand in cands_ids:
            c_tokens = [self.tokenizer.decode([i]) for i in cand]
            c_tokens_list.append(c_tokens)
        """ generate similarity matrix"""
        ref_embedding = bert_encode(self.bert_model, torch.tensor([ref_token_ids]), attention_mask = torch.ones((1, len(ref_token_ids))))[0]
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        sim_list = []
        cand_embeddings = []
        for cand in cands_ids:
            cand_embedding = bert_encode(self.bert_model, torch.tensor([cand]), attention_mask = torch.ones((1, len(cand))))[0]
            cand_embedding.div_(torch.norm(cand_embedding, dim=-1).unsqueeze(-1))
            sim = torch.mm(cand_embedding, ref_embedding.transpose(1, 0))
            sim_list.append(sim)
            cand_embeddings.append(cand_embedding) 
        return r_tokens, c_tokens_list, sim_list

    def score(self, method, sim_list, *args, **kwargs):
        if method == 'bertscore':
            score_func = self._bertscore
        elif method == 'wmd':
            score_func = self._wmd
        elif method == 'lazyemd':
            score_func = self._lazy_emd
        else:
            raise NotImplementedError("{} is not implemented".format(method))

        return score_func(sim_list, *args, **kwargs)

    def _bertscore(self, sim_list):
        p_score_list = []
        r_score_list = []
        p_w_list = []
        r_w_list = []
        for sim in sim_list:
            word_precision, p_weight = sim.max(dim=1)
            word_recall, r_weight = sim.max(dim=0)
            p_score = word_precision[1:-1].sum() / (len(word_precision) - 2.)
            r_score = word_recall[1:-1].sum() / (len(word_recall) - 2.)
            p_score_list.append(p_score.cpu().item())
            r_score_list.append(r_score.cpu().item())
            p_w_list.append((get_weight_mat(p_weight, len(r_weight))).tolist())
            r_w_list.append((get_weight_mat(r_weight, len(p_weight)).T).tolist())
        return p_score_list, r_score_list, p_w_list, r_w_list
    
    def _wmd(self, sim_list):
        score_list = []
        w_mat_list = []
        for sim in sim_list:
            [cand_len, ref_len] = sim.shape
            cand_w = get_uniform_weight(cand_len)
            ref_w = get_uniform_weight(ref_len)
            cost = (1. - sim).cpu().numpy()
            P = ot.emd(cand_w, ref_w, cost)
            score_list.append(float(np.sum(P * cost)))
            w_mat_list.append(P[1:-1,1:-1].tolist())
        return score_list, w_mat_list
    
    def _lazy_emd(self, sim_list, reg_c, reg_r):
        epsilon = 0.009
        score_list = []
        w_mat_list = []
        for sim in sim_list:
            [cand_len, ref_len] = sim.shape
            cand_w = get_uniform_weight(cand_len)
            ref_w = get_uniform_weight(ref_len)
            cost = (1. - sim).cpu().numpy()
            P = unbalanced.sinkhorn_unbalanced(cand_w, ref_w, cost, epsilon, reg_c, reg_r, method='sinkhorn', verbose=False, numItermax=200000)
            score_list.append(float(np.sum(P * cost)))
            w_mat_list.append(P[1:-1,1:-1].tolist())
        return score_list, w_mat_list

if __name__ == '__main__':
#     model_name = 'roberta-large'
    model_name = 'bert-base-uncased'
    ref_sent = 'A dog runs in the grass.' 
    cand_sents = ['A boy runs in the grass.', 'A dog is running in the grass.']
    scorer = Scorer(model_name)
    r_tokens, c_tokens_list, sim_list = scorer.get_sim(ref_sent, cand_sents). # tokenize sentences and get similarity matrix
    bert_scores = scorer.score('bertscore', sim_list).  # calculate bert score
    print(bert_scores[0])
    print(bert_scores[1])
    print(bert_scores[2])
    print(bert_scores[3])
    p = np.array(bert_scores[0])
    r = np.array(bert_scores[1])
    print("bert-f: ", 2*p*r/(p+r))
    print(scorer.score('wmd', sim_list)[0]) # calucate wmd 
    print(scorer.score('lazyemd', sim_list, reg_c=0.23, reg_r=0.31)[0]) # calculate our proposed lazy_emd
