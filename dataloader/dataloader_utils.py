
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math
from sentence_transformers.util import cos_sim
import torch as t
from torch import Tensor
from utils.BM25 import BM25Okapi as bm25

stopwords = []
with open(r'utils/stopwords.txt', encoding='utf-8') as file:
    for line in file.readlines():
        stopwords.append(line.strip('\n'))

def cal_F1_k(ranked_list: list, k: int) -> float:
    count, N_D = sum(ranked_list[:k]), sum(ranked_list)
    p_k = count / k
    r_k = (count / N_D) if N_D != 0 else 0
    return (2 * p_k * r_k / (p_k + r_k)) if p_k + r_k != 0 else 0

def cal_DCG_k(ranked_list: list, k: int, penalty=-1) -> float:
    value = 0
    for i in range(k):
        value += (1 / math.log(i + 2, 2)) if ranked_list[i] else (penalty / math.log(i + 2, 2))  # i从0开始
    return value

def cal_OIE_k(ranked_list: list, k: int, beta: int=1.05) -> float:
    H_S = 0
    H_G = 0
    H_SG = 0
    D_len = len(ranked_list)
    scale = D_len
    for i in range(k):
        S_d_hat_sum = i + 1
        H_S = H_S - math.log(S_d_hat_sum/D_len, 2)/scale
        G_d_hat_sum = sum(ranked_list[:k]) if ranked_list[i] else D_len
        H_G = H_G - math.log(G_d_hat_sum/D_len, 2)/scale
        SG_hat_sum = sum(ranked_list[:i+1]) if ranked_list[i] else i+1
        H_SG = H_SG - math.log(SG_hat_sum/D_len, 2)/scale
    OIE = H_S + H_G - beta * H_SG
    return OIE

def get_score_oracle(label_lists: list):
    F1_k, DCG_k = [], []
    OIE_k = []
    len_rt = 100
    for label_list in label_lists:
        per_k_F1, per_k_DCG = [0], [0]
        per_k_OIE = [0]
        for i in range(1, len_rt + 1):
            per_k_F1.append(cal_F1_k(label_list, i))
            per_k_DCG.append(cal_DCG_k(label_list, i))
            per_k_OIE.append(cal_OIE_k(label_list, i))
        F1_k.append(per_k_F1)
        DCG_k.append(per_k_DCG)
        OIE_k.append(per_k_OIE)
    F1_best, DCG_best = np.max(np.array(F1_k), axis=1), np.max(np.array(DCG_k), axis=1)
    OIE_best = np.max(np.array(OIE_k), axis=1)
    return np.mean(F1_best), np.mean(DCG_best), np.mean(OIE_best)

def get_stat(rank_list_cut):
    text_len = []
    text_unique = []
    for i in range(len(rank_list_cut)):
        clean_text = rank_list_cut[i]
        text_len.append(len(clean_text))  # 单词个数
        text_unique.append(len(np.unique(clean_text)))  # 不同单词个数
    return text_len, text_unique

def get_bm25_score(query_cut, rank_list_cut): # query and rank text
    model_bm25 = bm25(rank_list_cut)
    scores = np.array(model_bm25.get_scores(query_cut))
    return scores

def get_bm25_sim(query_cut, ctxs_cut_rank):
    bm25_score_lists = []
    for i in range(len(ctxs_cut_rank)):
        bm25_score = get_bm25_score(query_cut, ctxs_cut_rank)
        bm25_score_lists.append(bm25_score)
    return bm25_score_lists

def get_is_rel(gt_label, rank_list):
    return [gt_label[i] for i in rank_list]

def _get_cosine_score(q_vectors, ctx_vectors):
    score = cos_sim(q_vectors, ctx_vectors)
    return score

def _get_rank_list(q_emb, c_emb):
    score = _get_cosine_score(q_emb, c_emb)[0].cpu()
    rank_list = np.array(score).argsort().tolist()[::-1]
    rank_score = [score[i] for i in rank_list]
    return rank_list, rank_score

def _get_score_list(q_emb, c_emb, rank_list):
    score = _get_cosine_score(q_emb, c_emb)[0].cpu()
    rank_score = [score[i] for i in rank_list]
    return rank_score

def get_precedent_sim(q_emb, c_emb, rank_list):
    cosine_score = _get_cosine_score(q_emb, c_emb)[0].cpu()
    precedent_sim = [1]
    for i in range(1, len(rank_list)):
        precedent_score = np.array([cosine_score[idx] for idx in rank_list[:i]])
        precedent_weight = np.exp(precedent_score)/np.exp(precedent_score).sum()
        w = t.Tensor(precedent_weight, device='cpu').reshape(1, precedent_score.shape[0])
        precedent_embs = [c_emb[idx].reshape(1, 768).cpu() for idx in rank_list[:i]]
        precedent_emb = t.mm(w, t.cat(precedent_embs))
        cur_emb = c_emb[rank_list[i]].cpu()
        score = _get_cosine_score(cur_emb, precedent_emb)[0][0]
        precedent_sim.append(score)
    return np.array(precedent_sim)

def get_nb_sim(c_emb, rank_list, cuda=True):
    rank_emb = [c_emb[i] for i in rank_list]
    nb_sim = [cos_sim(rank_emb[0], rank_emb[1]).cpu()]
    for i in range(1, len(rank_list)-1):
        nb_sim.append((cos_sim(rank_emb[i], rank_emb[i-1]).cpu() + cos_sim(rank_emb[i], rank_emb[i+1]).cpu()) / 2)
    nb_sim.append(cos_sim(rank_emb[-2], rank_emb[-1]).cpu())
    return nb_sim