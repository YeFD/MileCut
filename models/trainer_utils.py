import random, torch, os, logging, math
import numpy as np
from utils.BM25 import BM25Okapi as bm25
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm, trange

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def get_rank_bm25(query_cut, ctxs_cut, k=100):
    model_bm25 = bm25(ctxs_cut)
    scores = np.array(model_bm25.get_scores(query_cut))
    rank_list = scores.argsort().tolist()[::-1]
    rank_score = [scores[i] for i in rank_list]
    return rank_list[:k], rank_score[:k]

def get_is_rel(gt_label, rank_list):
    return [gt_label[i] for i in rank_list]

def get_score(q_vectors, ctx_vectors):
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def get_cosine_score(q_vectors, ctx_vectors):
    score = cos_sim(q_vectors, ctx_vectors)
    return score

def cal_F1_k(ranked_list: list, k: int) -> float:
    count, N_D = sum(ranked_list[:k]), sum(ranked_list)
    p_k = count / k
    r_k = (count / N_D) if N_D != 0 else 0
    return (2 * p_k * r_k / (p_k + r_k)) if p_k + r_k != 0 else 0

def cal_DCG_k(ranked_list: list, k: int, penalty=-1) -> float:
    value = 0
    for i in range(k):
        value += (1 / math.log(i + 2, 2)) if ranked_list[i] else (penalty / math.log(i + 2, 2))
    return value

def cal_NCI_k(ranked_list: list, k: int, alpha=221.1) -> float:
    value = 0
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

def get_score_fix_k(label_lists: list, k: int) -> float:
    F1_score, DCG_score = [], []
    OIE_score = []
    for label_list in label_lists:
        F1_score.append(cal_F1_k(label_list, k))
        DCG_score.append(cal_DCG_k(label_list, k))
        OIE_score.append(cal_OIE_k(label_list, k))
    F1, DCG = np.mean(F1_score), np.mean(DCG_score)
    OIE = np.mean(OIE_score)
    return F1, DCG, OIE

def get_score_greedy_k(label_lists_train: list, label_lists_test: list):
    F1_k, DCG_k = [], []
    OIE_k = []
    len_rt = len(label_lists_train[0])
    for label_list in label_lists_train:
        per_k_F1, per_k_DCG = [0], [0]
        per_k_OIE = [0]
        for i in range(1, len_rt+1):
            per_k_F1.append(cal_F1_k(label_list, i))
            per_k_DCG.append(cal_DCG_k(label_list, i))
            per_k_OIE.append(cal_OIE_k(label_list, i))
        F1_k.append(per_k_F1)
        DCG_k.append(per_k_DCG)
        OIE_k.append(per_k_OIE)
    F1_k_mean, DCG_k_mean = np.mean(np.array(F1_k), axis=0), np.mean(np.array(DCG_k), axis=0)
    OIE_k_mean = np.mean(np.array(OIE_k), axis=0)
    F1_greedy, DCG_greedy = np.argmax(F1_k_mean), np.argmax(DCG_k_mean)
    OIE_greedy = np.argmax(OIE_k_mean)
    test_F1 = [cal_F1_k(label_list, F1_greedy) for label_list in label_lists_test]
    test_DCG = [cal_DCG_k(label_list, DCG_greedy) for label_list in label_lists_test]
    test_OIE = [cal_OIE_k(label_list, OIE_greedy) for label_list in label_lists_test]
    return sum(test_F1) / len(test_F1), sum(test_DCG) / len(test_DCG), sum(test_OIE) / len(test_OIE)

def get_score_oracle(label_lists: list):
    F1_k, DCG_k = [], []
    OIE_k = []
    len_rt = len(label_lists[0])
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

def get_cut_position_oracle(label_lists):
    F1 = []
    for label_list in label_lists:
        cur_F1 = []
        for i in range(1, len(label_lists[0])+1):
            cur_F1.append(cal_F1_k(label_list, i))
        F1.append(cur_F1)
    cut_position = np.argmax(np.array(F1), axis=1)
    return cut_position

def get_precedent_sim(q_emb, c_emb, rank_list):
    cosine_score = get_cosine_score(q_emb, c_emb)[0].cpu()
    precedent_sim = [1]
    for i in range(1, len(rank_list)):
        precedent_score = np.array([cosine_score[idx] for idx in rank_list[:i]])
        precedent_weight = precedent_score / precedent_score.sum()
        w = torch.Tensor(precedent_weight, device='cpu').reshape(1, precedent_score.shape[0])
        precedent_embs = [c_emb[idx].reshape(1, 768).cpu() for idx in rank_list[:i]]
        precedent_emb = torch.mm(w, torch.cat(precedent_embs))
        cur_emb = c_emb[rank_list[i]].cpu()
        score = get_cosine_score(cur_emb, precedent_emb)[0][0]
        precedent_sim.append(score)
    return np.array(precedent_sim)

def init_dual_model(init_bert_path):
    max_seq_length = 512
    word_embedding_model = models.Transformer(init_bert_path, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    dual_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return dual_model

def rank_bm25(dataset_train, dataset_test):
    rank_lists_train = []
    score_lists_train = []
    label_lists_train = []
    for i in tqdm(range(len(dataset_train))):
        query = dataset_train[i]["query"]
        query_cut = query["q_cut"]
        ctxs = dataset_train[i]["ctxs"]
        gt_label = [ctx['label'] for ctx in ctxs]
        ctxs_cut = [ctx['ajjbqk_cut'] for ctx in ctxs]
        rank_list_train, rank_score_train = get_rank_bm25(query_cut, ctxs_cut)
        rank_lists_train.append(rank_list_train)
        score_lists_train.append(rank_score_train)
        label_lists_train.append(get_is_rel(gt_label, rank_list_train))
    rank_lists_test = []
    score_lists_test = []
    label_lists_test = []
    for i in tqdm(range(len(dataset_test))):
        query = dataset_test[i]["query"]
        query_cut = query["q_cut"]
        ctxs = dataset_test[i]["ctxs"]
        gt_label = [ctx['label'] for ctx in ctxs]
        ctxs_cut = [ctx['ajjbqk_cut'] for ctx in ctxs]
        rank_list_test, rank_score_test = get_rank_bm25(query_cut, ctxs_cut)
        rank_lists_test.append(rank_list_test)
        score_lists_test.append(rank_score_test)
        label_lists_test.append(get_is_rel(gt_label, rank_list_test))
    return rank_lists_train, score_lists_train, label_lists_train, rank_lists_test, score_lists_test, label_lists_test

def get_oracle_pos(label_lists: list, len_rt):
    F1_k, DCG_k = [], []
    OIE_k = []
    for label_list in label_lists:
        per_k_F1, per_k_DCG = [0], [0]
        per_k_OIE = [0]
        for i in range(1, len_rt+1):
            per_k_F1.append(cal_F1_k(label_list, i))
            per_k_DCG.append(cal_DCG_k(label_list, i))
            per_k_OIE.append(cal_OIE_k(label_list, i))
        F1_k.append(per_k_F1)
        DCG_k.append(per_k_DCG)
        OIE_k.append(per_k_OIE)
    F1_best_pos, DCG_best_pos = np.argmax(np.array(F1_k), axis=1), np.argmax(np.array(DCG_k), axis=1)
    OIE_best_pos = np.argmax(np.array(OIE_k), axis=1)
    return F1_best_pos, DCG_best_pos, OIE_best_pos