import torch as t
import numpy as np
from torch.utils import data
from tqdm import tqdm, trange
from dataloader.dataloader_utils import _get_rank_list, get_is_rel, get_score_oracle, get_bm25_score, get_stat, get_precedent_sim, get_nb_sim, _get_score_list

def get_stat_rank_feature(dataset, rank_lists):
    bm25_score_lists = []
    for i in trange(len(dataset), desc='get_stat_rank_feature'):
        query = dataset[i]['query']
        ctxs = dataset[i]['ctxs']
        rank_list = rank_lists[i]
        ctxs_cut = [ctx['ajjbqk_cut'] for ctx in ctxs]
        ctxs_cut_rank = [ctxs_cut[j] for j in rank_list]
        bm25_score = get_bm25_score(query['q_cut'], ctxs_cut_rank)
        bm25_score_lists.append(bm25_score)
    return bm25_score_lists

def get_stat_len_feature(dataset, rank_lists):
    text_len_lists, text_unique_lists = [], []
    for i in trange(len(dataset)):
        query = dataset[i]['query']
        ctxs = dataset[i]['ctxs']
        rank_list = rank_lists[i]
        ctxs_cut = [ctx['ajjbqk_cut'] for ctx in ctxs]
        ctxs_cut_rank = [ctxs_cut[j] for j in rank_list]
        text_len, text_unique = get_stat(query['q_cut'], ctxs_cut_rank)
        text_len_lists.append(text_len)
        text_unique_lists.append(text_unique)
    return text_len_lists, text_unique_lists

def get_term_label_feature(dataset, rank_lists):
    term_zhang_lists = []
    term_tiao_lists = []
    term_n_lists = []
    for i in trange(len(dataset)):
        ctxs = dataset[i]['ctxs']
        rank_list = rank_lists[i]
        term_zhang_list = []
        term_tiao_list = []
        term_n_list = []
        for j in rank_list:
            cur_terms = [1, -1, -1, -1]
            for t in ctxs[j]['terms_label']:
                if t['label'][0]==1:
                    cur_terms = t['label']
                    continue
            term_zhang_list.append(cur_terms[1])
            term_tiao_list.append(cur_terms[2])
            term_n_list.append(cur_terms[3])
        term_zhang_lists.append(term_zhang_list)
        term_tiao_lists.append(term_tiao_list)
        term_n_lists.append(term_n_list)
    return term_zhang_lists, term_tiao_lists, term_n_lists

def get_rank_feature_multi_view(rank_emb):
    gt_idxs = []
    rank_lists_avg = []
    rank_scores_avg, rank_scores_accusation, rank_scores_reason, rank_scores_result = [], [], [], []
    pre_sim_avg_lists, pre_sim_accusation_lists, pre_sim_reason_lists, pre_sim_result_lists = [], [], [], []
    nb_sim_avg_lists, nb_sim_accusation_lists, nb_sim_reason_lists, nb_sim_result_lists = [], [], [], []
    for i in trange(len(rank_emb), desc='get_rank_feature_multi_view'):
        q_emb, c_emb_accusation, c_emb_reason, c_emb_result, gt_idx = rank_emb[i]
        c_emb_avg = (c_emb_accusation + c_emb_reason + c_emb_result) / 3
        rank_list_avg, rank_score_avg = _get_rank_list(q_emb, c_emb_avg)
        rank_lists_avg.append(rank_list_avg)
        gt_idxs.append(gt_idx)

        rank_scores_avg.append(rank_score_avg)
        rank_scores_accusation.append(_get_score_list(q_emb, c_emb_accusation, rank_list_avg))
        rank_scores_reason.append(_get_score_list(q_emb, c_emb_reason, rank_list_avg))
        rank_scores_result.append(_get_score_list(q_emb, c_emb_result, rank_list_avg))

        pre_sim_avg_lists.append(get_precedent_sim(q_emb, c_emb_avg, rank_list_avg))
        pre_sim_accusation_lists.append(get_precedent_sim(q_emb, c_emb_accusation, rank_list_avg))
        pre_sim_reason_lists.append(get_precedent_sim(q_emb, c_emb_reason, rank_list_avg))
        pre_sim_result_lists.append(get_precedent_sim(q_emb, c_emb_result, rank_list_avg))

        nb_sim_avg_lists.append(get_nb_sim(c_emb_avg, rank_list_avg))
        nb_sim_accusation_lists.append(get_nb_sim(c_emb_accusation, rank_list_avg))
        nb_sim_reason_lists.append(get_nb_sim(c_emb_reason, rank_list_avg))
        nb_sim_result_lists.append(get_nb_sim(c_emb_result, rank_list_avg))
    return rank_lists_avg, rank_scores_avg, rank_scores_accusation, rank_scores_reason, rank_scores_result, pre_sim_avg_lists, pre_sim_accusation_lists, pre_sim_reason_lists, pre_sim_result_lists, nb_sim_avg_lists, nb_sim_accusation_lists, nb_sim_reason_lists, nb_sim_result_lists, gt_idxs


import pickle
class MileCut_Lecard_Dataset(data.Dataset):
    def __init__(self, dataset_train, dataset_test, rank_emb_train, rank_emb_test, input_size=3):
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(dataset_train, dataset_test, rank_emb_train, rank_emb_test, input_size)

    def input_prepare(self, dataset, rank_emb, input_size):
        X_input, y_output = [], []
        rank_lists_avg, rank_scores_avg, rank_scores_accusation, rank_scores_reason, rank_scores_result, pre_sim_avg_lists, pre_sim_accusation_lists, pre_sim_reason_lists, pre_sim_result_lists, nb_sim_avg_lists, nb_sim_accusation_lists, nb_sim_reason_lists, nb_sim_result_lists, gt_idxs = get_rank_feature_multi_view(rank_emb)
        bm25_score_lists = get_stat_rank_feature(dataset, rank_lists_avg)
        text_len_lists, text_unique_lists = get_stat_len_feature(dataset, rank_lists_avg)
        term_zhang_lists, term_tiao_lists, term_n_lists = get_term_label_feature(dataset, rank_lists_avg)

        for i in trange(len(dataset), desc='input_prepare'):
            ctxs = dataset[i]['ctxs']
            gt_label = [ctx['label'] for ctx in ctxs]
            is_rel = get_is_rel(gt_label, rank_lists_avg[i])

            rank_score_avg = rank_scores_avg[i]
            rank_score_accusation = rank_scores_accusation[i]
            rank_score_reason = rank_scores_reason[i]
            rank_score_result = rank_scores_result[i]

            pre_sim_avg = pre_sim_avg_lists[i]
            pre_sim_accusation = pre_sim_accusation_lists[i]
            pre_sim_reason = pre_sim_reason_lists[i]
            pre_sim_result = pre_sim_result_lists[i]

            nb_sim_avg = nb_sim_avg_lists[i]
            nb_sim_accusation = nb_sim_accusation_lists[i]
            nb_sim_reason = nb_sim_reason_lists[i]
            nb_sim_result = nb_sim_result_lists[i]

            input_features = np.column_stack((rank_score_avg, pre_sim_avg, nb_sim_avg, 
                                                bm25_score_lists[i], text_len_lists[i], text_unique_lists[i],
                                                rank_score_accusation, pre_sim_accusation, nb_sim_accusation,
                                                rank_score_reason, pre_sim_reason, nb_sim_reason,
                                                term_zhang_lists[i], term_tiao_lists[i], term_n_lists[i],
                                                rank_score_result, pre_sim_result, nb_sim_result
                                                ))
            X_input.append(input_features.tolist())
            y_output.append(is_rel)
        return X_input, y_output


    def data_prepare(self, dataset_train, dataset_test, rank_emb_train, rank_emb_test, input_size):
        print('prepare dataset_train')
        X_train, y_train = self.input_prepare(dataset_train, rank_emb_train, input_size)
        print('prepare dataset_test')
        print(f'oracle train: {get_score_oracle(y_train)}')
        X_test, y_test = self.input_prepare(dataset_test, rank_emb_test, input_size)
        print(f'oracle test: {get_score_oracle(y_test)}')
        return t.Tensor(X_train), t.Tensor(X_test), t.Tensor(y_train), t.Tensor(y_test)

    def getX_train(self):
        return self.X_train

    def getX_test(self):
        return self.X_test

    def gety_train(self):
        return self.y_train

    def gety_test(self):
        return self.y_test


def dataloader(dataset_train, dataset_test, rank_emb_train, rank_emb_test, batch_size: int=1, input_size: int=3):

    rank_data = MileCut_Lecard_Dataset(dataset_train, dataset_test, rank_emb_train, rank_emb_test, input_size)

    X_train = rank_data.getX_train()
    X_test = rank_data.getX_test()
    y_train = rank_data.gety_train()
    y_test = rank_data.gety_test()

    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, test_loader, rank_data


if __name__ == '__main__':
    pass
