import numpy as np
import math, json

def cal_F1_k(ranked_list: list, k: int) -> float:
    """
    计算F1 score
    k: 截断到第k个，从1计数
    """
    count, N_D = sum(ranked_list[:k]), sum(ranked_list)  # 统计list中label为1数量
    p_k = count / k
    r_k = (count / N_D) if N_D != 0 else 0
    return (2 * p_k * r_k / (p_k + r_k)) if p_k + r_k != 0 else 0
def cal_DCG_k(ranked_list: list, k: int, penalty=-1) -> float:
    """
    计算DCG
    """
    value = 0
    for i in range(k):
        value += (1 / math.log(i + 2, 2)) if ranked_list[i] else (penalty / math.log(i + 2, 2))  # i从0开始
    return value
class RerankingEvaluator():
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    """
    def __init__(self,
                 name: str = 'dataset',
                 corpus_chunk_size: int = 100,
                 show_progress_bar: bool = True,
                 mrr_at_k = [10, 20 ,30 ,100],
                 ndcg_at_k = [10, 20 ,30 ,100],
                 accuracy_at_k = [1, 3, 5, 10, 50, 100],
                 precision_recall_at_k = [1, 3, 5, 10, 50, 100],
                 map_at_k = [10, 100],
                 oracle_at_k = [100],
                 ):
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.show_progress_bar = show_progress_bar
        self.corpus_chunk_size = corpus_chunk_size

        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.oracle_at_k = oracle_at_k

    def compute_metrices(self, rank_lists, gt_idx):
        print("LegalRerankingEvaluator: Evaluating result on " + self.name + "")
        #Compute scores
        scores = self.compute_metrics(rank_lists, gt_idx)

        #Output
        print("Dataset: {}".format(self.name))
        self.output_scores(scores)

        return scores

    def output_scores(self, scores):
        for k in scores['accuracy@k']:
            print("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

        for k in scores['precision@k']:
            print("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

        for k in scores['recall@k']:
            print("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))

        for k in scores['mrr@k']:
            print("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

        for k in scores['ndcg@k']:
            print("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))

        for k in scores['map@k']:
            print("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))

        for k in scores['oracle@f1']:
            print("ORACLE@F1: {:.4f}".format(scores['oracle@f1'][k]))

        for k in scores['oracle@dcg']:
            print("ORACLE@DCG: {:.4f}".format(scores['oracle@dcg'][k]))

    def compute_metrics(self, rank_lists, gt_idx):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}
        Oracle_F1 = {k: [] for k in self.oracle_at_k}
        Oracle_DCG = {k: [] for k in self.oracle_at_k}

        # Compute scores on results
        for query_key in range(len(rank_lists)):
            # Get rank
            cur_rank = rank_lists[query_key]
            top_hits = cur_rank
            query_relevant_docs = gt_idx[query_key]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)
            # Oracle@k
            for k_val in self.oracle_at_k:
                rel = [1 if hit in query_relevant_docs else 0 for hit in top_hits[0:k_val]]
                per_k_F1, per_k_DCG = [0], [0]
                for i in range(1, self.corpus_chunk_size + 1):  # 100
                    per_k_F1.append(cal_F1_k(rel, i))
                    per_k_DCG.append(cal_DCG_k(rel, i))
                # F1_best, DCG_best = np.max(np.array(F1_k), axis=1), np.max(np.array(DCG_k), axis=1)
                Oracle_F1[k_val].append(per_k_F1)
                Oracle_DCG[k_val].append(per_k_DCG)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(rank_lists)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(rank_lists)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        for k in Oracle_F1:
            F1_best = np.max(np.array(Oracle_F1[k]), axis=1)
            Oracle_F1[k] = np.mean(F1_best)
        for k in Oracle_DCG:
            DCG_best = np.max(np.array(Oracle_DCG[k]), axis=1)
            Oracle_DCG[k] = np.mean(DCG_best)

        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k, 'oracle@f1': Oracle_F1, 'oracle@dcg': Oracle_DCG}

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
