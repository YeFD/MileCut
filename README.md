# MileCut: A Multi-view Truncation Framework for Legal Case Retrieval
The official Github repository for paper [MileCut]() (WWW'24).

## Overview
[MileCut](#milecut)
[C3RD](#c3rd)

## MileCut
### Abstract
In the search process, it is essential to strike a balance between effectiveness and efficiency to improve search experience. Thus, ranking list truncation has become increasingly crucial. Especially in the legal domain, irrelevant cases can severely increase search costs and even compromise the pursuit of legal justice. However, there are truncation challenges that mainly arise from the distinctive structure of legal case documents, where the elements such as fact, reasoning, and judgement in a case serve as different but multi-view texts, which could result in a bad performance if the multi-view texts cannot be well-modeled. Existing approaches are limited due to their inability to handle multi-view elements information and their neglect of semantic interconnections between cases in the ranking list. In this paper, we propose a multi-view truncation framework for legal case retrieval, named MileCut. MileCut employs a case elements extraction module to fully exploit the multi-view information of cases in the ranking list. Then, MileCut applies a multi-view truncation module to select the most informative view and make a more comprehensive cut-off decision, similar to how legal experts look over retrieval results. As a practical evaluation, MileCut is assessed across three datasets, including criminal and civil case retrieval scenarios, and the results show that MileCut outperforms other methods on F1, DCG, and OIE metrics.


### Getting start
This guide will walk you through processing a dataset and training a model in `run.ipynb`.

#### Load a retrieval dataset
```python
from models.LegalTrainer import *
trainer = LegalTrainer()
trainer.prepare_civil_dataset_mini(path='dataset/C3RD_mini')
```
Here, we load C3RD_mini for a quick start. You can download complete C3RD from [here](#download). For access to LeCaRD and COLIEE datasets, please visit their official websites.

#### Get ranking results
```python
from models.trainer_utils import init_dual_model
model = init_dual_model(r'bert_ms') # bert-base-chinese
rank_emb_train = rank_emb_pre(trainer.dataset_train, model)
rank_emb_test = rank_emb_pre(trainer.dataset_test, model)
```
This step involves obtaining semantic representations for ranking throught a pre-train model.

#### Generate Oracle results
```python
get_stat_score(trainer.dataset_train, trainer.dataset_test, rank_emb_train, rank_emb_test)
```

#### Prepare the truncation dataset
```python
from dataloader.milecut_dataloader import dataloader as mile_dataloader
train_loader, test_loader, _ = mile_dataloader(trainer.dataset_train, trainer.dataset_test, rank_emb_train, rank_emb_test, batch_size=1, input_size=6)
trainer.train_loader, trainer.test_loader = train_loader, test_loader
```

#### Train a model
```python
trainer.run(reset_params=True, loss_name='MileCut', model_name='MileCut', epoch=30, coefficient=0.6, input_size=6, set_train_seed=True, train_seed=42,view_input_size=3, label_input_size=9)
```
You have the flexibility to select a model, loss function, and other configurations by adjusting the parameters.


## C3RD
**C**hinese **C**ivil **C**ase **R**etrieval **D**ataset(C3RD) comprises 1146 queries, each with 100 candidate civil cases documents.The statistic of C3RD is shown as follow:

| Datasets        | C3RD    |
|-----------------|---------|
| Language        | Chinese |
| Documents       | 114,600 |
| Queries         | 1,146   |
| Candidates/Query| 100     |
| Rel Case/Query  | 11.43   |

### Background
Civil case retrieval presents unique challenges and opportunities. First, civil cases outnumber criminal cases, indicating a significant demand for their retrieval in real-world applications. Secondly, civil cases are inherently more complex to retrieve civil cases. Unlike the clearer facts in criminal cases, civil cases often have muddled facts, as both parties emphasize favorable aspects. Last but not least, there are no publicly available datasets for civil case retrieval. Hence, our proposed **C**hinese **C**ivil **C**ase **R**etrieval **D**ataset(C3RD) aims to fill this void. C3RD comprises 1146 queries, and each query has 100 candidate civil case documents. 

<details>
<summary>The details of construction</summary>

To construct the C3RD dataset, we collect over 23 million civil case documents from [China Judgements Online website](https://wenshu.court.gov.cn), a resource published by the Supreme People's Court of China. 

For a case document, fact section is typically presented by plaintiffs and defendants. Reasoning section is summarized by judges and involves the extraction of key elements. Judgement section is the final decision made by the court. These elements are distinctly partitioned in the documents and exhibit clear features. These divisions are pre-defined with regular expression matching during data collection. For example, reasoning section is split by specific markers 'The court believes that ...' and judgement section is split by 'The judgement is as follows: ...'. For label of referenced law articles, the extraction processing can be found in our source code.

Next, we proceeded to refine the corpus by applying a filtering process. The intent was to exclude cases that might be considered too brief or excessively lengthy. In addition, we discarded cases that had been withdrawn in order to focus on cases that had proceeded to full legal resolution. 

After pre-processing, 8 million civil case documents are left. For retrieval purposes, we developed a criterion to identify relevant cases. According to a guidance document about relevant case retrieval published by the Supreme People's Court of China, a relevant case is defined as a case that shares similarities with a query case in aspects such as facts, cause reason and application of law articles. Based on this guidance, we design heuristic rules to filter cases related to the query. Specifically, we deem cases with the same legal cause reason and references to specific law articles as relevant cases. 

We then randomly select the fact section of a case to serve as a query and remove that case from the pool of relevant candidates. Lastly, we adopted BM25 to search for negative candidates to complete the candidate pools. In this process, we apply several filtering methods to ensure the identified cases aren't related to the query. These measures are put in place to maximize the likelihood that the selected cases differ considerably from the query. 

Finally, C3RD comprises 1146 queries, and each query has 100 candidate civil case documents. 

</details>

### Download
You can download C3RD through google drive([link](https://drive.google.com/file/d/1LwaEWc7iYEu6qDFKu5wworGOkKFouLm3)).

### Data structure
The dataset consists of several [query, candidate cases, ground-truth labels].
#### query
A query is the fact section for a civil case and describes the basic fact, evidence of the case. For each query, there are 3 to 30 similar cases in the candidate cases. The full query case is also given. An example of a query is: 
```
"query": "原告彭正坤诉称，2018年1月29日，原告彭正坤与三名被告及本案第三人签订了《股权转让协议书》及《股权转让补充协议》。约定被告冉启斌和马维驰将其持有的重庆可可希生态果业有限公司共计75%的股权转让给彭正坤，转让款共计375万元。协议签订后，2019年2月9日，原告向被告支付了股权转让款并办理了工商变更登记后，在合同履行过程中，原告发现重庆可可希生态果业有限公司存在重大问题，公司无法继续经营，原告要求被告解除股权转让合同并退还股权转让款。因此，彭正坤来院起诉，请求判令……"
```

#### candidates
The candidate cases contains 100 candidate civil cases. The name meanings for each candidate case dict are as follows:
* Case: Name of the case
* Category: Case category, including first-level category cat_1 and second-level category cat_2
* JudgeAccusation: Case fact section that describes the basic fact of the case
* JudgeReason: Analysis of case process by judge
* JudgeResult: Result of a court decision
* Keywords: Keywords of case
* Parties: Entity information
* LegalBasis: Referenced law articles
* Other non-important name: CaseId, CaseProc, CaseRecord, CaseType.

 An example of a candidate case is: 
```
"Case": "京铜牛集团有限公司与王金海供用热力合同纠纷一审民事判决书",
"CaseId": "b139b5166ba74fd98bcf505f5845d7eb",
"CaseProc": "民事一审",
"CaseRecord": "原告北京铜牛集团有限公司（以下简称原告）与被告王金海（以下简称被告）供用热力合同纠纷一案，本院受理后，依法组成合议庭，公开开庭进行了审理。原告的委托代理人高邵军、栗文山，被告均到庭参加了诉讼。本案现已审理终结",
"CaseType": "民事案件",
"Category": [{"cat_1": "合同事务", "cat_2": "合同纠纷", "case_cause": "供用热力合同纠纷"}],
"JudgeAccusation": "原告诉称：被告居住在北京市朝阳区甘露园南里ＸＸＸ号，原告为被告提供供暖服务工作。在供暖服务期间内，原告的供暖工作完全符合京政容发（2010）126号《北京市市政市容管理委员会关于印发贯彻执行住宅采暖室内空气温度测量方法若干规定的通知》的规定，达到了供暖标准。……",
"JudgeReason": "本院认为：原告已为包括被告在内的小区业主提供了供暖服务，双方形成事实上供暖服务合同关系。民事活动应当遵循公平和等价有偿的原则，被告享受了原告提供的供暖服务，理应交纳供暖费。……综上，依照《中华人民共和国民法通则》第一百零六条之规定，判决如下",
"JudgeResult": "一、被告王金海于本判决生效后七日内给付原告北京铜牛集团有限公司二〇一三年至二〇一四年的供暖费一千六百一十元一角。\n二、驳回原告北京铜牛集团有限公司的其他诉讼请求。……如不服本判决，可在判决书送达之日起十五日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于北京市第三中级人民法院",
"LegalBasis": [
    {"terms": "第一百零六条第一款", "law": "《中华人民共和国民法通则（2009修正）》"}, 
    {"terms": "第一百零六条第二款", "law": "《中华人民共和国民法通则（2009修正）》"}, 
    {"terms": "第一百零六条第三款", "law": "《中华人民共和国民法通则（2009修正）》"}
    ],
"Parties": [{"NameText": "王金海", "Name": "王金海", "LegalEntity": "Person", "Prop": "被告"}]
```

#### labels

For a query, number i in gt_idx indicates that the i-th candidate case is a relevant case. A sample of a gt_idx is

```
"gt_idx": [8, 31, 37, 54, 58]
```

### Baseline
For evaluation, we implement several existing retrieval models on C3RD as baselines. This will provide a comprehensive view of the characteristics and its applicability for different retrieval methods. The results are shown in Table.


| Metrics                        | P@5    | P@10   | MAP@10 | NDCG@10 | NDCG@20 | NDCG@30 | MRR    |
| ------------------------------ | ------ | ------ | ------ | ------- | ------- | ------- | ------ |
| BM25                           | 0.5079 | 0.4146 | 0.4835 | 0.5642  | 0.5810  | 0.5993  | 0.6929 |
| BERT-Civil(dual, w/o training) | 0.5670 | 0.4773 | 0.7419 | 0.6235  | 0.6528  | 0.6841  | 0.7455 |
| BERT-Chinese(dual)             | 0.7108 | 0.6231 | 0.6999 | 0.7786  | 0.8208  | 0.8451  | 0.8149 |
| BERT-Civil(dual)               | 0.7732 | 0.6736 | 0.7905 | 0.8546  | 0.8813  | 0.8978  | 0.8788 |
| BERT-Civil(cross)              | 0.7609 | 0.6355 | 0.7682 | 0.8406  | 0.8669  | 0.8863  | 0.9137 |


In the table, `dual` and `cross` denote dual encoder and cross encoder, respectively.


## Ackownledgement
* [MtCut](https://github.com/Woody5962/Ranked-List-Truncation.git) This repository implement most of baselines.
