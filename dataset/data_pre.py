import re
import json
with open(r'lecard/lecard.json', 'r', encoding='utf-8') as f:
    lecard_dataset = json.load(f)
    f.close()
numdict = {1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",0:"零"} #个位数的字典
digitdict = {1:"十",2:"百",3:"千",4:"万"} #位称的字典
 

def maxdigit(number,count):
    num = number//10 #整除是//
    if num != 0:
        return maxdigit(num,count+1) #加上return才能进行递归
    else:
        digit_num = number%10 #digit_num是最高位上的数字
        return count,digit_num #count记录最高位
 
def No2Cn(number):
    max_digit,digit_num = maxdigit(number,0)
 
    temp = number
    num_list = [] #储存各位数字（最高位的数字也可以通过num_list[-1]得到
    while temp > 0:
        position = temp%10
        temp //= 10 #整除是//
        num_list.append(position)
 
    chinese = ""
    if max_digit == 0: #个位数
        chinese = numdict[number]
    elif max_digit == 1: #十位数
        if digit_num == 1: #若十位上是1，则称为“十几”，而一般不称为“一十几”（与超过2位的数分开讨论的原因）
            chinese = "十"+numdict[num_list[0]]
        else:
            chinese = numdict[num_list[-1]]+"十"+numdict[num_list[0]]
    elif max_digit > 1: #超过2位的数
        while max_digit > 0:
            if num_list[-1] != 0: #若当前位上数字不为0，则加上位称
                chinese += numdict[num_list[-1]]+digitdict[max_digit]
                max_digit -= 1
                num_list.pop(-1)
            else: #若当前位上数字为0，则不加上位称
                chinese += numdict[num_list[-1]]
                max_digit -= 1
                num_list.pop(-1)
        chinese += numdict[num_list[-1]]
        
    if chinese.count("零") > 1: #中文数字中最多只有1个零
        count_0 = chinese.count("零")
        chinese = chinese.replace("零","",count_0-1)
    if chinese.endswith("零"): #个位数如果为0，不读出
        chinese = chinese[:-1]
    return chinese
# No2Cn(3)

import json, re
with open(r'../utils/xingfa.txt', 'r', encoding='utf-8') as f:
    # dataset_json = json.dumps(dataset)
    xs_file = f.read()
    f.close()
xs_terms = [t if t[0]=='第' else '第'+t for t in xs_file.split('\n第')]
terms_xs_dict = {}
termx_xs_list = []
terms_xs_dict2 = {}
cur_tiao_num = 0
cur_bian_num = 0
cur_bian_str = ''
cur_zhang_num = 0
cur_zhang_str = ''

cur_zhang_num2 = 0
cur_tiao_num2 = 0
for t in xs_terms:
    t = t.replace(' ', '')
    next_bian_num_str = '第'+No2Cn(cur_bian_num+1)+'编'
    if next_bian_num_str == t[:len(next_bian_num_str)]:
        cur_bian_str = t
        cur_bian_num = cur_bian_num + 1
        cur_tiao_num2 = 0
        cur_zhang_num = 0
        continue
    
    next_zhang_num_str = '第'+No2Cn(cur_zhang_num+1)+'章'
    if next_zhang_num_str == t[:len(next_zhang_num_str)]:
        cur_zhang_str = t
        cur_zhang_num = cur_zhang_num + 1
        cur_tiao_num2 = 0
        continue

    cur_zhang_num_str = '第'+No2Cn(cur_tiao_num+1)+'条'
    if cur_zhang_num_str == t[:len(cur_zhang_num_str)]:
        if cur_bian_str == '' or cur_zhang_str == '':
            print('KeyError')
        if cur_bian_str not in terms_xs_dict.keys():
            terms_xs_dict[cur_bian_str] = {cur_zhang_str: {cur_zhang_num_str: t[len(cur_zhang_num_str):]}}
        elif cur_zhang_str not in terms_xs_dict[cur_bian_str]:
            terms_xs_dict[cur_bian_str][cur_zhang_str] = {cur_zhang_num_str: t[len(cur_zhang_num_str):]}
        else:
            terms_xs_dict[cur_bian_str][cur_zhang_str][cur_zhang_num_str] = t[len(cur_zhang_num_str):]

        termx_xs_list.append((cur_zhang_num_str, (cur_bian_num-1, cur_zhang_num-1, cur_tiao_num2, cur_tiao_num))) # , t[len(cur_zhang_num_str):]
        terms_xs_dict2[cur_zhang_num_str] = {'label': [cur_bian_num-1, cur_zhang_num-1, cur_tiao_num2, cur_tiao_num], 'text': t[len(cur_zhang_num_str):]}
        cur_tiao_num = cur_tiao_num + 1
        cur_tiao_num2 = cur_tiao_num2 + 1
        continue
    print(t)

term_set = set()
re_pattern = '刑法》?[第零一二三四五六七八九十百千,条之（）项款的、，《刑法修正案》〉公布实施前年订a-zA-Z0-9。：]{1,}'
for k in lecard_dataset:
    data = lecard_dataset[k]
    ctxs = data['ctxs']
    for ctx in ctxs:
        text = ctx['cpfxgc'].replace(' ', '')
        text2 = ctx['ajjbqk'].replace(' ', '')
        text3 = ctx['pjjg'].replace(' ', '')
        id = ctx['id']
        res = re.findall(re_pattern, text) # 〇0123456789
        res2 = re.findall(re_pattern, text2)
        res3 = re.findall(re_pattern, text3)
        res = res + res2 + res3
        if len(res) == 0:
            continue
        # elif len(res) > 1:
        #     print(res)
        cur_terms = set()
        for r in res:
            r = r.replace('T', '')
            terms = re.findall('第?[零一二三四五六七八九十百千0-9]{1,7}条', r)
            if len(terms) > 0:
                # print(terms)
                terms2 = [t if t[0]=='第' else '第'+t for t in terms]
                terms3 = [t if t[-1]=='条' else t+'条' for t in terms2]
                terms3 = list(filter(lambda t: not t[1:-1].isdigit() or int(t[1:-1])<1000, terms3))
                terms4 = ['第'+No2Cn(int(t[1:-1]))+'条' if t[1:-1].isdigit() else t for t in terms3]
                terms4 = list(filter(lambda t: t in terms_xs_dict2.keys(), terms4))
                cur_terms.update(terms4)
        term_set.update(cur_terms)

from sklearn.preprocessing import LabelEncoder
term_list = list(term_set)
print(len(term_list))
term_label_encoder = LabelEncoder()
term_label_encoder.fit(term_list)

re_pattern = '刑法》?[第零一二三四五六七八九十百千,条之（）项款的、，《刑法修正案》〉公布实施前年订a-zA-Z0-9。：]{1,}'
for k in lecard_dataset:
    data = lecard_dataset[k]
    ctxs = data['ctxs']
    for ctx in ctxs:
        text = ctx['cpfxgc'].replace(' ', '')
        text2 = ctx['ajjbqk'].replace(' ', '')
        text3 = ctx['pjjg'].replace(' ', '')
        id = ctx['id']
        res = re.findall(re_pattern, text)
        res2 = re.findall(re_pattern, text2)
        res3 = re.findall(re_pattern, text3)
        res = res + res2 + res3

        cur_terms = set()
        for r in res:
            r = r.replace('T', '')
            terms = re.findall('第?[零一二三四五六七八九十百千0-9]{1,7}条', r)
            if len(terms) > 0:
                terms2 = [t if t[0]=='第' else '第'+t for t in terms]
                terms3 = [t if t[-1]=='条' else t+'条' for t in terms2]
                terms3 = list(filter(lambda t: not t[1:-1].isdigit() or int(t[1:-1])<1000, terms3))
                terms4 = ['第'+No2Cn(int(t[1:-1]))+'条' if t[1:-1].isdigit() else t for t in terms3]
                terms4 = list(filter(lambda t: t in terms_xs_dict2.keys(), terms4))
                cur_terms.update(terms4)
        cur_terms = list(cur_terms)        
        ctx['terms_label'] = [terms_xs_dict2[t] for t in cur_terms]
        ctx['terms'] = cur_terms
        
import json
import numpy as np
with open(r'lecard/lecard_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(lecard_dataset, f, ensure_ascii=False, indent=4)
    f.close()
