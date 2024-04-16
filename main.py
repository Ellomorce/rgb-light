#%%
import os
import sys
import time
import logging
import json
import random
import math
import tqdm
import pandas as pd
from models import FFMLLama2, AzureGPT35, BreeXe8x7b, Breeze7b
#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler('runlog.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
#%%
#Global
save_result_dir = r'Your Result Directory Path'
test_noise = r'Your Data Path'
zh_noise = r'Your Data Path'
zh_int = r'Your Data Path'
zh_fact = r'Your Data Path'
skfh_noise = r'Your Data Path'
skfh_int = r'Your Data Path'
skfh_fact = r'Your Data Path'
#%%
def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        maxnum = max([len(i) for i in instance['positive']])
        for i in range(1,maxnum):
            for j in instance['positive']:
                if len(j) > i:
                    docs.append(j[i])
                    if len(docs) == pos_num:
                        break
            if len(docs) == pos_num:
                break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs,min(len(indexs),pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain,min(len(remain),correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative

    random.shuffle(docs)
    # query=str, ans=list, docs=list
    return query, ans, docs

def run_eval(data_path, model_name):

    if model_name == 'ffmllama':
        model = FFMLLama2()
    elif model_name == 'gpt35':
        model = AzureGPT35()
    elif model_name == 'bree8x7b':
        model = BreeXe8x7b
    elif model_name == 'bree7b':
        model = Breeze7b

    match data_path:
        case data_path if 'test.json' in data_path:
            test_category = 'Noise'
            dataset = 'Test'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案，你將生成'因文件中信息不足，無法回答問題。'。如果某些文件中的信息與事實不一致，請生成'提供的文件中有錯誤事實。'，並提供正確的答案。"
        case data_path if 'zh.json' in data_path:
            test_category = 'Noise'
            dataset = 'RGB_ZH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案，你將生成'因文件中信息不足，無法回答問題。'。如果某些文件中的信息與事實不一致，請生成'提供的文件中有錯誤事實。'，並提供正確的答案。"
        case data_path if 'zh_int.json' in data_path:
            test_category = 'Integration'
            dataset = 'RGB_ZH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，問題會詢問兩件事，你需要檢索所有文件並回答兩個答案。外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案或僅有一個答案，你將生成'因文件中信息不足，無法回答問題。'"
        case data_path if 'zh_fact.json' in data_path:
            test_category = 'Factual_error'
            dataset = 'RGB_ZH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案，你將生成'因文件中信息不足，無法回答問題。'。如果某些文件中的信息與事實不一致，請生成'提供的文件中有錯誤事實。'，並提供正確的答案。"
        case data_path if 'skfh.json' in data_path:
            test_category = 'Noise'
            dataset = 'SKFH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案，你將生成'因文件中信息不足，無法回答問題。'。如果某些文件中的信息與事實不一致，請生成'提供的文件中有錯誤事實。'，並提供正確的答案。"
        case data_path if 'skfh_int.json' in data_path:
            test_category = 'Integration'
            dataset = 'SKFH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，問題會詢問兩件事，你需要檢索所有文件並回答兩個答案。外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案或僅有一個答案，你將生成'因文件中信息不足，無法回答問題。'"
        case data_path if 'skfh_fact.json' in data_path:
            test_category = 'Factual_error'
            dataset = 'SKFH'
            system = "你是一個精確可靠的AI助手，可以在外部文件的幫助下回答問題。請注意，外部文件提供的信息可能包含雜訊或錯誤事實。如果文件中的資訊包含正確答案，你將給出準確的答案。如果文件中的信息不包含答案，你將生成'因文件中信息不足，無法回答問題。'。如果某些文件中的信息與事實不一致，請生成'提供的文件中有錯誤事實。'，並提供正確的答案。"

    #
    instances = []
    qid_list = [] # list
    ans_list = [] # if int>list of lists, other>list
    doc_list = []
    model_ans_list = []
    #
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            qid_list.append(json.loads(line)['id'])
            ans_list.append(json.loads(line)['answer'])
            instances.append(json.loads(line))

    for instance in tqdm.tqdm(instances):
        random.seed(2333)
        query, ans, docs = processdata(
            instance,
            noise_rate=0.4,
            passage_num=5,
            filename=data_path,
            correct_rate= 0)
        text = f'''"Document":{{"DOCS":{docs}}}, "Question":{{"QUERY":{query}}}'''
        doc_list.append(docs)
        model_ans = model.conversation(text=text, system=system)
        logger.info('-'*50)
        logger.info(f'{model_ans}')
        model_ans_list.append(model_ans)
        time.sleep(3)

    val_df = pd.DataFrame(data=zip(qid_list, doc_list, ans_list, model_ans_list), columns=['QID', 'Docs', 'Real_Answers', 'Model_Answers'])
    val_df.to_csv(
        os.path.join(save_result_dir, f"{dataset}_{test_category}_val.csv"),
        encoding='utf-8-sig',
        index=False)
    return val_df
#%%
def main() -> None:
    data_path = test_noise
    model = 'bree7b'
    run_eval(data_path, model_name=model)

if __name__ == "__main__":
    main()
#%%
