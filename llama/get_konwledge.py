# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import random
from typing import List, Optional
from itertools import zip_longest
from collections import OrderedDict
import fire
from sklearn import metrics
from tqdm import tqdm
from llama import Llama, Dialog
import json
import re
import numpy as np
# pip install fuzzywuzzy
# from fuzzywuzzy import fuzz
# --ckpt_dir
# llama - 2 - 7
# b - chat / --tokenizer_path
# tokenizer.model - -max_seq_len
# 512 - -max_batch_size
# 6
# ckpt_dir = 'llama-2-7b-chat/',
# tokenizer_path = 'tokenizer.model',
# def fuzzy_match(sentence, labels):
#     max_score = 0
#     matched_label = None
#
#     for label in labels:
#         score = fuzz.partial_ratio(sentence, label)
#         if score > max_score:
#             max_score = score
#             matched_label = label
#
#     return matched_label
#
# # 示例
# sentence1 = "Based on the given sentence, the most relevant aspect category is:\n* atmosphere (salon)'}"
# labels = ["staff_master", "food_mealtype_lunch", "restaurant_location", "food_food_dessert", "salon_atmosphere"]
#
# result = fuzzy_match(sentence1, labels)
# print("Matched Label:", result)

def generate_(
    sentence_category,
    generator,
    max_batch_size,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None):

    template = "Give a sentence-category pair, complete the following reasoning task. Directly return the answer without other text. "
    input_list = []
    content_values = []
    for (sentence, key, category) in sentence_category:
        input_list.append([
                {"role": "system", "content": "Use two extremely short sentences to reply."
                                              " Use the output template of 'The reason is...."},
                { "role": "user",
                        "content": template + '\n'
                               "Sentence: '{sentence}'. Category: '{category}'."
                                # " Explain the meaning of '{category}"
                                "Infer the reason why the sentence belongs to '{category}' ".format(category=key, sentence=sentence),
                }
            ])

    # dialogs: List[Dialog] = input_list
    # try:

    dialogs: List[Dialog] = input_list
    data = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for ((sentence, key, category), reason) in zip(sentence_category, data):
        content_values.append((sentence, category, reason['generation']['content']))

    # content_value = data[0]['generation']['content']

    # except Exception as e:
    #     print(e)
    #     print('e')

    return content_values


def make_label(classes, aspect):
    # classes: N class names
    # aspects: class names (<=N) contained in current instance
    label = [0] * len(classes)
    # a label of length N
    for i in range(len(classes)):
        if classes[i] in aspect:
            label[i] = 1
    return label

def main(
    ckpt_dir: str = './llama/llama-2-7b-chat/',
    tokenizer_path: str = './llama/tokenizer.model',
    max_seq_len: int = 1024,
    max_batch_size: int = 200):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    N = 5
    result_dict = {}
    path = './data/FewAsp(Random)/train.json'
    path_res = './data/FewAsp(Random)/train_llm.json'
    f1 = open(path, 'r')
    test_data = json.load(f1)
    f1.close()
    N = 5
    keys = test_data.keys()

    # print(keys)
    # 生成随机的类别
    # target_classes = random.sample(test_data.keys(), N)
    for key in keys:
        category_res = []
        category_sentences = test_data[key]

        sentence_category = []
        for id in tqdm(range(len(category_sentences))):
            sen = category_sentences[id][0]
            # print(sen)
            sentnce = ' '.join(sen)
            category = category_sentences[id][1]  # 多标签的问题
            sentence_category.append((sentnce, key, category))
            if len(sentence_category) == 20 or id == len(category_sentences)-1:
                # print(len(sentence_category), sentence_category)
                # print('*' * 18)
                reasons = generate_(sentence_category, generator, max_batch_size)
                for (sentence, category, reason) in reasons:
                    elems = []
                    res_list = reason.split()
                    elems.append(sentence)
                    elems.append(category)
                    elems.append(res_list)
                    category_res.append(elems)
                # print(category_res)
                # print('*' * 18)
                sentence_category = []


        # result_dict[key] = category_res
        #
        # for id in tqdm(range(len(category_sentences))):
        #     elems = []
        #     sen = category_sentences[id][0]
        #     sentnce = ' '.join(sen)
        #     category = category_sentences[id][1]  # 多标签的问题
        #     reason = generate_(sentnce, category, generator, max_batch_size)
        #     res_list = reason.split()
        #     elems.append(sen)
        #     elems.append(category)
        #     elems.append(res_list)
        #     category_res.append(elems)

        result_dict[key] = category_res
    # print(result_dict)
    # print("Test auc", AUC_eval)
    # print("Test f1", macro_f1_eval)
    # # 保存回JSON文件
    with open(path_res, 'w') as file:
        json.dump(result_dict, file, indent=2)
        file.close()

if __name__ == "__main__":
    fire.Fire(main)