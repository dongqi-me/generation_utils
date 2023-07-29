import spacy
import json

import numpy as np

import multiprocessing as mp



nlp = spacy.load("en_core_web_sm")


def get_dependency_labels(input_data):
    # avg sentence number & word number
    if not input_data:
        raise ValueError("Input data cannot be empty.")



    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]
    
    article_docs = nlp.pipe(articles, n_process=4, disable=['tagger', 'ner', 'lemmatizer', 'textcat', 'custom'])
    reference_docs = nlp.pipe(references, n_process=4, disable=['tagger', 'ner', 'lemmatizer', 'textcat', 'custom'])
    
    article_label_list = []
    reference_label_list = []

    reference_dict  = {}
    article_dict = {}

    all_keys = []
    for doc in article_docs:
        label_dict = {}
        for token in doc:
            if token.dep_ not in label_dict.keys():
                label_dict[token.dep_] = 0               
            else:
                label_dict[token.dep_] += 1
            if token.dep_ not in article_dict.keys():
                article_dict[token.dep_] = 0               
            else:
                article_dict[token.dep_] += 1
            if token.dep_ not in all_keys:
                all_keys.append(token.dep_)
        article_label_list.append(label_dict)
    
    for doc in reference_docs:
        label_dict = {}
        for token in doc:
            if token.dep_ not in label_dict.keys():
                label_dict[token.dep_] = 0               
            else:
                label_dict[token.dep_] += 1
            if token.dep_ not in reference_dict.keys():
                reference_dict[token.dep_] = 0               
            else:
                reference_dict[token.dep_] += 1
            if token.dep_ not in all_keys:
                all_keys.append(token.dep_)
        reference_label_list.append(label_dict)

    for label_dict in article_label_list:
        for label in all_keys:
            if label not in label_dict.keys():
                label_dict[label] = 0
    
    for label_dict in reference_label_list:
        for label in all_keys:
            if label not in label_dict.keys():
                label_dict[label] = 0
    
    for label in all_keys:
        if label not in article_dict.keys():
            article_dict[label] = 0
        if label not in reference_dict.keys():
            reference_dict[label] = 0

    return {"article_label_list": article_label_list, 'reference_label_list': reference_label_list, "reference_dict": reference_dict, "article_dict": article_dict}

if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    train_length_stat = get_dependency_labels(train_data)
    print(train_length_stat["article_dict"], train_length_stat["reference_dict"])
    # {'compound': 23880801, 'nsubjpass': 4291822, 'case': 288771, 'advmod': 7124419, 'auxpass': 4407306, 'ROOT': 9591120, 'det': 19301280, 'amod': 22753747, 'advcl': 2906982, 'prep': 28414458, 'pobj': 28596338, 'cc': 8249064, 'conj': 10369011, 'nummod': 9521064, 'punct': 45414469, 'nsubj': 8947595, 'aux': 3680715, 'npadvmod': 3493313, 'poss': 1126613, 'acomp': 1427192, 'mark': 2081277, 'agent': 921208, 'nmod': 4763338, 'ccomp': 1688084, 'pcomp': 1075473, 'dobj': 9036075, 'appos': 9032411, 'quantmod': 448941, 'relcl': 1538932, 'acl': 2531915, 'parataxis': 258501, 'neg': 527695, 'attr': 1031752, 'xcomp': 1230713, 'dep': 753859, 'prt': 177141, 'expl': 125488, 'oprd': 154426, 'csubj': 120216, 'meta': 278604, 'predet': 67292, 'preconj': 191870, 'csubjpass': 22514, 'dative': 29246, 'intj': 49429} {'punct': 3012651, 'nsubj': 1567324, 'relcl': 302467, 'dobj': 1124253, 'prep': 2730771, 'det': 2260210, 'compound': 1589992, 'pobj': 2630904, 'acl': 241136, 'agent': 71499, 'ROOT': 897302, 'amod': 2066304, 'cc': 733030, 'conj': 806596, 'ccomp': 423270, 'advmod': 972254, 'acomp': 203159, 'mark': 286547, 'advcl': 319872, 'nmod': 115390, 'poss': 267576, 'case': 96606, 'appos': 235410, 'attr': 145830, 'aux': 795830, 'xcomp': 220740, 'prt': 55431, 'nummod': 212044, 'npadvmod': 122704, 'quantmod': 50251, 'nsubjpass': 230395, 'auxpass': 255075, 'neg': 71640, 'csubj': 29194, 'pcomp': 165372, 'predet': 10210, 'preconj': 15082, 'expl': 23667, 'parataxis': 7352, 'oprd': 30107, 'dep': 14573, 'csubjpass': 1304, 'dative': 7917, 'intj': 1197, 'meta': 4272}
    with open("train_dependency_labels.json", 'w') as f:
        json.dump(train_length_stat, f)

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    val_length_stat = get_dependency_labels(val_data)
    print(val_length_stat["article_dict"], val_length_stat["reference_dict"])
    with open("val_dependency_labels.json", 'w') as f:
        json.dump(val_length_stat, f)

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    test_length_stat = get_dependency_labels(test_data)
    print(test_length_stat["article_dict"], test_length_stat["reference_dict"])
    with open("test_dependency_labels.json", 'w') as f:
        json.dump(test_length_stat, f)