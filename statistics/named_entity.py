import spacy
import json

import numpy as np

import multiprocessing as mp



nlp = spacy.load("en_core_web_sm")


def get_ner_labels(input_data):
    # avg sentence number & word number
    if not input_data:
        raise ValueError("Input data cannot be empty.")



    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]
    
    article_docs = nlp.pipe(articles, n_process=4, disable=['tagger', 'parser', 'lemmatizer', 'textcat', 'custom'])
    reference_docs = nlp.pipe(references, n_process=4, disable=['tagger', 'parser', 'lemmatizer', 'textcat', 'custom'])
    
    article_label_list = []
    reference_label_list = []

    reference_dict  = {}
    article_dict = {}

    all_keys = []
    for doc in article_docs:
        label_dict = {}
        for ent in doc.ents:
            if ent.label_ not in label_dict.keys():
                label_dict[ent.label_] = 0               
            else:
                label_dict[ent.label_] += 1
            if ent.label_ not in article_dict.keys():
                article_dict[ent.label_] = 0               
            else:
                article_dict[ent.label_] += 1
            if ent.label_ not in all_keys:
                all_keys.append(ent.label_)
        article_label_list.append(label_dict)
    
    for doc in reference_docs:
        label_dict = {}
        for ent in doc.ents:
            if ent.label_ not in label_dict.keys():
                label_dict[ent.label_] = 0               
            else:
                label_dict[ent.label_] += 1
            if ent.label_ not in article_dict.keys():
                article_dict[ent.label_] = 0               
            else:
                article_dict[ent.label_] += 1
            if ent.label_ not in all_keys:
                all_keys.append(ent.label_)
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

    #train_length_stat = get_ner_labels(train_data)
    #print(train_length_stat["article_dict"], train_length_stat["reference_dict"])


    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = get_ner_labels(val_data)
    #print(val_length_stat["article_dict"], val_length_stat["reference_dict"])


    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = get_ner_labels(test_data)
    #print(test_length_stat["article_dict"], test_length_stat["reference_dict"])


    with open("statistics_results/ner_label.json", 'w') as f:
        data = train_data+val_data+test_data
        data_ner_stat = get_ner_labels(data)
        json.dump(data_ner_stat, f)