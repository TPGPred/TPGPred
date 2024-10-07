#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import readFasta
from features_code_ys import *


def todataframe(encoding):
    index_list = []
    new_list = []
    for i in encoding[1:]:
        new_list.append(i[1:])
        index_list.append(i[0])
    return new_list, index_list



def feature_inputGenterator(fasta, **kw):
    features = ["AAC","CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DPC","GAAC","GDPC"]
    other = ["CKSAAGP", "DDE","GTPC","KSCTriad", "TPC"]
    features = features + other
    features_number = [{"AAC":20}, {"CKSAAP": 2400}, {"CTDC": 39}, {"CTDT": 39}, {"CTDD": 195}, {"CTriad": 343}, {"DPC": 400}, {"GAAC": 5}, {"GDPC": 25}]  # ys:
    features_number_other = [{"CKSAAGP":150}, {"DDE": 400}, {"GTPC": 125}, {"KSCTriad": 343}, {"TPC": 8000}]  # ys:
    feature_dict = {}
    feature_list = []
    for i in features:
        cmd = i + '.' + i + '(fasta, **kw)'
        encoding = eval(cmd)
        content, index = todataframe(encoding)
        feature_dict[i] = pd.DataFrame(content, columns=encoding[0][1:])
        feature_list.append(pd.DataFrame(content, columns=encoding[0][1:]))
        
            
    df = pd.concat(feature_list, axis=1)
    return df


def generator(fasta_file=None, output_name=None, test_size=100):
    if not fasta_file:
        print("Missing input files")
        print("Start prediction on independent test set samples")
        from del_nanList import del_nanList
        fasta_N = readFasta.readfasta("data/salt_uniprotkb_organism_id_999141_2024_09_23.fasta")
        fasta_N = del_nanList(fasta_N)
        fasta_P = readFasta.readfasta("data/hot_uniprotkb_organism_id_271_2024_09_23.fasta")
        fasta_P = del_nanList(fasta_P)
        fasta = fasta_N[:-test_size] + fasta_P[:-test_size]
        print(f'train data size: fasta_N:{len(fasta_N[:-test_size])}, fasta_P:{len(fasta_P[:-test_size])}')
        fasta_test = fasta_N[-test_size:] + fasta_P[-test_size:]
        print(f'test data size: fasta_N:{len(fasta_N[-test_size:])}, fasta_P:{len(fasta_P[-test_size:])}')
    else:
        fasta = readFasta.readfasta(fasta_file)


    kw = {'order': 'ACDEFGHIKLMNPQRSTVWY'}
    df_feature = feature_inputGenterator(fasta, **kw)
    df_feature_test = feature_inputGenterator(fasta_test, **kw)

    # ys:源数据输出csv
    fasta_df = pd.DataFrame(fasta, columns=['ID', 'Sequence'])
    fasta_test_df = pd.DataFrame(fasta_test, columns=['ID', 'Sequence'])
    fasta_df[['Sequence']].to_csv("features_ys/train_src.csv", index=False)
    fasta_test_df[['Sequence']].to_csv("features_ys/test_src.csv", index=False)

    if not output_name:
        df_feature.to_csv("features_ys/train.csv", index=False)
        df_feature_test.to_csv("features_ys/test.csv", index=False)

if __name__ == "__main__":
    generator(test_size=100)






