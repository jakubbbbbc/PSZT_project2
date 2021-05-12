#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fundamentals of Artificial Intelligence
Project 2
Jakub Ciemięga, Krzysztof Piątek 2021
Warsaw University of Technology
"""

import pprint
from tree_ID3 import *
from data_manager import *

# decide what to predict, workday or weekend
target = 'weekend'

split = 1000

if __name__ == "__main__":
    df = create_prepared_table(target)
    # df = df.loc[:500]
    # df.iloc[0]['babaa'] = 3
    print(df)

    # tree = load_tree_from_file('test')
    tree = build_tree(df.loc[:split])
    # pprint.pprint(tree)
    # save_tree_to_file(tree, 'test')

    df_pred = predict_table(tree, df.loc[split:], 'pred')
    print(df_pred)
    print(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1]))
