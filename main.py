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
target = 'workday'

# for test: table before split used for training, after split for testing
split = 1000

if __name__ == "__main__":
    df = create_prepared_table(target)
    print(df)

    # tree = load_tree_from_file('test')
    tree = build_tree(df.loc[:split])
    pprint.pprint(tree)
    # save_tree_to_file(tree, 'test')

    df_pred = predict_table(tree, df.loc[split:], 'pred')
    print(df_pred)
    print('Accuracy: ', np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1])/df_pred.shape[0])
