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

# size of one data frame
split = 104
# number of attributes to delete for experiment
delete_columns = 0
acc_list = []


if __name__ == "__main__":

    df = create_prepared_table(delete_columns, target)
    print(df)

    for i in range(1, 11):
        if i == 1:
            learn_df = df.loc[split+1:]
            print(learn_df)
            # tree = load_tree_from_file('test')
            tree = build_tree(learn_df)
            pprint.pprint(tree)
            # save_tree_to_file(tree, 'test')
            df_pred = predict_table(tree, df.loc[:split], 'pred')
        elif i == 10:
            learn_df = df.loc[:split*9]
            print(learn_df)
            # tree = load_tree_from_file('test')
            tree = build_tree(learn_df)
            pprint.pprint(tree)
            # save_tree_to_file(tree, 'test')
            df_pred = predict_table(tree, df.loc[(split*9)+1:], 'pred')
        else:
            learn_df = pd.concat([df.loc[:(split*i)], df.loc[split*(i+1)+1:]])
            print(learn_df)
            # tree = load_tree_from_file('test')
            tree = build_tree(learn_df)
            pprint.pprint(tree)
            # save_tree_to_file(tree, 'test')
            df_pred = predict_table(tree, df.loc[(split*i)+1:split*(i+1)], 'pred')

        print(df_pred)
        print('Accuracy: ', np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1])/df_pred.shape[0], '\n')
        acc_list.append(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1])/df_pred.shape[0])

    for n in range(len(acc_list)):
        print("Data set ", n+1, " accuracy:", acc_list[n])
    average_acc = sum(acc_list)/10
    print('Average accuracy: ', average_acc)
