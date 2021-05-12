#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" tree_ID3
Authors: Jakub Ciemięga, Krzysztof Piątek
"""

import numpy as np
import pandas as pd
import pickle


def get_entropy(df: pd.DataFrame) -> int:
    """ Calculate entropy for pandas DataFrame

    :param df: DataFrame to calculate the entropy for
    :type df: pd.DataFrame

    :return entropy: calculated entropy
    :rtype entropy: int
    """

    # get unique values and their counts for the last column of df
    values, val_freq = np.unique(df[df.columns[-1]], return_counts=True)
    val_freq = val_freq / val_freq.sum()

    entropy = 0
    for freq in val_freq:
        entropy -= freq * np.log2(freq)
    return entropy


def get_inf(df: pd.DataFrame, col: str) -> int:
    """ Calculate normalized entropy for df divided by values of col

    :param df: DataFrame to work on
    :type df: pd.DataFrame

    :param col: column name which values are used to divide df
    :type col: str

    :return inf: calculated normalized entropy
    :rtype inf: int
    """

    inf = 0
    # for each unique value in col
    for val in np.unique(df[col]):
        # create subtable where df[col]=val
        df_temp = df[df[col] == val]
        # calculate normalized entropy
        inf += get_entropy(df_temp) * df_temp.shape[0] / df.shape[0]

    return inf


def best_inf_gain_att(df: pd.DataFrame) -> str:
    """ determine column of df that produces best information gain

    :param df: DataFrame to work on
    :type df: pd.DataFrame

    :return: name of column offering best information gain
    :rtype: str
    """
    # exclude target column
    columns = df.columns[:-1]
    inf_gain = []
    for i, col in enumerate(columns):
        inf_gain.append(get_entropy(df) - get_inf(df, col))

    return columns[np.argmax(inf_gain)]


def get_subtable(df, col, val) -> pd.DataFrame:
    """ return subtable created from df by picking rows where df[col]=val and then dropping col """
    return df[df[col] == val].drop(columns=col)


def build_tree(df) -> dict:
    """ generate an ID3 decision tree based on df

    :param df: dataframe to base the tree on
    :type df: pd.DataFrame

    :return tree: ID3 decision tree based on df
    :rtype tree: dict
    """
    # initialize empty tree as a dictionary
    tree = {}
    # find column associated with best information gain
    next_att = best_inf_gain_att(df)
    # next_att = find_winner(df)
    tree[next_att] = {}

    # for each value of the attribute at hand
    for val in np.unique(df[next_att]):
        # get new table
        subtable = get_subtable(df, next_att, val)
        # get information on new y characteristics
        sub_val, sub_val_counts = np.unique(subtable.iloc[:, -1], return_counts=True)

        # if there's only one label value left, assign it
        if 1 == sub_val.shape[0]:
            tree[next_att][val] = sub_val[0]
        # if there are no more columns except the label column, assign the most frequent label
        elif 1 == subtable.columns.shape[0]:
            tree[next_att][val] = sub_val[np.argmax(sub_val_counts)]
        # otherwise add node recursively
        else:
            tree[next_att][val] = build_tree(subtable)

    return tree


def predict_record(tree: dict, data: pd.Series):
    """ predict outcome for a data series by using an ID3 decision tree

    :param tree: ID3 decision tree for classification
    :type tree: dict

    :param data: series to base the classification on
    :type data: pd.Series

    :return: predicted value for the series
    :rtype: dependant on predicted value
    """
    tree = tree.copy()
    while type(tree) == dict:  # while the tree doesn't only contain a label (leaf)
        col = list(tree.keys())[0]  # assign the next column to consider
        try:
            tree = tree[col][data[col]]  # move to the next column part of the tree
        except:
            tree = tree[col][
                list(tree[col].keys())[0]]  # if unknown class encountered (not included in the tree), pick first class

    predicted_value = tree
    return predicted_value


def predict_table(tree: dict, df: pd.DataFrame, col_name: str = 'predicted') -> pd.DataFrame:
    """ Predict values for a data set using an ID3 decision tree

    :param tree: ID3 decision tree for classification
    :type tree: dict

    :param df: data set to base the classification on
    :type df: pd.DataFrame

    :param col_name: name of the column for predicted values; if nonexistent, added
    :type col_name: str

    :return df_predict: df with added column of predictions
    :rtype df_predict: pd.DataFrame
    """
    df = df.copy()
    df[col_name] = np.nan
    for j in range(df.shape[0]):
        df.iloc[j, -1] = predict_record(tree, df.iloc[j])

    df_predict = df
    return df_predict


def save_tree_to_file(tree: dict, fname: str = 'tree') -> None:
    """ save decision tree to a .pkl file

    :param tree: tree to save
    :type tree: dict

    :param fname: name of the file, the tree is saved in <fname>.pkl
    :type fname: str

    :return: None
    """
    fid = open(fname + ".pkl", "wb")
    pickle.dump(tree, fid)
    fid.close()


def load_tree_from_file(fname: str = 'tree') -> dict:
    """ load ID3 decision tree from a .pkl file

    :param fname: name of the file, the tree is loaded from <fname>.pkl
    :type fname: str

    :return tree: ID3 decision tree
    :rtype tree: dict
    """
    fid = open(fname + ".pkl", "rb")
    return pickle.load(fid)
