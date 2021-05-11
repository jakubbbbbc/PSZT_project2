import numpy as np
import pandas as pd
import pickle


def entropy (df: pd.DataFrame) -> int:
    values, val_freq = np.unique(df[df.columns[-1]], return_counts=True)
    val_freq = val_freq / val_freq.sum()
    entropy = 0
    for freq in val_freq:
        entropy -= freq * np.log2(freq)
    return entropy


def inf(df: pd.DataFrame, col: str) -> int:
    inf = 0
    for val in np.unique(df[col]):
        df_temp = df[df[col]==val]
        inf += entropy(df_temp) * df_temp.shape[0] / df.shape[0]

    return inf


def best_inf_gain_att(df: pd.DataFrame) -> str:
    columns = df.columns[:-1]
    inf_gain = []
    for i, col in enumerate(columns):
        inf_gain.append(entropy(df) - inf(df, col))
    print(inf_gain)

    return columns[np.argmax(inf_gain)]


def get_subtable(df, col, val):
    return df[df[col] == val].drop(columns=col)


def build_tree(df):
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


def predict_record(tree, data):
    tree = tree.copy()
    while type(tree) == dict:  # while the tree doesn't only contain a label (leaf)
        col = list(tree.keys())[0]  # assign the next column to consider
        tree = tree[col][data[col]]  # move to the next column part of the tree
    return tree


def predict_table(tree, df, col_name):
    for j in range(df.shape[0]):
        df.loc[j, col_name] = predict_record(tree, df.loc[j])


def save_tree_to_file(tree: dict, fname: str = 'tree'):
    fid = open(fname + ".pkl", "wb")
    pickle.dump(tree, fid)
    fid.close()


def load_tree_from_file(fname: str = 'tree') -> dict:
    fid = open(fname + ".pkl", "rb")
    return pickle.load(fid)
