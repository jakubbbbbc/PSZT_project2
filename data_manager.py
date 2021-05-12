#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" tree_ID3
Authors: Jakub Ciemięga, Krzysztof Piątek
"""
import pandas as pd


def create_test_table() -> pd.DataFrame:
    """ create simple data set for tests

    :return df: test data set
    :rtype df: pd.DataFrame
    """
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny,sunny,sunny'.split(
        ',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild,mild,mild'.split(',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal,normal,normal'.split(
        ',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes,no,yes'.split(',')

    dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy, 'play': play}
    df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])

    return df


def create_prepared_table(variant: str = 'workday') -> pd.DataFrame:
    """ merge and prepare two datasets from https://www.kaggle.com/uciml/student-alcohol-consumption/.

    Merge datasets containing information about students in a maths and Portuguese courses. Depending on the target the
    last column can be either Walc or Dalc (weekend or workday alcohol consumption, respectively)

    :param variant: if to prepare dataframe for predicting alcohol consumption on workdays or on weekends
    :type variant: str

    :return df: merged and prepared dataframe to apply ID3 algorithm to
    :rtype df: pd.DataFrame
    """
    df1 = pd.read_csv('student-mat.csv')
    df2 = pd.read_csv('student-por.csv')
    df = pd.concat([df1, df2], ignore_index=True) # ignore index because there are index duplicates
    columns = list(df.columns)
    # delete Walc and Dalc from the middle
    del columns[26:28]
    # add column to predict
    if 'workday' == variant:
        columns.append('Dalc')
    elif 'weekend' == variant:
        columns.append('Walc')

    df = df[columns]

    return df
