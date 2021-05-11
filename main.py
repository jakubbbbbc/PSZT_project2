import numpy as np
import pandas
import pandas as pd
import pprint
import pickle
from numpy import log2 as log
from tree_ID3 import *



if __name__ == "__main__":
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny,sunny,sunny'.split(
        ',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild,mild,mild'.split(',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal,normal,normal'.split(
        ',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes,no,yes'.split(',')

    dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy, 'play': play}
    df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])

    # df = df.loc[:13]
    print(df)
    df = pd.read_csv('student-mat.csv')
    df=df.iloc[:, :-6]
    print(df)


    tree = build_tree(df)
    pprint.pprint(tree)
    save_tree_to_file(tree, 'test')

    # print(best_inf_gain_att(df))


    # for i in range(5):
    #     print(predict_record(tree, df.iloc[6, :-1]))

    predict_table(tree, df, 'pred')
    print(df)
