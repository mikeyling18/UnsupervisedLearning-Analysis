import pandas as pd
from clustertesters import lr_KMeansTestCluster as tester
import numpy as np

def encode_data(df, target_column):
    df_new = df.copy()
    targets = df_new[target_column].unique()
    intMap = {name: n for n, name in enumerate(targets)}
    df_new[target_column].replace(intMap, inplace=True)
    return(df_new, intMap)

if __name__ == "__main__":
    # letterRec = pd.read_csv("letter_recognition_ica.csv", header=0)
    # ica_df, mapping = encode_data(letterRec, "class")
    # X = ica_df.ix[:,:-1]
    # y = ica_df.ix[:,-1]

    # letterRec = pd.read_csv(
    #     "letter_recognition_pca.csv", header=0)
    # pca_df, mapping = encode_data(letterRec, "class")
    # X = pca_df.ix[:,:-1]
    # y = pca_df.ix[:,-1]

    # letterRec = pd.read_csv("letter_recognition_rp.csv", header=0)
    # rp_df, mapping = encode_data(letterRec, "class")
    # X = rp_df.ix[:,:-1]
    # y = rp_df.ix[:,-1]

    letterRec = pd.read_csv("letter_recognition_ig.csv", header=0)
    ig_df, mapping = encode_data(letterRec, "class")
    X = ig_df.ix[:,:-1]
    y = ig_df.ix[:,-1]

    # y['class'] = y['class'].astype('category').cat.codes
    # y = y.replace({1:0,0:1})
    # y = np.transpose(y.values).flatten()
    runner = tester.KMeansTestCluster(X,y, clusters = range(1,31), plot=True, targetcluster=2, stats=True)
    runner.run()