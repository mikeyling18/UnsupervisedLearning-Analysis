
from sklearn import  datasets, metrics
from clustertesters import lr_ExpectationMaximizationTestCluster as emtc
import pandas as pd

def encode_data(df, target_column):

    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

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

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,31), plot=True, targetcluster=26, stats=True)
    tester.run()

