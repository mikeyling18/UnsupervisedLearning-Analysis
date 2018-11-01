from sklearn import  datasets
from clustertesters import ExpectationMaximizationTestCluster as tester
import pandas as pd
import numpy as np



if __name__ == "__main__":
    # Load data set
    breast_cancer = datasets.load_breast_cancer()

    # ica_df = pd.read_csv("breastcancer_ica.csv")
    # X = ica_df.iloc[:,0:-1]
    # y = pd.DataFrame(ica_df.iloc[:,-1])
    #
    # pca_df = pd.read_csv("breastcancer_pca.csv")
    # X = pca_df.iloc[:,0:-1]
    # y = pd.DataFrame(pca_df.iloc[:,-1])

    # ig_df = pd.read_csv("breastcancer_ig.csv")
    # X = ig_df.iloc[:,0:-1]
    # y = pd.DataFrame(ig_df.iloc[:,-1])

    rp_df = pd.read_csv("breastcancer_rp.csv")
    X = rp_df.iloc[:, 0:-1]
    y = pd.DataFrame(rp_df.iloc[:, -1])

    y['class'] = y['class'].astype('category').cat.codes
    y = y.replace({1:0,0:1})
    y = np.transpose(y.values).flatten()

    # Get Data and Labels
    # X, y = breast_cancer.data, breast_cancer.target
    # Initialize Tester and Run instance
    runner = tester.ExpectationMaximizationTestCluster(X, y, clusters=range(1,10), plot=True, targetcluster=2, stats=True)
    runner.run()