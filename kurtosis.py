from scipy.stats import kurtosis
import pandas as pd

print('Letter Recognition Kurtosis\n')
df = pd.read_csv('letter_recognition_ica.csv')
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
for column in x:
    print('Kurtosis of {} is {}'.format(column, kurtosis(df[column])))
print('#############################################################\n')

print('Breast Cancer Kurtosis\n')
df = pd.read_csv('breastcancer.csv')
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
for column in x:
    print('Kurtosis of {} is {}'.format(column, kurtosis(df[column])))