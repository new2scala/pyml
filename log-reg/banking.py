import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

# from https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('banking.csv', header=0)
data = data.dropna()

print(data.shape)
print(list(data.columns))

print(data.head())

print(data['education'].unique())

data['education']=np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] == 'basic.4y', 'Basic', data['education'])

print(data['education'].unique())

print(data['y'].value_counts())

# sns.countplot(x='y', data=data, palette='hls')
# plt.show()
#plt.savefig('count_plot')

print(data.groupby('y').mean())
print(data.groupby('job').mean())
print(data.groupby('marital').mean())

# pd.crosstab(data.job, data.y).plot(kind='bar',stacked=True)
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.show()

tb = pd.crosstab(data.education, data.y)
tb.div(tb.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)

plt.title('Purchase Frequency for education')
plt.xlabel('education')
plt.ylabel('Frequency of Purchase')
plt.show()
# plt.savefig('')

data.age.hist()
plt.title("histogram of age")
plt.show()