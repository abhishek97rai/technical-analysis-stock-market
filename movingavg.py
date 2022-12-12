#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('TATASTEEL_data.csv')

#print the head
print(df.head())

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
#plt.figure(figsize=(16,8))
#plt.plot(df['Close'], label='Close Price history')

#plt.show()

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['close'][i] = data['close'][i]
     
#splitting into train and validation
train = new_data[:1200]
valid = new_data[1200:]

print(new_data.shape)
print(train.shape)
print(valid.shape)

print(train['Date'].min())
print(train['Date'].max())
print(valid['Date'].min())
print(valid['Date'].max())

#make predictions
preds = []
for i in range(0,380):
    a = train['close'][len(train)-380+i:].sum() + sum(preds)
    b = a/380
    preds.append(b)
    
#calculate rmse
rms=np.sqrt(np.mean(np.power((np.array(valid['close'])-preds),2)))
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])

plt.show()
