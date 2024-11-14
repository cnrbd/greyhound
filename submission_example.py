import pandas as pd
from sklearn.linear_model import LinearRegression

# Reading the CSV files
x1 = pd.read_csv("df.csv")
x2 = pd.read_csv("unseendf.csv")

X = x1[['time1', 'distance1', 'distance2', 'trap2']]
y = x1['time2']

# Create the linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict using the new data
x2['predtime'] = model.predict(x2[['time1', 'distance1', 'distance2', 'trap2']])
x2.to_csv("~/Downloads/mypred.csv", index=False)

