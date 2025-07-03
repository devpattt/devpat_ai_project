import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('exercise_data.csv')

x = df[['Minutes']]
y = df['Calories']

model = LinearRegression()
model.fit(x, y)
model.predict([[90]])

print(f"Predicted calories burned: {model.predict([[90]])[0]:.2f}")

print (df)