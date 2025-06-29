from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[10], [20], [30], [40], [50]])
y = np.array([100, 200, 300, 400, 500])

model = LinearRegression()
model.fit(x, y)

model.predict([[60]])

print(f"Predicted value of calories burned is: {model.predict([[60]])[0]:.2f}")

