from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[10], [20], [30], [40], [50]])
y = np.array([100, 200, 300, 400, 500])

model = LinearRegression()
model.fit(x, y)
plt.scatter(x.ravel(), y, color='blue',label='Training Data' )

model.predict([[60]])
plt.plot(x, model.predict(x), color='red', label='Regression Line')

print(f"Predicted value of calories burned is: {model.predict([[60]])[0]:.2f}")
plt.xlabel('Minutes of Exercise')
plt.ylabel('Calories Burned')
plt.title('Calories Burned vs. Minutes of Exercise')
plt.legend()
plt.show()

