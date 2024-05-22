"""
Notes
  1.Data normalization is a must
  2.You should find the best learning rate and number of iterations that suits you
  3.If you are going to use pandas please add .values after reading your dataset
      Ex. df = pd.read_csv('Your_Dataset').values
  4.You might need to visualize the output to make sure it works as expected
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#----------------------------------------------------------------------------
#----------------------- Logistic Regression --------------------------------
#----------------------------------------------------------------------------

class LogisticRegression():

#----------------------------------------------------------------------------
#----------------------- Constructor ----------------------------------------
#----------------------------------------------------------------------------

    def __init__(self, alpha=0.001, number_iterations = 10000):
        self.alpha = alpha
        self.intercept = 0
        self.slope = 0
        self.number_iterations = number_iterations

#----------------------------------------------------------------------------
#----------------------- Generate Random Weights ----------------------------
#----------------------------------------------------------------------------

    def generate_random(self):
        self.slope = np.random.randn()
        self.intercept = np.random.randn()

#----------------------------------------------------------------------------
#----------------------- Predict (Use Model) --------------------------------
#----------------------------------------------------------------------------

    def predict(self, X, visualization):
        z = self.intercept + np.array(X) * self.slope
        result = 1 / (1 + np.exp(-z))
        if visualization == 'class':
          return np.round(result)
        elif visualization == 'percentage':
          return result

#----------------------------------------------------------------------------
#----------------------- Calculate Cost -------------------------------------
#----------------------------------------------------------------------------

    def cost(self, Y, predicted):
        error_avoidance = 1e-15
        cost = np.zeros_like(predicted)
        cost[Y == 1] = -np.log(predicted[Y == 1] + error_avoidance)
        cost[Y == 0] = -np.log(1 - predicted[Y == 0] + error_avoidance)
        return cost

#----------------------------------------------------------------------------
#----------------------- Fitting Function -----------------------------------
#----------------------------------------------------------------------------

    def fit(self, X, Y):
        self.generate_random()
        for i in range(10000):
            predicted = self.predict(X, visualization = "percentage")
            cost = self.cost(Y, predicted)
            J = np.mean(cost)
            self.intercept -= self.alpha * np.mean(predicted - Y)
            self.slope -= self.alpha * np.mean((predicted - Y) * X)


# Example Usage
X = np.arange(1, 101, 1).reshape(-1, 1)
Y = np.array([0] * 50 + [1] * 50)

model = LogisticRegression(alpha=0.01, number_iterations = 1500)

scaler = MinMaxScaler(copy = False)
X = scaler.fit_transform(X).reshape(-1, 1)

model.fit(X, Y)

print(f'Intercept: {model.intercept}')
print(f'Slope: {model.slope}')

"""
X = scaler.inverse_transform(X)
x_axis = np.arange(-50, 50, 1)
plt_predictions = model.predict(x_axis, "percentage")
plt.scatter(X, plt_predictions)
plt.xlabel('X')
plt.ylabel('Predictions')
plt.show()
print(f'Special Prediction: {model.predict([0.1], visualization = "percentage")}')
"""