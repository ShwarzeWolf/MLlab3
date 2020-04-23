import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

chipsDataset = pd.read_csv("C://Users//1358365//PycharmProjects//lab3//datasets//chips.csv")
geyserDataset = pd.read_csv("C://Users//1358365//PycharmProjects//lab3//datasets//geyser.csv")

from matplotlib import pyplot as plt
chipsDatasetP = chipsDataset[chipsDataset.className == "P"]
chipsDatasetN = chipsDataset[chipsDataset.className == "N"]

plt.scatter(chipsDatasetN['x'], chipsDatasetN['y'], color='green', marker='+')
plt.scatter(chipsDatasetP['x'], chipsDatasetP['y'], color='red', marker='+')
plt.title("dataset visualisation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

from sklearn.model_selection import train_test_split
X = chipsDataset.drop(['className'], axis='columns')
Y = chipsDataset.className
Y = Y.replace('P', 0)
Y = Y.replace('N', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

kernels = ['linear', 'sigmoid', 'rbf']

for kernel in kernels:
    bestScore = 100
    bestC = 0
    bestGamma = 0

    for C in range(10):
        for gamma in range(20):
                model = SVC(kernel=kernel, C = (C + 1) / 10.0 , gamma = (gamma + 1) / 100.0)
                model.fit(X_train, Y_train)
                currentScore = model.score(X_test, Y_test)
                if (currentScore < bestScore):
                    bestScore = currentScore
                    bestC = (C + 1) / 10.0
                    bestGamma = (gamma + 1) / 100.0



    model = SVC(kernel=kernel, C = bestC, gamma = bestGamma)
    model.fit(X_train, Y_train)
    print(kernel + " " + str(bestGamma) + " " +str(bestC) + " " + str(bestScore))
    plt.title("best result for kernel " + kernel)

    plot_decision_regions(X.to_numpy(), Y.to_numpy(),  clf=model, legend=2)
    plt.show()


#poly
bestScore = 100
bestDegree = 3


for degree in range(50):
                model = SVC(kernel=kernel, degree = degree)
                model.fit(X_train, Y_train)
                currentScore = model.score(X_test, Y_test)
                if (currentScore < bestScore):
                    bestScore = currentScore
                    bestDegree = degree


model = SVC(kernel=kernel, degree= bestDegree)
model.fit(X_train, Y_train)
print("poly " + str(bestGamma) + " " +str(bestC) + " " + str(bestDegree) + " " + str(bestScore))
plt.title("best result for kernel poly")

plot_decision_regions(X.to_numpy(), Y.to_numpy(),  clf=model, legend=2)
plt.show()