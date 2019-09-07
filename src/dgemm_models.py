#######################################################################
# Import packages
#######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as sty
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import csv

sty.use('seaborn-bright')

#######################################################################
# Making data ready to use
#######################################################################

data = pd.read_csv('../data/dgemm.csv')
print(data.head(5))

newData = (
    data
    .rename(columns={'time': 'Time', 'ps': 'ProblemSize', 'energy': 'WattsUpPro',
                     'rapl': 'RAPL', 'flops': 'CPU FLOPS'
                     })
    )

"""
print(newData.head(5))
print(newData.shape)
"""

######################################################################
# correlation of data
#######################################################################

corr = newData.corr()
print(corr)

# Plot correlation coefficiengs as a heat map

sns.heatmap(corr, vmax=1, linewidths=0.5, cbar_kws={"shrink": .5})
plt.title('Correlation matrix for DGEMM')
plt.show()

# Uncomment the line below to save the heatmap in local machine
# plt.savefig('ResultsDgemm/dgemm-correlation.png', format='png', dpi=500)

#######################################################################
# pair plots for data
#######################################################################

"""
sns.pairplot(newData, x_vars=['Rapl', 'CPU FLOPS', 'ProblemSize'],
             y_vars='WattsUpPro', height=4, kind='reg', aspect=0.7)
plt.show()

sns.pairplot(newData, diag_kind='kde', markers='+')
plt.show()
"""

#######################################################################
# preparing input (X) and target data (y) for models
#######################################################################

X = newData[['Time', 'RAPL', 'CPU FLOPS', 'ProblemSize']]
y = newData[['WattsUpPro']]

"""
print(X.head(5))
print(y.head(5))
"""

#######################################################################
# split data in training and test sets
#######################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(X_test.head(6))
print(y_train.shape)
print(y_test.shape)

# export_csv = y_test.to_csv(r'C:\Users\Arslan\Desktop\pmc models python\y_test.csv', index=None, header=True)
# export_csv = X_test.to_csv(r'C:\Users\Arslan\Desktop\pmc models python\X_test.csv', index=None, header=True)
# export_csv = X_train.to_csv(r'C:\Users\Arslan\Desktop\pmc models python\X_train.csv', index=None, header=True)
# export_csv = y_train.to_csv(r'C:\Users\Arslan\Desktop\pmc models python\y_train.csv', index=None, header=True)

#######################################################################
# Applying linear regression to data
#######################################################################

# Model A
# Type: Linear
# Predictor variables: Rapl, FLOPS
# Output variable = WattsUp
# With Intercept

modelA = LinearRegression(fit_intercept=True)
modelAfit = modelA.fit(X_train, y_train)

print("\n Model A")
print("Intercept", modelAfit.intercept_)
print("Coefficients", modelAfit.coef_)

# reset y_test indexes
y_test_new = y_test.reset_index(drop=True)

modelApredict = modelA.predict(X_test)

modelAprediction = pd.DataFrame(modelApredict)\
    .rename(columns={0: 'ModelAPrediction'})\
    .assign(errors=lambda x: (((y_test_new.WattsUpPro -
                               x['ModelAPrediction']) /
                              y_test_new.WattsUpPro)*100).abs()
            )

print("Prediction Mean: %.2f , Max: %.2f , Min: %.2f " %
      (modelAprediction.errors.mean(),
       modelAprediction.errors.max(),
       modelAprediction.errors.min()
       )
      )

"""
# uncomment to print prediction results
print(modelAprediction)
"""

# Model B
# Type: Linear
# Predictor variables: Rapl, FLOPS
# Output variable = WattsUp
# Without Intercept

modelB = LinearRegression(fit_intercept=False)

modelBfit = modelB.fit(X_train, y_train)

print("\n Model B")
print("Intercept", modelBfit.intercept_)
print("Coefficients", modelBfit.coef_)

modelBpredict = modelB.predict(X_test)

modelBprediction = pd.DataFrame(modelBpredict)\
    .rename(columns={0: 'ModelBPrediction'})\
    .assign(errors=lambda x: (((y_test_new.WattsUpPro -
                               x['ModelBPrediction']) /
                              y_test_new.WattsUpPro)*100).abs()
            )

print("Prediction Mean: %.2f , Max: %.2f , Min: %.2f " %
      (modelBprediction.errors.mean(),
       modelBprediction.errors.max(),
       modelBprediction.errors.min()
       )
      )

"""
# uncomment to print prediction results
print(modelAprediction.head(5))
print(modelBprediction.head(5))
"""

# Model C
# Type: Linear
# Predictor variables: Rapl, FLOPS
# Output variable = WattsUp
# Without Intercept
# positive coefficients only

modelC = Lasso(alpha=0.0001, precompute=True, max_iter=10000,
               fit_intercept=False, positive=True, random_state=9999,
               selection='random')

modelCfit = modelC.fit(X_train, y_train)

print("\n Model C")
print("Intercept", modelCfit.intercept_)
print("Coefficients", modelCfit.coef_)

modelCpredict = modelC.predict(X_test)

modelCprediction = pd.DataFrame(modelCpredict)\
    .rename(columns={0: 'ModelCPrediction'})\
    .assign(errors=lambda x: (((y_test_new.WattsUpPro -
                               x['ModelCPrediction']) /
                              y_test_new.WattsUpPro)*100).abs()
            )

print("Prediction Mean: %.2f , Max: %.2f , Min: %.2f " %
      (modelCprediction.errors.mean(),
       modelCprediction.errors.max(),
       modelCprediction.errors.min()
       )
      )

#######################################################################
# Plotting Model predictions
#######################################################################

modelPlot = sns.lineplot(x=y_test.index, y=y_test.WattsUpPro,
                          label='HCLWattsUp', color='brown',
                          markers=True, marker='o', markersize=5,
                          markeredgecolor='red'
                         )

modelPlot = sns.lineplot(x=y_test.index, y=modelCprediction.
                         ModelCPrediction, label='Model MM',
                         color='green', markers=True,
                         marker='+', markersize=5,
                         markeredgecolor='green'
                         )

modelPlot = sns.lineplot(x=y_test.index, y=X_test.RAPL, label='RAPL',
                         color='blue', markers=True,
                         marker='*', markersize=5,
                         markeredgecolor='blue'
                         )

modelPlot.set(xlabel='Problem Sizes', ylabel='Dynamic Energy [J]')
plt.title('Comparison of Model Predictions')
plt.show()
# Uncomment the line below to save the heatmap in local machine
#plt.savefig('ResultsDgemm/fft-predictions.png', format='png', dpi=1000)

sns.distplot(modelAprediction.errors.abs(), rug=True, hist=True, color='r', label='Model A')
sns.distplot(modelBprediction.errors.abs(), rug=True, hist=True, color='g', label='Model B')
sns.distplot(modelCprediction.errors.abs(), rug=True, hist=True, color='b', label='Model C')
plt.xlabel('Model Errors')
plt.legend()
plt.show()

