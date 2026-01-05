import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl_lm

import statsmodels.api as sm

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')

moviedata = pd.read_csv("train.csv")

# Fraction of female major speaking roles over time

actors_with_time = moviedata[["Year", "Number of female actors", "Number of male actors"]].groupby("Year").sum()
actors_with_time[["Fraction of women"]] = actors_with_time["Number of female actors"] / (
                                        actors_with_time["Number of female actors"] + 
                                        actors_with_time["Number of male actors"])


Years = np.sort(moviedata["Year"].unique()).reshape(-1, 1)
model = skl_lm.LinearRegression()
model.fit(Years, actors_with_time["Fraction of women"])
actors_predict = model.predict(Years)
print(model.intercept_)
print(model.coef_)

X2 = sm.add_constant(Years)
est = sm.OLS(actors_with_time["Fraction of women"], X2)
est2 = est.fit()
print(est2.summary())
plt.plot(Years, actors_with_time["Fraction of women"], 'o', label="Data")
plt.plot(Years, actors_predict, label="Linear regression")
plt.title("Fraction of women in major speaking roles over the years")
plt.xlabel("Years")
plt.ylabel("Fraction of women major speaking roles")
plt.legend()
plt.show()

# Comparison of total major speaking roles
actors_female = sum(moviedata["Number of female actors"])
actors_male = sum(moviedata["Number of male actors"])
plt.bar(["Female", "Male"], [actors_female, actors_male])
plt.title("Total amount of males and females in major speaking roles")
plt.ylabel("Major speaking roles")
plt.show()




