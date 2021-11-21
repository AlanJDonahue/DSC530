# -*- coding: utf-8 -*-
"""
Author: Alan Donahue
DSC 530 Final Term Project
Date: 20 Nov 21

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm

#import data
df = pd.read_csv("heart.csv")
print(df.columns)

# Age variable information / histogram
n = len(pd.unique(df["Age"]))
print(n)
print(df["Age"].min())
print(df["Age"].max())

age = df["Age"]
n, bins, patches = plt.hist(age, 50, facecolor = "b", alpha = 0.75, edgecolor = "black")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Histogram")
plt.show()

# Sex variable histogram
n = len(pd.unique(df["Sex"]))
print(n)
sex = df["Sex"]
n, bins, patches = plt.hist(sex, 2, facecolor = "b", alpha = 0.75, edgecolor = "black", align = "mid")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex Histogram")
plt.show()

# ChestPainType variable histogram
n = len(pd.unique(df["ChestPainType"]))
print(n)
chestPain = df["ChestPainType"]
n, bins, patches = plt.hist(chestPain, 4, facecolor = "b", alpha = 0.75, edgecolor = "black", rwidth = .6)
plt.xlabel("Types of Chest Pain")
plt.ylabel("Count")
plt.title("Chest Pain Histogram")
plt.show()

# RestingBP variable histogram
n = len(pd.unique(df["RestingBP"]))
print(n)
print(df["RestingBP"].min())
print(df["RestingBP"].max())

restingBP = df["RestingBP"]
n, bins, patches = plt.hist(restingBP, 67, facecolor = "b", alpha = 0.75, edgecolor = "black")
plt.xlabel("Resting BP")
plt.ylabel("Count")
plt.title("Resting BP Histogram")
plt.show()

# Choleesterol variable histogram
n = len(pd.unique(df["Cholesterol"]))
print(n)
print(df["Cholesterol"].min())
print(df["Cholesterol"].max())

cholesterol = df["Cholesterol"]
n, bins, patches = plt.hist(cholesterol, 60, facecolor = "b", alpha = 0.75, edgecolor = "black")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.title("Cholesterol Histogram")
plt.show()


# Descriptive Statistics
# Age descriptive
df["Age"].var()
df["Age"].mode()
df["Age"].describe()

# Sex descriptive
#df["Sex"].var() Doesn't make sense
df["Sex"].mode()
df["Sex"].describe()

#Chest Pain descriptive
#df["ChestPainType"].var() Doesn't make sense
df["ChestPainType"].mode()
df["ChestPainType"].describe()

#RestingBP descriptive
df["RestingBP"].replace(0, np.NaN, inplace = True)
df["RestingBP"].var()
df["RestingBP"].mode()
df["RestingBP"].describe()

#Cholesterol descriptive
df["Cholesterol"].replace(0, np.NaN, inplace = True)
df["Cholesterol"].var()
df["Cholesterol"].mode()
df["Cholesterol"].describe()

# PMF section
# split data by chest pain type
TA = df[df.ChestPainType == "TA"]
TA_pmf = TA["HeartDisease"].value_counts().sort_index() / len(TA["HeartDisease"])


ATA = df[df.ChestPainType == "ATA"]
ATA_pmf = ATA["HeartDisease"].value_counts().sort_index() / len(ATA["HeartDisease"])


NAP = df[df.ChestPainType == "NAP"]
NAP_pmf = NAP["HeartDisease"].value_counts().sort_index() / len(NAP["HeartDisease"])

ASY = df[df.ChestPainType == "ASY"]
ASY_pmf = ASY["HeartDisease"].value_counts().sort_index() / len(ASY["HeartDisease"])


# Plotting PMF 
X = ["No Heart Disease", "Heart Disease"]
X_axis = np.arange(len(X))
width = 0.05
ticks = (0, 1)

bar1 = plt.bar(X_axis, TA_pmf, width, color = "r")
bar2 = plt.bar(X_axis+width, ATA_pmf, width, color = "b")
bar3 = plt.bar(X_axis+width*2, NAP_pmf, width, color = "g")
bar4 = plt.bar(X_axis+width*3, ASY_pmf, width, color = "y")

plt.xticks(ticks)
plt.xlabel("Chest Pain Types")
plt.ylabel("Probability")
plt.title("Heart Disease PMF")
plt.legend((bar1, bar2, bar3, bar4), ("TA", "ATA", "NAP", "ASY"))
plt.show()

# CDF section
chestPain = df["ChestPainType"]
heartDisease = df["HeartDisease"]
age = df["Age"]
def cdf(chestPainType):
    x, counts = np.unique(chestPainType, return_counts = True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def plot_cdf(chestPainType):
    x, y = cdf(chestPainType)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle = "steps-post")
    plt.grid(True)
    plt.show()
    
plot_cdf(chestPain)
plot_cdf(heartDisease)
plot_cdf(age)

# Analytical Distribution 
mu, sigma = 53.510893, 9.432617
BR_ND = np.random.normal(mu, sigma, 1000)
abs(mu-np.mean(BR_ND))
abs(sigma-np.std(BR_ND, ddof = 1))

count, bins, ignored = plt.hist(BR_ND, 30, density = True)
plt.plot(bins, 1/(sigma * np.sqrt(2 *np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.show()

# Scatter Plot Section
plt.scatter(cholesterol, restingBP, alpha = 0.2)
plt.xlabel("Cholesterol")
plt.ylabel("Resting BP")
plt.title("Cholesterol vs Resting BP")
plt.show()

plt.scatter(restingBP, heartDisease, alpha = .2)
plt.xlabel("Resting BP")
plt.ylabel("Heart Disease")
plt.title("Resting BP vs Heart Disease")
plt.show()

plt.scatter(cholesterol, heartDisease, alpha = 0.2)
plt.xlabel("Cholesterol")
plt.ylabel("Heart Disease")
plt.title("Cholesterol vs Heart Disease")
plt.show()

# covariance of cholesterol and restingBP
df.cov()
scipy.stats.spearmanr(df["Cholesterol"], df["RestingBP"], nan_policy = "omit")
scipy.stats.pearsonr(df["Age"], df["HeartDisease"])
scipy.stats.spearmanr(df["Cholesterol"], df["HeartDisease"], nan_policy = "omit")
scipy.stats.spearmanr(df["RestingBP"], df["HeartDisease"], nan_policy = "omit")

# test of hypothesis
scipy.stats.ttest_ind(restingBP, heartDisease, nan_policy = "omit", equal_var = False)
scipy.stats.ttest_ind(restingBP, cholesterol, nan_policy = "omit", equal_var = False)

# regression analysis
logit_model = sm.Logit(heartDisease, restingBP, missing = "drop")
result = logit_model.fit()
print(result.summary2())
