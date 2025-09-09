
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

d1 = pd.read_csv("C:\\Users\\Administrator\\Desktop\\project\\data\\dataset1.csv")
d2 = pd.read_csv("C:\\Users\\Administrator\\Desktop\\project\\data\\dataset2.csv")

print("Dataset1 shape:", d1.shape)
print("Dataset2 shape:", d2.shape)

print(d1.info())
print(d2.info())
d1 = d1.dropna()
d2 = d2.dropna()

print("\nDescription of  Dataset1:\n", d1.describe())
print("\nDescription of Dataset2:\n", d2.describe())


risk_count = d1['risk'].value_counts()
print("\nRisk behaviour counts:\n", risk_count)


sns.countplot(x="risk", data=d1)
plt.title("Bat Risk-taking vs Avoidance")
plt.savefig("output/figures/risk_counts.png")
plt.close()

sns.boxplot(x="risk", y="bat_landing_for_food", data=d1)
plt.title("How quickly bats approach food (by risk behaviour)")
plt.savefig("output/figures/landing_time.png")
plt.close()

sns.scatterplot(x="rat_arrival", y="bat_landing", data=d2)
plt.title("Rat arrivals vs Bat landings")
plt.savefig("output/figures/rat_vs_bat.png")
plt.close()


contingency = pd.crosstab(d1['risk'], d1['reward'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\nChi-square Test between risk and reward:")
print("Chi2 =", chi2, " p-value =", p)

with_rats = d1[d1['seconds_after_rat_arrival'] > 0]['bat_landing_to_food']
without_rats = d1[d1['seconds_after_rat_arrival'] == 0]['bat_landing_to_food']
t_stat, p_val = stats.ttest_ind(with_rats, without_rats)
print("\nT-test for landing time (with vs without rats):")
print("t =", t_stat, " p-value =", p_val)



formula = "risk ~ seconds_after_rat_arrival + hours_after_sunset"
model = smf.logit(formula, data=d1)
result = model.fit()
print(result.summary())

print("\nLogistic Regression Result:\n", model.summary())


f= open("outputs/tables/results.txt", "w")
f.write("Chi-square Test between risk and reward:\n")
f.write(f"Chi2={chi2}, p={p}\n\n")
f.write("T-test for landing time:\n")
f.write(f"t={t_stat}, p={p_val}\n\n")
f.write(str(model.summary()))
