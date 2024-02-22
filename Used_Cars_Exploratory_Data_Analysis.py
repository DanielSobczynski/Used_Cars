import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Downloading data
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath, header=0)

# Getting familiar with data
print(df.head().to_string())
print(df.dtypes)

print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Exploring linear relationships (Numerical Variables)
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
#plt.show()
print(df[["engine-size", "price"]].corr())

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#plt.show()
print(df[['highway-mpg', 'price']].corr())

# Exploring relationships (Categorical Variables)
sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.show()

# Descriptive Statistical Analysis
df.describe()
df.describe(include=['object'])

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

# Exploring Variables: Drive Wheels and Body Style vs. Price

df_group = df[['body-style','price']]
grouped_test_bodystyle = df_group.groupby(['body-style'],as_index= False).mean()
print(grouped_test_bodystyle)

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

# Visualize how the price is related to 'drive-wheel' and 'body-style' (Heat Map)
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# Move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# Rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Exploring Correlation between Price and other variables
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#linear relationship is positive but not strong (~0.585)

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#linear relationship is positive and quite strong

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#linear relationship is negative and moderately strong.

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#linear relationship is negative and quite strong

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#linear relationship is positive and quite strong.

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#inear relationship is positive and very strong.

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value)
#statistically significant, but the linear relationship is only moderate.

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#relationship is negative and moderately strong.

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#relationship is negative and moderately strong.