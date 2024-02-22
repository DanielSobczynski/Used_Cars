import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Loading data about used cars prices
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filepath, names=headers)

# ------------ Dealing with missing values ---------------
#Replacing data missed values with NaN
df.replace("?", np.nan, inplace = True)

#Detecting and counting missed values in columns
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

#Calculating mean value of "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

#Replacing "NaN" values of "normalized-losses" column with mean value
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculating mean value of "bore" column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

#Replacing "NaN" values of "bore" column with mean value
df["bore"].replace(np.nan, avg_bore, inplace=True)

#Calculating mean value of "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)

#Replacing "NaN" values of "stroke" column with mean value
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

#Calculating mean value of "horsepower" column
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)

#Replacing "NaN" values of "horsepower" column with mean value
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

#Calculating mean value of "peak-rpm" column
avg_peakrpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm:", avg_peakrpm)

#Replacing "NaN" values of "peak-rpm" column with mean value
df["peak-rpm"].replace(np.nan, avg_peakrpm, inplace=True)

#Checking which values are present in num-of-doors column
print(df['num-of-doors'].value_counts())

#Replacing the missing values of 'num-of-doors' by the most frequent one
df["num-of-doors"].replace(np.nan, "four", inplace=True)

#Dropping all rows that do not have price data (which will be predicted)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# ------------ Dealing with data formatting ---------------
#print(df.dtypes)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

# ------------ Data standarization ---------------
#print(df.head())

df['city-L/100km'] = 235/df["city-mpg"]
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
print(df.head())

# ------------ Data normalization ---------------
# Replacing original values by normalized index: (original value)/(maximum value) [0 to 1]
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# Checking normalized columns
df[["length","width","height"]].head()

# ------------ Binning ---------------
# Converting data into int format
df["horsepower"]=df["horsepower"].astype(int, copy=True)

# Plotting horsepower column on histogram to check 3 classes for further analysis
plt.hist(df["horsepower"])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()

# Binning horsepower column
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)
print(df["horsepower-binned"].value_counts())

# Ploting binned horsepower column
plt.bar(group_names, df["horsepower-binned"].value_counts())
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()

# Histogram to visualize the distribution of bins
plt.hist(df["horsepower"], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

# ------------ Indicator Variable ---------------
print(df.columns)

#Getting indicators
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()

# Merging data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# Dropping original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
df.head()

#Exporting results
df.to_csv('Preprocessed_df.csv')