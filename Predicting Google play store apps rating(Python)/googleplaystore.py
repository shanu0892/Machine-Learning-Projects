import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


df = pd.read_csv("C:\\Users\\Arun\Documents\\shanu\\kaggle\\googleplaystore.csv")
df.describe()
df.info()

df.isna().sum().sort_values(ascending=False)

df.dropna(how="any", inplace= True)

df.isna().sum().sort_values(ascending=False)


sns.kdeplot(df["Rating"],legend= True)
plt.show()
# Rating ranges between 4 and 5 and so many have given it

df["Rating"].mean()
sns.kdeplot(df["Rating"],legend= True)
plt.show()
