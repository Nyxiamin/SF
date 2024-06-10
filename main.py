# %%
import pandas as pd

# %%
df = pd.read_csv("../EFREI - LIPSTIP - 50k elements EPO.csv")

# %%
print(df.info())
print(df.head())

# %%
print(df)

# %%
print((df.isnull().sum() / len(df)) * 100)
# il n'y a pas de valeur null

# %%
columns = ["Numéro d'application", "Date d'application", "Numero de publication", "date de publication", "IPC"]
df_cleaned = df.drop(columns, axis=1)
print(df_cleaned.head())

#%%
df_cleaned.to_csv('EFREI_LIPSTIP_50k_elements_EPO_clean.csv', sep=',', index=False, encoding='utf-8')