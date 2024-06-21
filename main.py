# %%
import random
from functionDataFrame import readDataframe, createdCleanCSV
from functionKNNcomb import KNN
from functionTXT import transform_to_txt
from functiontest import functiontest

# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)
#%%
#==== Fonction de création de fichiers ====

filenames = []
codes_to_find = []
# Loop pour créer 100 fichiers textes aléatoires contenant des descriptions (nécessaire pour le KNN suivant, mais attention ça crée beaucoup de fichiers)
for i in range(1, 101):
    random_id = random.randint(0, 49999)
    filename = f"element_{i}.txt"
    codes = transform_to_txt(df_cleaned, filename, random_id)
    print(f"Fichier {filename} créé pour l'ID {random_id}")
    filenames.append(filename)
    codes_to_find.append(codes)

#%%
KNN(df_cleaned, filenames, codes_to_find)

#%%
functiontest(df_cleaned)
