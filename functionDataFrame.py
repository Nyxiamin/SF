from pathlib import Path
import pandas as pd


def readDataframe():
    file_path = Path("../EFREI_LIPSTIP_50k_elements_EPO_clean.csv")
    if file_path.is_file():
        df = pd.read_csv("../EFREI_LIPSTIP_50k_elements_EPO_clean.csv")
        return df

    file_path = Path("../EFREI - LIPSTIP - 50k elements EPO.csv")
    if file_path.is_file():
        df = pd.read_csv("../EFREI - LIPSTIP - 50k elements EPO.csv")
        columns = ["Numéro d'application", "Date d'application", "Numero de publication", "date de publication", "IPC"]
        df_cleaned = df.drop(columns, axis=1)
        return df_cleaned

    print(
        "Le fichier est introuvable. Vérifier que le fichier 'EFREI - LIPSTIP - 50k elements EPO.csv', est bien dans le dossier parent de ce projet.")
    return


def createdCleanCSV(df):
    file_path = Path("../EFREI_LIPSTIP_50k_elements_EPO_clean.csv")
    if file_path.is_file():
        return
    df.to_csv('../EFREI_LIPSTIP_50k_elements_EPO_clean.csv', sep=',', index=False, encoding='utf-8')
    return
