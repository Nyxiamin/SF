import pandas as pd

def transform_to_txt(data, filename, indice):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data["CPC"][indice] + "\n")
        f.write(data["claim"][indice] + "\n")
        f.write(data["description"][indice] + "\n")

        print("File created successfully")
        
def lire_fichier_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        print(f.read())
    return
        
def main():
    df = pd.read_csv("EFREI_LIPSTIP_50k_elements_EPO_clean.csv", nrows=10 )
    print(df.info())
    print(df.head())
    print(df)
    print((df.isnull().sum() / len(df)) * 100)
    # il n'y a pas de valeur null
    
    transform_to_txt(df, "element_1.txt", 0)
    lire_fichier_txt("element_1.txt")
    
if __name__ == "__main__":
    main()