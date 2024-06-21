import pandas as pd
import ast

def transform_to_txt(data, filename, indice):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data["CPC"][indice] + "\n")
        #f.write(data["claim"][indice] + "\n")
        f.write(data["description"][indice] + "\n")
        print("File created successfully")
    cpc_codes_str = data['CPC'][indice]
    cpc_codes = ast.literal_eval(cpc_codes_str)
    first_letters = set(code[0] for code in cpc_codes)
    return first_letters
        
def lire_fichier_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        print(f.read())
    return