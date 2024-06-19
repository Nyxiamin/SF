import pandas as pd
import random as rd
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

def create_random_test_sample(data, n):
    l_random = rd.sample(range(len(data)), n)
    random_test_sample = dict()
    for i in l_random:
        transform_to_txt(data, "element_" + str(i) + ".txt", i)
        key = "element_" + str(i) + ".txt"
        value = lire_fichier_txt("element_" + str(i) + ".txt")
        random_test_sample[key] = value
        
    return random_test_sample
                
def main():
    df = pd.read_csv("EFREI - LIPSTIP - 50k elements EPO.csv", nrows=100 )
    test_sample = create_random_test_sample(df, 5)
    for key, value in test_sample.items():
        print(key, value[:10])

    
if __name__ == "__main__":
    main()