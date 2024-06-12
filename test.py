import pandas as pd

# %%
df = pd.read_csv("../EFREI - LIPSTIP - 50k elements EPO.csv")


# %%
def findOccurrences(string, textSearch):
    import re
    indices = [match.start() for match in re.finditer(re.escape(textSearch), string)]
    return indices


# %%
def findFirstOccurrence(string, textSearch):
    index = string.find(textSearch)
    return index


# %%
text = df['description'][0]
indexStartBalise = findFirstOccurrence(text, "<")
indexEndBalise = findFirstOccurrence(text, ">")


# %%
def suppBetweenIndices(string, startIndex, endIndex):
    if startIndex < 0 or endIndex >= len(string) or startIndex > endIndex:
        return string
    return string[:startIndex] + string[endIndex+1:]


# %%
def suppEveryBalise(string):
    indexStartBalise = findFirstOccurrence(string, '<')
    indexEndBalise = findFirstOccurrence(string, '>')
    while indexStartBalise != -1:
        indexsStartBalise = findOccurrences(string[indexStartBalise-1:indexEndBalise], '<')
        indexsEndBalise = findOccurrences(string[indexStartBalise-1:indexEndBalise], '>')
        if(len(indexsStartBalise) != len(indexsEndBalise)):
            indexStartBalise = indexsStartBalise[0]
        string = suppBetweenIndices(string, indexStartBalise, indexEndBalise)
        indexStartBalise = findFirstOccurrence(string, '<')
        indexEndBalise = findFirstOccurrence(string, '>')
    return string


# %%
newText = text[:]
newText = suppBetweenIndices(text, indexStartBalise, indexEndBalise)
print(newText)

# %%
print(suppEveryBalise(text))

# %%
print(text)
