# %%
def findFirstOccurrence(string, textSearch):
    index = string.find(textSearch)
    return index


# %%
def suppBetweenIndices(string, startIndex, endIndex):
    if startIndex < 0 or endIndex >= len(string) or startIndex > endIndex:
        return string
    return string[:startIndex] + ' ' + string[endIndex + 1:]


# %%
def findSmallestBalise(string):
    smallestIndexStart = -1
    smallestIndexEnd = -1
    smallestSize = float('inf')

    indexStartBalise = string.find('<')
    while indexStartBalise != -1:
        indexEndBalise = string.find('>', indexStartBalise)
        if indexEndBalise == -1:
            break
        baliseContent = string[indexStartBalise + 1:indexEndBalise].strip()
        baliseSize = len(baliseContent)
        if baliseSize < smallestSize:
            smallestSize = baliseSize
            smallestIndexStart = indexStartBalise
            smallestIndexEnd = indexEndBalise
        indexStartBalise = string.find('<', indexEndBalise + 1)

    return smallestIndexStart, smallestIndexEnd



# %%
def suppEveryBalise(string):
    indexStartBalise, indexEndBalise = findSmallestBalise(string)
    while indexStartBalise != -1:
        string = suppBetweenIndices(string, indexStartBalise, indexEndBalise)
        indexStartBalise, indexEndBalise = findSmallestBalise(string)
    return string
