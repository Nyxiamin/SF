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
    i = 1
    indexStartBalise = findFirstOccurrence(string, '<')
    indexEndBalise = findFirstOccurrence(string, '>')

    newString = string[indexStartBalise + i:indexEndBalise]
    smallestStartBalise = findFirstOccurrence(newString, '<')
    smallestEndBalise = findFirstOccurrence(newString, '>')

    while ((smallestStartBalise != -1) or (smallestEndBalise != -1)):
        if smallestStartBalise != -1:
            indexStartBalise += smallestStartBalise
            i += 1
        if smallestEndBalise != -1:
            indexEndBalise += smallestEndBalise

        newString = string[indexStartBalise + i:indexEndBalise]
        smallestStartBalise = findFirstOccurrence(newString, '<')
        smallestEndBalise = findFirstOccurrence(newString, '<')

    return indexStartBalise, indexEndBalise


# %%
def suppEveryBalise(string):
    indexStartBalise, indexEndBalise = findSmallestBalise(string)
    while indexStartBalise != -1:
        string = suppBetweenIndices(string, indexStartBalise, indexEndBalise)
        indexStartBalise, indexEndBalise = findSmallestBalise(string)
    return string
