class Search:

    def __init__():
        pass

    
    def searchValuesAndReturnKeys(d, searchFor):
        listOfKeys = []
        for k in d.values().items():
            if searchFor in k[1]:
                listOfKeys.append(k[0])
        return listOfKeys

    def searchKeysAndReturnValues(d, searchFor, term_to_avoid = None):

        listOfValues = []
        for k in d.keys():
            if searchFor in k and term_to_avoid not in k:
                listOfValues.append(d[k])
        return listOfValues