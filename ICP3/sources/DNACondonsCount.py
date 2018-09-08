import csv
from collections import Counter

#initialisations
myDic={}
DNAList=[]

#Taking Input from User
codonInput = input("Please enter Sequence of DNA codon")
#splitting the input sequence into 3 character long sub strings
DNAList=[codonInput[i:i+3] for i in range(0, len(codonInput), 3)]

#Opening tsv and reading from it
with open("codon.tsv") as fd:
    reader = csv.reader(fd, delimiter='\t')
    for row in reader:
    #If present in row
     if row[0] in DNAList:
        if row[1] in myDic:
            myDic[row[1]] = myDic[row[1]] + 1 #Incrementing it
        else:
            myDic[row[1]] = 1
            #If occuring for first time just keeping it as one





print(DNAList)

print(myDic)






