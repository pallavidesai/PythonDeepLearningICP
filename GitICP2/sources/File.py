#Write a python program to count number of words and characters in a file and the print the output
with open('file.txt') as WholeFile:
    #Iterate with each line
    for line in WholeFile:
        #Initialize variables
        words = 0
        characters = 0
        ListOfWords = line.split()
        #Count Words
        words += len(ListOfWords)
        #Count characters
        characters += sum(len(word) for word in ListOfWords)
        print(line,words,characters)


