#accepts a sentence and prints the number of letters anddigits in Sentence.

string = input("Please Enter Your String")
digit=0
letter=0
for count in string:
    if count.isdigit():
        digit=digit+1
    elif count.isalpha():
        letter=letter+1
    else:
        pass
print("Number of Letters in Sentence", letter)
print("Number of Digits in Sentence", digit)