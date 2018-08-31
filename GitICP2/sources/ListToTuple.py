#Problem: Write a python program to take list of numbers as input and to return a tuple of first and last numbers in the list.

values = input("Input some comma seprated numbers : ")
#Spliting it by comma and put it in list
list = values.split(",")
#Make that list as tuple
tuple = tuple(list)
print('List : ',list)
#Print first and last element in tuple
print('Tuple : ',tuple[0],tuple[-1])
