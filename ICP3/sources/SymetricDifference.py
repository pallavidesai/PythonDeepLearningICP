#Intialising set 1
Set1 = set()
#Taking user input of number of values in set
Iterate_Set1 = int(input("how many value you want in a Set: "))
#get set elements from user
for i in range(0, Iterate_Set1):
    InputSet1 = input("enter your choice Iterate_Set1:")
    Set1.update(InputSet1)
print(Set1)

#Intialising set 2
Set2 = set()
#Taking user input of number of values in set
Iterate_Set2 = int(input("how many value you want in a Set: "))
#get set elements from user
for ic in range(0, Iterate_Set2):
    InputSet2 = input("enter your choice Iterate_Set1:")
    Set2.update(InputSet2)
print(Set2)

#symmetric difference between two sets.
print(Set1^Set2)
