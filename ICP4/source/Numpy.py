#Problem Statement: Using NumPy create random vector of size 15 having only Integers
#  in the range 0-20. Write a program to find the most frequent item/value in the vector list.
import numpy as np

#Generated 1D array of 15 random integers with in range if 1 to 20
randomInt=np.random.random_integers(0,20,15)
print(randomInt)

#bincount counts number of occurrences of each value in array of non-negative ints.
counts = np.bincount(randomInt)
# argmax() returns which array element has maximum count
print (np.argmax(counts))