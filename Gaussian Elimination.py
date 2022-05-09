# Please check the README.ipyb NoteBook first
# import necessary libs

import numpy as np
from scipy import linalg

# construct the matracies using NumPy mat function
# a reprents the coefficient matrix, b represents the constant terms matrix(Y side)

a =np.mat([
        [1.0, 1.0, -1.0],
        [2.0, -1.0, 1.0],
        [-1.0, 2.0, 2.0],],dtype=float)
b = np.mat([-2, 5, 1],dtype=float).T

print('The Coefficient matrix is')
print(a)
print('The constants terms are')
print(b)

"""First, we will need to check if the diagonal elements in the matrix contain zero. If so, the Gaussian elimination will not work"""

# Since the diagonal elements cannot be zeros:
# The code below will check each diagonal element and compare it to zero, 
# if zero value is detected, the code will generate an error message 
# and print the coordinate of the zero value diagonal element
row,column=a.shape
row_idx=row-1
column_idx=0
while row_idx>=0 and column_idx<=column-1:
  if a[row_idx,column_idx]==0:
    print('diagonal element coordinate',str([row_idx+1,column_idx+1]),'is zero')
  break
  row_idx-=1
  column_idx+=1

"""The code did not generate any error message, so we will be able to apply Gaussian Elimination method to the matrix.<br>
<br>
The codes below excute the procedures of finding common divisor of two equations, multiplying the first equation and then substruct the second equation in order to eliminate a variable. Once this process is done, it will update b, the constant terms accordingly.
"""

row,column=a.shape #getting the dimensions/size of the matrix, and set them as the upbond of row and column iteration
count=1
for j in range(0,column-1):
  for i in range(0,row-1):
    if a[i,j] == 0: #checking if an element is already zero, if so, skip
      continue
    else:
      common_divisor=a[i+1,j]/a[i,j]  #finding the common divisor of two equations
      a[i,j:]=common_divisor*a[i,j:]-a[i+1,j:]  #update the first equation -- multiplying the first equation by the common divisor and substrcut the second equation
      b[i]=common_divisor*b[i]-b[i+1] #update the constant term accordingly
      print('step',count) 
      print(a) #print the current number of step, and the current matrix
      count+=1
  row-=1

"""Printing the new coefficient matrix and the new constant terms"""

print('The new coefficient matrix is')
print(a)
print('The new constants terms are')
print(b)

"""Creating an array 'x' to store the values of variables"""

row,column=a.shape
x=np.mat(np.zeros(row)).T

"""Calculating variables"""

ranks=0 #the current rank of the matrix in interation
col=column-1
while ranks<=row-1:
  lst=[]
  for n in range(0,ranks): # the ranks starts from 0, but the iteration of column starts from the greatest, and becomes smaller as n increases
    lst.append(a[ranks,col-n]*x[-1-n]) #col-n is the current column in the iteration, since n will always be one less than current rank(row), the first non-zero element will be kept untouched.
  x[-1-ranks]=(b[ranks]-sum(lst))/a[ranks,col-ranks]
  ranks+=1

"""Printing the array x containing variables:"""

print(x)

"""Using linear algebra solver in Scipy Module to validate/testify our answers"""

print(linalg.solve(a,b))

"""The result is identical to our answers"""
