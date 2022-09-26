# import necessary libs
import numpy as np
from scipy import linalg

# construct the matracies using NumPy mat function
# a represents the coefficient matrix, b represents the constant terms matrix(Y side)

a =np.mat([
        [1.0, 1.0, -1.0],
        [2.0, -1.0, 1.0],
        [-1.0, 2.0, 2.0],],dtype=float)
b = np.mat([-2, 5, 1],dtype=float).T



# 1.Check if the matrix is a sqaure matrix, if so generate an error message
# 2.Since the diagonal elements cannot be zeros:
# The code below will check each diagonal element and compare it to zero, 
# if zero value is detected, the code will generate an error message 
# and print the coordinate of the zero value diagonal element
row,column=a.shape
if row != column:
  print('error, matrix is not a square matrix')
row_idx=0
column_idx=0
while row_idx<=row-1 and column_idx<=column-1:
  if a[row_idx,column_idx]==0:
    print('diagonal element coordinate',str([row_idx+1,column_idx+1]),'is zero')
    break
  row_idx+=1
  column_idx+=1





## Row Reduction ##


row,column=a.shape #getting the dimensions/size of the matrix, and set them as the upbond of row and column iteration
stop=0 #since we will be interating from the bottom, set a stop variable that stop the row elimination
for j in range(0,column-1):
  for i in range(row-1,0,-1):
    common_divisor=a[i-1,j]/a[i,j]  #finding the common divisor of two equations
    a[i,:]=common_divisor*a[i,:]-a[i-1,:]  #update the equation -- multiplying the equation by the common divisor and substruct the equation by the other one to eliminiate an element
    b[i]=common_divisor*b[i]-b[i-1] #update the constant term accordingly
  column-=1
  stop+=1


## Solve the Row Reduced Echelon Matrix ##
x=np.mat(np.zeros(row)).T #creating a vector x to store the variables


# now a is in reduced row echelon form


row,column=a.shape #reassign row, column
ranks=row-1 #the current rank of the matrix in interation
while ranks>=0:
  lst=[]
  for n in range(row-1,ranks,-1): # n is the current column in the interation, the first round of loop will be skipped since row-1=ranks
    lst.append(a[ranks,n]*x[n]) #col-n is the current column in the iteration, since n will always be one less than current rank(row), the first non-zero element will be kept untouched.
  x[ranks]=(b[ranks]-sum(lst))/a[ranks,ranks] #assign value to x
  ranks-=1



## Encapsulate chunks of code into a function ##
def gauss(a,b):
  
  row,column=a.shape
  if row != column:
    print('error, matrix is not a square matrix')
    return
  row_idx=0
  column_idx=0
  while row_idx<=row-1 and column_idx<=column-1:
    if a[row_idx,column_idx]==0:
      print('diagonal element coordinate',str([row_idx+1,column_idx+1]),'is zero')
      return
    row_idx+=1
    column_idx+=1
  stop=0 
  for j in range(0,column-1):
    for i in range(row-1,stop,-1):
      common_divisor=a[i-1,j]/a[i,j]  
      a[i,:]=common_divisor*a[i,:]-a[i-1,:]
      b[i]=common_divisor*b[i]-b[i-1] 
    column-=1
    stop+=1
  row,column=a.shape 
  x=np.mat(np.zeros(row)).T
  ranks=row-1 
  while ranks>=0:
    lst=[]
    for n in range(row-1,ranks,-1): 
      lst.append(a[ranks,n]*x[n]) 
    x[ranks]=(b[ranks]-sum(lst))/a[ranks,ranks] 
    ranks-=1
  return x
