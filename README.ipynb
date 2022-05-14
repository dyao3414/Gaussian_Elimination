# Gaussian Elimination Implementation with NumPy
## Mathematical Concept of Gaussian Elimination
### Introduction

Guassian Jordan Elimination is an algorithm in mathematical linear algebra that can convert a matrix into a row echelon matrix. Gaussian elimination can be used to solve systems of linear equations, find the rank of matrices, and find the inverse of an invertible square matrix.
The concept of Gaussian Elimination is to add, substruct, or multiply one equation to make the coefficient of a variable matches itself in another equation in order to eliminate the variable and to continue this process until only one variable is left. Once this final variable is determined, its value is substituted back into the other equations in order to evaluate the remaining unknowns. This method, characterized by step by step elimination of the variables, is called Gaussian elimination.
However, the Gaussian Elimination method has its limitation, the most common situation that Gaussian method is not suitable is diagonal elements in the matrix contain zero.
<br>
<br>
For example, we have a 3 x 3 matrix shown as below<br>
$x_{1}$+$x_{2}$-$x_{3}$=-2   **($l_{1}$)**<br>
$2x_{1}$-$x_{2}$+$x_{3}$=5   ($l_{2}$)<br>
$-x_{1}$+$2x_{2}$+$2x_{3}$=-2 ($l_{3}$)<br>
<br>
<br>
### The Traditional Method
<br>

The traditional algebratic method usually rewrites one of the equations as an equation for an unknown variable, and then use that equation to replace the particular variable in another equation, and so on so forth until there is only one unknown variable left in the matrix and thus becomes solvable.
For example, the (L1) in the matrix above could be written as<br>
$x_{1}$=-2-$x_{2}$+$x_{3}$ (L1)<br>
Then use the right side of the equation to replace the $x_{1}$ in (L2) <BR>
2(-2-$x_{2}$+$x_{3}$)-$x_{2}$+$x_{3}$=5 (L2) <br>
After simplification of (L2), the same procedure will be used on (L2) and make it an equation for variable 
$x_{2}$
to replace the x2 in (L3) and finally solve for $x_{3}$ in (L3).<br>
<br>
This method usually takes too many substituions and requires simplification after each replacement, thus causing unneccessary computations and taking longer time, espcially when there are more equations in the matrix. In contrast, Gaussian Elimination is a better way of solving linear algebra functions like this, Gaussian method can also be used in computers to solve thousands of equations and unknowns variables in a fast speed.
<br>
<br>
### Gaussian Jordan Elimination Method
<br>
As we disscussed in the introduction, the nature of Guassian Jordan Elimination is to add or substruct vectors with common elements in order to convert a matrix into a reduced row echelon form (RREF)
matrix of the previous linear equations <br>

$x_{1}$+$x_{2}$-$x_{3}$=-2   ($l_{1}$)<br>
$2x_{1}$-$x_{2}$+$x_{3}$=5   ($l_{2}$)<br>
-$x_{1}$+$2x_{2}$+$2x_{3}$=-2 ($l_{3}$)<br>
# =


[1.0, 1.0, -1.0 | -2],   ($l_{1}$)<br>
[2.0, -1.0, 1.0 | 5],    ($l_{2}$)<br>
[-1.0, 2.0, 2.0 | 1]    ($l_{3}$)<br>
<br>
<br>
### Limitations
There are limitations when to use Gaussian Elimination, one prerequesite is the matrix needs to be a square matrix, another prerequesite is that the diagnal elements in the matrix cannot be zero, so we will check this to make sure the method is applicable in the particular matrix.
Then we will start finding the common divisor/factor of the first elements on each row.
We first multuply (L1) by 2 to make it's first element match (L2), then we have (L1) as <br>
**[2.0, 2.0, -2.0 | -4], ($l_{1}$)**<br>
using the new (L1) to substruct (L2), we have:<br>
<br>
[2.0, 2.0, -2.0 | -4] - [2.0, -1.0, 1.0 | 5]<br>
**=[0, 3.0, -3.0 | -9]** **($l_{1}$**)<br>
<br>
We then multuply (L2) by -0.5 to make it's first element match (L3), then we have (L2) as: <br>
**[-1.0, 0.5, -0.5 | -2.5], ($l_{2}$)**<br>
<br>
using the new (L2) to substruct (L3), we have:<br>
<br>
[-1.0, 0.5, -0.5 | -2.5]-[-1.0, 2.0, 2.0 | 1]<br>
**=[0, -1.5, -2.5 | -1.5]** **($l_{2}$**)<br>
<br>
Now, we have finished processing the first elements in this matrix, and the matrix now looks like this:<br>
<BR>
**[ 0.   3.   -3. ]** <br>
**[0.  -1.5 -2.5]**<br>
**[-1.   2.   2. ]** <br>
<br>
As we can see, there is only one first element left in the last equation and the else have been eliminated, and now we will do the same thing to the second elements, but this time we will leave one more elements in the second equation. Once it's done, the matrix looks like:<br>
<BR>

**[ 0,   0,  4 | 8]** <br>
**[ 0,-1.5,-2.5 | -3.5]**<br>
**[-1, 2, 2 | 1]**<br>
<br>
Now, the value of $x_{3}$ becomes intutive which is 8/4 = 2, once $x_{3}$ is known, we can put the value in the second equation to solve for $x_{2}$, and thus the $x_{1}$ in (L3) is also solvable. <BR>
<br>
Thus the result for $x_{1}$, $x_{2}$, $x_{3}$ are respectively **[ 1 ] , [ -1 ] , [ 2 ]**

## Python Implementation of Gaussian Elimination
```python
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

```
