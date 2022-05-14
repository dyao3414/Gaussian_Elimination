# Please check the README.ipyb NoteBook first
# import necessary libs

import numpy as np
from scipy import linalg

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
