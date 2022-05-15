def inverse(a):
  I=np.mat(np.identity(len(a)))
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
      if a[i,j] == 0:continue
      common_divisor=a[i-1,j]/a[i,j]  
      a[i,:]=common_divisor*a[i,:]-a[i-1,:]
      I[i,:]=common_divisor*I[i,:]-I[i-1,:] 
    column-=1
    stop+=1
  row,column=a.shape
  stop=0
  for j in range(column-1,0,-1):
    for i in range(0,row-1):
      if a[i,j] == 0:continue
      common_divisor=a[i+1,j]/a[i,j]  
      a[i,:]=common_divisor*a[i,:]-a[i+1,:]
      I[i,:]=common_divisor*I[i,:]-I[i+1,:]
    column-=1
    row-=1
  row,column=a.shape
  row_idx=0
  column_idx=0
  print(a)
  print(I)
  while row_idx<=row-1 and column_idx<=column-1:
    divisor=1/a[row_idx,column_idx]
    a[row_idx,:]=a[row_idx,:]*divisor
    I[row_idx,:]=I[row_idx,:]*divisor
    print(I)
    row_idx+=1
    column_idx+=1
  return I