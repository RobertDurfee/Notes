def k_merge_sort(A, i, j, w):

  if j - i == 1:
    return
  
  m = (i + j) // 2

  k_merge_sort(A, i, m, w)
  k_merge_sort(A, m, j, w)

  merge(A, i, m, m, j, w)

  while i < j:

    A[i], A[w] = A[w], A[i]
    i += 1; w += 1
  

def merge(A, i, m, j, n, w):

  while i < m and j < n:

    if A[i] < A[j]:

      A[w], A[i] = A[i], A[w]
      i += 1
    
    else:

      A[w], A[j] = A[j], A[w]
      j += 1
    
    w += 1
  
  while i < m:

    A[w], A[i] = A[i], A[w]
    w += 1; i += 1
  
  while j < n:

    A[w], A[j] = A[j], A[w]
    w += 1; j += 1


if __name__ == '__main__':

  A = [4, 3, 2, 1, -1, -2, -3, -4]
  print(f'Original: {A}')

  k_merge_sort(A, 0, 4, 4)
  print(f'Sorted: {A}')
