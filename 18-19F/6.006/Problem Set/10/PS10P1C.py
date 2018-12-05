def a_merge(A, i, m, j, n, w):

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


if __name__ == '__main__':

  n = 16
  a = 4

  A = [43, 44, 45, 46, -100, -100, -100, -100, 6, 7, 8, 9, 10, 11, 12, 13]
  print(f'Original: {A}')

  a_merge(A, 0, a, 2 * a, len(A), a)
  print(f'Sorted: {A}')
