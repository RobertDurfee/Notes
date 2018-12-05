from PS10P1A import k_merge_sort

def four_two_merge_sort(A, i, j):

  k_merge_sort(A, (i + j) // 2, (i + j), 0)
  k_merge_sort(A, 0, (i + j) // 4, (i + j) // 4)


if __name__ == '__main__':

  A = [100, -100, 6, 5, 4, 3, 2, 1, -100, -100, -100, -100, -100, -100, -100, -100]
  print(f'Original: {A}')

  four_two_merge_sort(A, 0, 8)
  print(f'Sorted: {A}')
