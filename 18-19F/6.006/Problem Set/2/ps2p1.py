def find_maximum(A, i=0, j=None):

    if j is None:
        j = len(A)

    if len(A) == 0:
        return None

    m = A[i]
    index = i

    for k in range(i + 1, j):

        if A[k] > m:

            m = A[k]
            index = k

    return m, index


def longest_zipline_path_starts_at(A, i, j):

    for k in range(1, j - i):

        if A[i + k] > A[i]:
            return k + 1

    return j - i + 1


def longest_zipline_path_ends_at(A, i, j):

    for k in range(1, j - i):

        if A[j - 1 - k] > A[j - 1]:
            return k + 1

    return j - i + 1


def longest_zipline_path_divide(A, i=0, j=None):

    if j is None:
        j = len(A)

    if j == i + 1:
        return 2

    c = (i + j) // 2

    left_max, left_max_index = find_maximum(A, i, c)
    right_max, right_max_index = find_maximum(A, c, j)

    a = longest_zipline_path_divide(A, i, c)
    b = longest_zipline_path_divide(A, c, j)

    if left_max < right_max:
        c = longest_zipline_path_starts_at(A, left_max_index, j)

    else:
        c = longest_zipline_path_ends_at(A, i, right_max_index + 1)

    return max(a, b, c)


test = [0, 1, 2, 3, 9, 5, 6, 7, 8]
print(longest_zipline_path_divide(test))
