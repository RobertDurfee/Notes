def parent_index(i):

    p_index = (i - 1) // 2
    return p_index if 0 < i else i


def left_index(i, n):

    l_index = 2 * i + 1
    return l_index if l_index < n else i


def right_index(i, n):

    r_index = 2 * i + 2
    return r_index if r_index < n else i


def max_heapify_up(A, n, c_index):

    p_index = parent_index(c_index)

    if A[p_index] < A[c_index]:
        A[c_index], A[p_index] = A[p_index], A[c_index]
        max_heapify_up(A, n, p_index)


def max_heapify_down(A, n, p_index):

    l_index, r_index = left_index(p_index, n), right_index(p_index, n)
    c_index = l_index if A[r_index] < A[l_index] else r_index

    if A[p_index] < A[c_index]:
        A[c_index], A[p_index] = A[p_index], A[c_index]
        max_heapify_down(A, n, c_index)


def set_second_largest(A, k):

    l_index, r_index = left_index(0, len(A)), right_index(0, len(A))
    second_max_index = l_index if A[l_index] > A[r_index] else r_index

    if k > A[second_max_index]:

        A[second_max_index] = k
        max_heapify_up(A, len(A), second_max_index)

    else:

        A[second_max_index] = k
        max_heapify_down(A, len(A), second_max_index)

    return A


A = [11, 7, 10, 2, 6, 4, 9]
print(set_second_largest(A, 0))
print(set_second_largest(A, 22))
