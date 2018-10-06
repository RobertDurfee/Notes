def parent_index(i):

    p_index = (i - 1) // 2
    return p_index if 0 < i else i


def left_index(i, n):

    l_index = 2 * i + 1
    return l_index if l_index < n else i


def right_index(i, n):

    r_index = 2 * i + 2
    return r_index if r_index < n else i


def min_heapify_up(A, n, c_index):

    p_index = parent_index(c_index)

    if A[p_index] > A[c_index]:
        A[c_index], A[p_index] = A[p_index], A[c_index]
        min_heapify_up(A, n, p_index)


def min_heapify_down(A, n, p_index):

    l_index, r_index = left_index(p_index, n), right_index(p_index, n)
    c_index = l_index if A[r_index] > A[l_index] else r_index

    if A[p_index] > A[c_index]:
        A[c_index], A[p_index] = A[p_index], A[c_index]
        min_heapify_down(A, n, c_index)


class Heap:

    def __init__(self, A):  # O(k log k)

        self.k = len(A)

        self.arr = [None] * self.k  # O(k)

        for i in range(self.k):  # O(k log k)
            self.arr[i] = A[i]
            min_heapify_up(self.arr, i + 1, i)  # O(log k)

    def swap_top(self, x):

        x, self.arr[0] = self.arr[0], x
        min_heapify_down(self.arr, self.k, 0)  # O(log k)

        return x

    def pop(self):  # O(log k)

        self.arr[0], self.arr[self.k - 1] = self.arr[self.k - 1], self.arr[0]
        self.k -= 1
        min_heapify_down(self.arr, self.k, 0)  # O(log k)

        return self.arr[self.k]


def proximate_sort(A, k):

    k_heap = Heap(A[:k + 1])

    A_srted = []

    for i in range(k + 1, len(A)):
        A_srted.append(k_heap.swap_top(A[i]))

    for i in range(k + 1):
        A_srted.append(k_heap.pop())

    return A_srted


#A = (0, 11, 14, 15, 3, 6, 10, 24, 28, 17, 20, 35, 32, 42, 41, 37, 52, 50, 48, 45)
#k = 4

#print(proximate_sort(A, k))
