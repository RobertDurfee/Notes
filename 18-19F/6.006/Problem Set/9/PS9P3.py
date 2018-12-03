from random import randrange
from math import inf


def max_district(A, d, i=0, j=None, memo={}):

    def value(a, b):
        if sum(A[a:b + 1]) >= (b - a + 1) / 2:
            return 1
        else:
            return 0

    if j is None:
        j = len(A) - 1

    if d == 1:

        if (j - i + 1) % 2 == 0:
            return -inf, []
        else:
            return value(i, j), [(i, j)]

    mx = (-inf, [])

    for k in range(i, j):

        if (i, k, 1) not in memo:
            memo[(i, k, 1)] = max_district(A, 1, i, k)

        if (k + 1, j, d - 1) not in memo:
            memo[(k + 1, j, d - 1)] = max_district(A, d - 1, k + 1, j)

        val1, seq1 = memo[(i, k, 1)]
        val2, seq2 = memo[(k + 1, j, d - 1)]

        vala, seqa = val1 + val2, seq1 + seq2

        mx = max(mx, (vala, seqa), key=lambda x: x[0])

    return mx


d = randrange(2, 10 + 1)
A = [randrange(0, 1 + 1) for i in range(randrange(d * 2 + 1, 4 * d + 1, 2))]

print(f'd = {d}')

for i in range(len(A)):

    print()
    print(A[i:] + A[:i])
    print(max_district(A[i:] + A[:i], d))
