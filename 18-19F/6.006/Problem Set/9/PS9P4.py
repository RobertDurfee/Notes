from math import inf
from random import randint


def max_score(A, i=0, j=None, memo={}):

    if j is None:
        j = len(A) - 1

    if j - i <= -1:
        return 0, []
    elif j - i == 0:
        return 0, [(A[i],)]

    mx = (-inf, None)

    for k in range(i, j):

        # Case x(i, k - 1) x(k + 1, j) for all k from i to j
        if (i, k) not in memo:
            memo[(i, k)] = max_score(A, i, k, memo)
        if (k + 1, j) not in memo:
            memo[(k + 1, j)] = max_score(A, k + 1, j, memo)

        val1, seq1 = memo[(i, k)]
        val2, seq2 = memo[(k + 1, j)]

        vala, seqa = val1 + val2, seq1 + seq2

        # Case x(i + 1, j - 1) + A[i] * A[j]
        if (i + 1, j - 1) not in memo:
            memo[(i + 1, j - 1)] = max_score(A, i + 1, j - 1, memo)

        val1, seq1 = memo[(i + 1, j - 1)]

        valb = val1 + (A[i] * A[j])
        seqb = seq1 + [(A[i], A[j])]

        mx = max(mx, (vala, seqa), (valb, seqb), key=lambda x: x[0])

    return mx


A = tuple([randint(-10, 10) for i in range(randint(1, 20))])

print(A)
print(max_score(A))

