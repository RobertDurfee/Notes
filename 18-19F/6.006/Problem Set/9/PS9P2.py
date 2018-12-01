from random import randint


def best_ships(A, B, i=0, j=0, memo={}):

    # Case: i and j are even at the end.
    #  --- ---
    # ... | i |
    #  --- ---
    # ... | j |
    #  --- ---
    if i == len(A) - 1 and j == len(B) - 1:
        # Either leave or take both
        #  --- ---        --- ---
        # ... | N |      ... | Y |
        #  --- ---   or   --- ---
        # ... | N |      ... | Y |
        #  --- ---        --- ---
        return max((0, []), (A[i] + B[i], [((i, 0), (j, 1))]), key=lambda x: x[0])

    # Case: i is beyond end and j is a single square.
    #  --- ---
    # ... | ? | i
    #  --- ---
    # ... | j |
    #  --- ---
    elif i > j and j == len(A) - 1:
        # Must leave single block
        #  --- ---
        # ... | ? | i
        #  --- ---
        # ... | N |
        #  --- ---
        return 0, []

    # Case: j is beyond end and i is a single square.
    #  --- ---
    # ... | i |
    #  --- ---
    # ... | ? | j
    #  --- ---
    elif i < j and i == len(B) - 1:
        # Must leave single block
        #  --- ---
        # ... | N |
        #  --- ---
        # ... | ? | j
        #  --- ---
        return 0, []

    # Case: i and j are even at beginning
    #  --- ---
    # | i | ...
    #  --- ---
    # | j | ...
    #  --- ---
    if i == j:

        # Case: A[i] + A[i + 1] + x(i + 2, j + 1)
        #  --- --- ---
        # | Y | Y | ...
        #  --- --- ---
        # | N |   | ...
        #  --- --- ---
        if (i + 2, j + 1) not in memo:
            memo[(i + 2, j + 1)] = best_ships(A, B, i + 2, j + 1, memo)

        vala, seqa = memo[(i + 2, j + 1)]
        vala += A[i] + A[i + 1]
        seqa = [((i, 0), (i + 1, 0))] + seqa

        # Case: B[j] + B[j + 1] + x(i + 1, j + 2)
        #  --- --- ---
        # | N |   | ...
        #  --- --- ---
        # | Y | Y | ...
        #  --- --- ---
        if (i + 1, j + 2) not in memo:
            memo[(i + 1, j + 2)] = best_ships(A, B, i + 1, j + 2, memo)

        valb, seqb = memo[(i + 1, j + 2)]
        valb += B[j] + B[j + 1]
        seqb = [((j, 1), (j + 1, 1))] + seqb

        # Case: A[i] + B[j] + x(i + 1, j + 1)
        #  --- ---
        # | Y | ...
        #  --- ---
        # | Y | ...
        #  --- ---
        if (i + 1, j + 1) not in memo:
            memo[(i + 1, j + 1)] = best_ships(A, B, i + 1, j + 1, memo)

        valc, seqc = memo[(i + 1, j + 1)]
        valc += A[i] + B[j]
        seqc = [((i, 0), (j, 1))] + seqc

        # Case: x(i + 1, j + 1)
        #  --- ---
        # | N | ...
        #  --- ---
        # | N | ...
        #  --- ---
        vald, seqd = memo[(i + 1, j + 1)]

        return max((vala, seqa), (valb, seqb), (valc, seqc), (vald, seqd), key=lambda x: x[0])

    # Case: i is past j at beginning
    #      --- ---
    #     | i | ...
    #  --- --- ---
    # | j |   | ...
    #  --- --- ---
    elif i > j:

        # Case: B[j] + B[j + 1] + x(i, j + 2)
        #      --- ---
        #     |   | ...
        #  --- --- ---
        # | Y | Y | ...
        #  --- --- ---
        if (i, j + 2) not in memo:
            memo[(i, j + 2)] = best_ships(A, B, i, j + 2, memo)

        vala, seqa = memo[(i, j + 2)]
        vala += B[j] + B[j + 1]
        seqa = [((j, 1), (j + 1, 1))] + seqa

        # Case: x(i, j + 1)
        #      --- ---
        #     |   | ...
        #  --- --- ---
        # | N |   | ...
        #  --- --- ---
        if (i, j + 1) not in memo:
            memo[(i, j + 1)] = best_ships(A, B, i, j + 1, memo)

        valb, seqb = memo[(i, j + 1)]

        return max((vala, seqa), (valb, seqb), key=lambda x: x[0])

    # Case: j is past i at beginning
    #  --- --- ---
    # | i |   | ...
    #  --- --- ---
    #     | j | ...
    #      --- ---
    elif i < j:

        # Case: A[i] + A[i + 1] + x(i + 2, j)
        #  --- --- ---
        # | Y | Y | ...
        #  --- --- ---
        #     |   | ...
        #      --- ---
        if (i + 2, j) not in memo:
            memo[(i + 2, j)] = best_ships(A, B, i + 2, j, memo)

        vala, seqa = memo[(i + 2, j)]
        vala += A[i] + A[i + 1]
        seqa = [((i, 0), (i + 1, 0))] + seqa

        # Case: x(i + 1, j)
        #  --- --- ---
        # | N |   | ...
        #  --- --- ---
        #     |   | ...
        #      --- ---
        if (i + 1, j) not in memo:
            memo[(i + 1, j)] = best_ships(A, B, i + 1, j, memo)

        valb, seqb = memo[(i + 1, j)]

        return max((vala, seqa), (valb, seqb), key=lambda x: x[0])


n = randint(1, 10)
A = [randint(-10, 10) for i in range(n)]
B = [randint(-10, 10) for i in range(n)]
print(A)
print(B)
print(best_ships(A, B))
