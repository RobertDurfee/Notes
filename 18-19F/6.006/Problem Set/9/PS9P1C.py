from math import log2, floor


def choose(n, k, memo={}):

    # n choose k is the same as n choose n - k which might be smaller
    if k > n - k:
        k = n - k

    if k == 0:
        return 1

    if (n - 1, k - 1) not in memo:
        memo[(n - 1, k - 1)] = choose(n - 1, k - 1)

    return n * memo[(n - 1, k - 1)] / k


def count_heaps(n):

    if n == 1 or n == 0:
        return 1

    h = floor(log2(n))

    # Case: Last level is more than half full.
    #        X
    #    X       X
    #  X   X   X   X
    # X X X X X X
    if n - (2 ** h - 1) >= 2 ** (h - 1):
        l = 2 ** h - 1

    # Case: Last level is less than half full.
    #        X
    #    X       X
    #  X   X   X   X
    # X X
    else:  # n - (2 ** h - 1) < 2 ** (h - 1)
        l = n - 2 ** (h - 1)

    r = n - l - 1

    return choose(n - 1, l) * count_heaps(l) * count_heaps(r)


for i in range(1, 13):
    print(f'count_heaps({i}) = {count_heaps(i)}')
