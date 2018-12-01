from math import log2, ceil


def count_avl(n, h, memo={}):

    if h < 0:
        return 0
    elif h == n == 0 or h == n == 1:
        return 1

    sm = 0

    for i in range(n):

        if (i, h - 1) not in memo:
            memo[(i, h - 1)] = count_avl(i, h - 1, memo)

        if (i, h - 2) not in memo:
            memo[(i, h - 2)] = count_avl(i, h - 2, memo)

        if (n - (i + 1), h - 1) not in memo:
            memo[(n - (i + 1), h - 1)] = count_avl(n - (i + 1), h - 1, memo)

        if (n - (i + 1), h - 2) not in memo:
            memo[(n - (i + 1), h - 2)] = count_avl(n - (i + 1), h - 2, memo)

        sm += memo[(i, h - 1)] * memo[(n - (i + 1), h - 1)]
        sm += memo[(i, h - 1)] * memo[(n - (i + 1), h - 2)]
        sm += memo[(i, h - 2)] * memo[(n - (i + 1), h - 1)]

    return sm


for n in range(1, 20):
    sum = 0
    for h in range(ceil(1.44 * log2(n)) + 1):
        sum += count_avl(n, h)
    print(f'count_avl({n}) = {sum}')
