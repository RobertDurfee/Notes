def count_bst(n, memo={}):

    if n == 0 or n == 1:
        return 1

    sm = 0

    for i in range(n):

        if i not in memo:
            memo[i] = count_bst(i, memo)

        if n - (i + 1) not in memo:
            memo[n - (i + 1)] = count_bst(n - (i + 1), memo)

        sm += memo[i] * memo[n - (i + 1)]

    return sm


for i in range(10):
    print(f'catalan({i}) = {count_bst(i)}')
