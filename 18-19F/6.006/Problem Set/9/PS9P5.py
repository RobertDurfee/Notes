def longest_palindrome(A, i=0, j=None, memo={}):

    if j is None:
        j = len(A) - 1

    if j - i == -1:
        return 0, []
    if j - i == 0:
        return 1, [A[i]]

    if A[i] != A[j]:

        if (i + 1, j) not in memo:
            memo[(i + 1, j)] = longest_palindrome(A, i + 1, j, memo)

        if (i, j - 1) not in memo:
            memo[(i, j - 1)] = longest_palindrome(A, i, j - 1, memo)

        return max(memo[(i + 1, j)], memo[(i, j - 1)], key=lambda x: x[0])

    else:  # A[i] == A[j]

        if (i + 1, j - 1) not in memo:
            memo[(i + 1, j - 1)] = longest_palindrome(A, i + 1, j - 1, memo)

        cnt, mes = memo[(i + 1, j - 1)]

        return cnt + 2, [A[i]] + mes


def decode_message(A):

    return longest_palindrome(A)[1]
