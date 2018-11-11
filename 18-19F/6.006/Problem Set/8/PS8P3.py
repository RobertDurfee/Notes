def max_score_seq(A):

    global counter
    counter += 1

    if len(A) == 2:

        score = A[0] * A[1]

        if score > 0:
            return score, [(A[0], A[1])]
        else:
            return 0, []

    else:

        max_score = 0
        max_seq = []

        for i in range(1, len(A)):

            score_a, seq_a = max_score_seq(A[:i])
            score_b, seq_b = max_score_seq(A[i:])

            if score_a + score_b >= max_score:
                max_score = score_a + score_b
                max_seq = seq_a + seq_b

        return max_score, max_seq


def max_score_seq_memoized(A, cache):

    global counter
    counter += 1

    if len(A) == 2:

        score = A[0] * A[1]

        if score > 0:

            print(f'A: {A}')
            return score, [(A[0], A[1])]

        else:

            print(f'A: {A}')
            return 0, []

    else:

        max_score = 0
        max_seq = []

        for i in range(1, len(A)):

            if A[:i] in cache:
                score_a, seq_a = cache[A[:i]]
            else:
                score_a, seq_a = max_score_seq_memoized(A[:i], cache)
                cache[A[:i]] = (score_a, seq_a)

            if A[i:] in cache:
                score_b, seq_b = cache[A[i:]]
            else:
                score_b, seq_b = max_score_seq_memoized(A[i:], cache)
                cache[A[i:]] = (score_b, seq_b)

            if score_a + score_b >= max_score:
                max_score = score_a + score_b
                max_seq = seq_a + seq_b

        print(f'A: {A}')
        return max_score, max_seq


counter = 0
A = (5, -3, -5, 1, 2, 9, -4)

print(max_score_seq(A))
print(counter)

counter = 0
print(max_score_seq_memoized(A, {}))
print(counter)
