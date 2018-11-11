from math import inf


def max_score(track, velocity):

    global counter
    counter += 1

    if len(track) == 0:
        return 0

    # Decelerate
    score_dec = -1 + max_score(track[1:], max(velocity - 1, 1))

    # Accelerate
    score_acc = -1 + max_score(track[1:], min(velocity + 1, 40))

    # Jump
    score_jump = -inf
    if track[0] is not None:

        # Black Ramp
        if track[0]:

            if velocity <= len(track):
                score_jump = (3 * velocity) + max_score(track[velocity:], velocity)
            else:
                score_jump = 0

        # White Ramp
        else:

            if velocity <= len(track):
                score_jump = velocity + max_score(track[velocity:], velocity)
            else:
                score_jump = 0

    return max(score_dec, score_acc, score_jump)


def max_score_memoized(track, velocity, cache):

    global counter
    counter += 1

    if len(track) == 0:

        print(f'track: {track}, velocity: {velocity}')
        return 0

    # Decelerate
    new_velocity = max(velocity - 1, 1)

    if (track[1:], new_velocity) not in cache:
        cache[(track[1:], new_velocity)] = max_score_memoized(track[1:], new_velocity, cache)

    score_dec = -1 + cache[(track[1:], new_velocity)]

    # Accelerate
    new_velocity = min(velocity + 1, 40)

    if (track[1:], new_velocity) not in cache:
        cache[(track[1:], new_velocity)] = max_score_memoized(track[1:], new_velocity, cache)

    score_acc = -1 + cache[(track[1:], new_velocity)]

    # Jump
    score_jump = -inf
    if track[0] is not None:

        # Black Ramp
        if track[0]:

            if velocity <= len(track):

                if (track[velocity:], velocity) not in cache:
                    cache[(track[velocity:], velocity)] = max_score_memoized(track[velocity:], velocity, cache)

                score_jump = (3 * velocity) + cache[(track[velocity:], velocity)]

            else:
                score_jump = 0

        # White Ramp
        else:

            if velocity <= len(track):

                if (track[velocity:], velocity) not in cache:
                    cache[(track[velocity:], velocity)] = max_score_memoized(track[velocity:], velocity, cache)

                score_jump = velocity + cache[(track[velocity:], velocity)]

            else:
                score_jump = 0

    print(f'track: {track}, velocity: {velocity}')
    return max(score_dec, score_acc, score_jump)


counter = 0
A = (None, True, True, False, True, False, True, True)

print(max_score(A, velocity=1))
print(counter)

counter = 0
print(max_score_memoized(A, velocity=1, cache={}))
print(counter)
