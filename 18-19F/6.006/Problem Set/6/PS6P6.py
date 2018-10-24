from collections import deque


def is_solved(config):

    k = len(config)

    for y in range(k):
        for x in range(k):
            if config[y][x] != x + k*y + 1:
                return False

    return True


def extract(config, mv):

    k = len(config)
    s, i = mv

    if s == "row":
        return list(config[i])
    else:
        return [config[j][i] for j in range(k)]


def replace(config, mv, arr):

    k = len(config)
    s, i = mv

    config = list([list(row) for row in config])

    if s == "row":
        config[i] = arr
    else:
        for j in range(k):
            config[j][i] = arr[j]

    return tuple([tuple(row) for row in config])


def move(config, mv):

    arr = extract(config, mv)
    arr.reverse()
    arr = [-x for x in arr]

    return replace(config, mv, arr)


def reachable(config):

    k = len(config)

    for i in range(k):
        yield (("row", i), move(config, ("row", i)))

    for i in range(k):
        yield (("col", i), move(config, ("col", i)))


def solve_ksquare(config):

    visited = {config: (None, None)}
    q = deque()

    previous_config = config

    while not is_solved(previous_config):

        for next_move, next_config in reachable(previous_config):

            if next_config not in visited:

                visited[next_config] = next_move, previous_config
                q.append(next_config)

        previous_config = q.popleft()

    moves = []
    mv, parent = visited[previous_config]

    while parent is not None:

        moves.append(mv)
        mv, parent = visited[parent]

    moves.reverse()
    return moves
