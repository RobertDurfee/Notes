import math


def topological_sort(wadj):

    def adj(u):
        return wadj[u].keys() if u in wadj else []

    parent_backing = {}

    def get_parent(u):
        return parent_backing[u] if u in parent_backing else None

    def set_parent(u, v):
        parent_backing[u] = v

    order = []

    def visit(u):

        for v in adj(u):

            if get_parent(v) is None:

                set_parent(v, u)
                visit(v)

        order.append(u)

    for s in wadj.keys():

        if get_parent(s) is None:

            set_parent(s, s)
            visit(s)

    order.reverse()
    return order


def ss_maximum_path(wadj, s):

    def w(u, v):
        return wadj[u][v] if u in wadj and v in wadj[u] else -math.inf

    def adj(u):
        return wadj[u].keys() if u in wadj else []

    parent_backing = {}

    def set_parent(u, v):
        parent_backing[u] = v

    set_parent(s, s)

    d_backing = {}

    def get_d(u):
        return d_backing[u] if u in d_backing else -math.inf

    def set_d(u, v)
        d_backing[u] = v

    set_d(s, 0)

    def contract(u, v):

        if get_d(v) < get_d(u) + w(u, v):

            set_d(v, get_d(u) + w(u, v))
            set_parent(v, u)

    for u in topological_sort(wadj):

        for v in adj(u):
            contract(u, v)

    return parent_backing


def create_wadj(transformations):

    wadj = {}

    for i in range(len(transformations)):

        for j in range(len(transformations[i][0])):

            v = transformations[i][0][j]
            u = transformations[i][1]
            w = transformations[i][2]

            if u not in wadj:
                wadj[u] = {}

            if v not in wadj:
                wadj[v] = {}

            wadj[u][v] = w

    return wadj


def build_time(ss, transformations, t):

    def w(u, v):
        return wadj[u][v] if u in wadj and v in wadj[u] else -math.inf

    wadj = create_wadj(transformations)

    parents = ss_maximum_path(wadj, t)

    max_path_length = -math.inf

    for s in ss:

        current = s
        parent = parents[current]

        path_length = 0

        while current != parent:

            path_length += w(parent, current)

            current = parent
            parent = parents[current]

        if path_length > max_path_length:
            max_path_length = path_length

    return max_path_length
