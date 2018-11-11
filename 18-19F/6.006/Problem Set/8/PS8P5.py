from math import inf


def relax(w, d, parent, u, v):
    if d[v] > d[u] + w[(u, v)]:
        d[v] = d[u] + w[(u, v)]
        parent[v] = u


class Item:
    def __init__(self, label, key):
        self.label, self.key = label, key


class PriorityQueue:                      # Binary Heap Implementation
    def __init__(self):                   # stores keys with unique labels
        self.A = []
        self.label2idx = {}

    def min_heapify_up(self, c):
        if c == 0: return
        p = (c - 1) // 2
        if self.A[p].key > self.A[c].key:
            self.A[c], self.A[p] = self.A[p], self.A[c]
            self.label2idx[self.A[c].label] = c
            self.label2idx[self.A[p].label] = p
            self.min_heapify_up(p)

    def min_heapify_down(self, p):
        if p >= len(self.A): return
        l = 2 * p + 1
        r = 2 * p + 2
        if l >= len(self.A): l = p
        if r >= len(self.A): r = p
        c = l if self.A[r].key > self.A[l].key else r
        if self.A[p].key > self.A[c].key:
            self.A[c], self.A[p] = self.A[p], self.A[c]
            self.label2idx[self.A[c].label] = c
            self.label2idx[self.A[p].label] = p
            self.min_heapify_down(c)

    def insert(self, label, key):         # insert labeled key
        self.A.append(Item(label, key))
        idx = len(self.A) - 1
        self.label2idx[self.A[idx].label] = idx
        self.min_heapify_up(idx)

    def find_min(self):                   # return minimum key
        return self.A[0].key

    def extract_min(self):                # remove a label with minimum key
        self.A[0], self.A[-1] = self.A[-1], self.A[0]
        self.label2idx[self.A[0].label] = 0
        del self.label2idx[self.A[-1].label]
        min_label = self.A.pop().label
        self.min_heapify_down(0)
        return min_label

    def decrease_key(self, label, key):   # decrease key of a given label
        if label in self.label2idx:
            idx = self.label2idx[label]
            if key < self.A[idx].key:
                self.A[idx].key = key
                self.min_heapify_up(idx)


def reverse_adj(adj):

    reversed_adj = [[] for _ in range(len(adj))]

    for i in range(len(adj)):
        for j in range(len(adj[i])):
            reversed_adj[adj[i][j]].append(i)

    return reversed_adj


def bidirectional_dijkstra(forward_adj, forward_w, s, t):

    # Forward Initialization

    forward_parent = [None] * len(forward_adj)
    forward_parent[s] = s

    forward_distance = [inf] * len(forward_adj)
    forward_distance[s] = 0

    forward_queue = PriorityQueue()
    for u in range(len(forward_adj)):
        forward_queue.insert(u, forward_distance[u])

    def forward_relax(u, v):

        if forward_distance[v] > forward_distance[u] + forward_w[(u, v)]:

            forward_distance[v] = forward_distance[u] + forward_w[(u, v)]
            forward_parent[v] = u

            forward_queue.decrease_key(v, forward_distance[v])
            D_queue.decrease_key(v, forward_distance[v] + backward_distance[v])

    # Backward Initialization

    backward_adj = forward_adj

    backward_parent = [None] * len(backward_adj)
    backward_parent[t] = t

    backward_distance = [inf] * len(backward_adj)
    backward_distance[t] = 0

    backward_queue = PriorityQueue()
    for u in range(len(backward_adj)):
        backward_queue.insert(u, backward_distance[u])

    def backward_w(u, v):
        return forward_w[(v, u)]

    def backward_relax(u, v):

        if backward_distance[v] > backward_distance[u] + backward_w(u, v):

            backward_distance[v] = backward_distance[u] + backward_w(u, v)
            backward_parent[v] = u

            backward_queue.decrease_key(v, backward_distance[v])
            D_queue.decrease_key(v, forward_distance[v] + backward_distance[v])

    # Joint Initialization

    D_queue = PriorityQueue()
    for v in range(len(forward_adj)):
        D_queue.insert(v, forward_distance[v] + backward_distance[v])

    # Dijkstra

    while forward_queue.find_min() <= (D_queue.find_min() / 2) \
            and backward_queue.find_min() <= (D_queue.find_min() / 2):

        # Forward Dijkstra
        if forward_queue.find_min() < backward_queue.find_min():

            u = forward_queue.extract_min()

            for v in forward_adj[u]:
                forward_relax(u, v)

        # Backward Dijkstra
        else:

            u = backward_queue.extract_min()

            for v in backward_adj[u]:
                backward_relax(u, v)

    # Calculate Forward Path [s, v*]

    v_star = D_queue.extract_min()

    forward_path = []

    current = v_star
    parent = forward_parent[current]

    forward_path.append(current)

    while current != parent:

        current = parent
        parent = forward_parent[current]

        forward_path.append(current)

    forward_path.reverse()

    # Calculate Backward Path [v*, t]

    backward_path = []

    current = v_star
    parent = backward_parent[current]

    backward_path.append(current)

    while current != parent:

        current = parent
        parent = backward_parent[current]

        backward_path.append(current)

    # Combine Paths for [s, t]

    return forward_path + backward_path[1:]
