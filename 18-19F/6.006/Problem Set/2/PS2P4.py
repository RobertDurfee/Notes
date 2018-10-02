import math

def squared_distance(p, q):

    (px, py), (qx, qy) = p, q

    return (px - qx) ** 2 + (py - qy) ** 2


def find_V(P, i, j, x_star, delta):

    V = []

    for k in range(i, j):

        if P[x_star][0] - delta <= P[k][0] <= P[x_star][0] + delta:
            V.append(P[k])

    return V


def closest_pair(P):

    P = sorted(P, key=lambda p: p[0])  # sort by x-component O(n log n)
    P = sorted(P, key=lambda p: p[1])  # sort by y-component O(n log n)

    return closest_pair_distance_recursive(P, 0, len(P))  # divide conquer O(n log n)


def closest_pair_distance_recursive(P, i, j):

    if j == i + 1:
        return math.inf

    x_star = (i + j) // 2

    delta = min(closest_pair_distance_recursive(P, i, x_star),
                closest_pair_distance_recursive(P, x_star, j))

    V = find_V(P, i, j, x_star, delta)

    for k in range(0, len(V) - 1):

        for m in range(k + 1, len(V)):  # this actually 'constant' as it will always run < 7x

            if abs(V[k][1] - V[m][1]) > math.sqrt(delta):
                break

            delta = min(delta, squared_distance(V[k], V[m]))

    return delta
