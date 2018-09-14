def search_sorted_2D_array(A, v, x = 0, y = None):

    if y is None:
        y = len(A) - 1

    s = A[y][x]

    if v == s:
        return (x, y)

    if v > s and x < len(A[y]) - 1:
        return search_sorted_2D_array(A, v, x + 1, y)

    if v < s and y > 0:
        return search_sorted_2D_array(A, v, x, y - 1)

    return None

