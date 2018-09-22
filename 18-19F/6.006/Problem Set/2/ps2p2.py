from random import randint

def shadow_art_linear(A):

    B = []

    for i in range(0, len(A)):

        if len(B) < A[i][1]:
            B.extend([0] * (A[i][1] - len(B)))

        for j in range(A[i][0], A[i][1]):
            B[j] += 1

    C = []
    last_index = 0
    last_value = B[0]

    for i in range(1, len(B)):

        if B[i] != last_value:

            C.append(((last_index, i), last_value))

            last_index = i
            last_value = B[i]

    C.append(((last_index, len(B)), last_value))

    return C


def shadow_art_divide(A):

    sorted_lower = sorted(map(lambda p: p[0], A))
    sorted_upper = sorted(map(lambda p: p[1], A))

    number_of_panels = 1
    lower_index = 1
    upper_index = 0
    last_value = sorted_lower[0]

    output = []

    while lower_index < len(sorted_lower) and upper_index < len(sorted_upper):

        if sorted_lower[lower_index] < sorted_upper[upper_index]:
            if last_value != sorted_lower[lower_index]:
                output.append(((last_value, sorted_lower[lower_index]), number_of_panels))
            number_of_panels += 1
            last_value = sorted_lower[lower_index]
            lower_index += 1

        elif sorted_lower[lower_index] > sorted_upper[upper_index]:
            if last_value != sorted_upper[upper_index]:
                output.append(((last_value, sorted_upper[upper_index]), number_of_panels))
            number_of_panels -= 1
            last_value = sorted_upper[upper_index]
            upper_index += 1

        else:
            upper_index += 1
            lower_index += 1

    while lower_index < len(sorted_lower):
        if last_value != sorted_lower[lower_index]:
            output.append(((last_value, sorted_lower[lower_index]), number_of_panels))
        number_of_panels += 1
        last_value = sorted_lower[lower_index]
        lower_index += 1

    while upper_index < len(sorted_upper):
        if last_value != sorted_upper[upper_index]:
            output.append(((last_value, sorted_upper[upper_index]), number_of_panels))
        number_of_panels -= 1
        last_value = sorted_upper[upper_index]
        upper_index += 1

    return output


for i in range(10):

    A = []

    for j in range(randint(1, 10)):

        a = randint(0, 9)
        b = randint(a + 1, 10)

        A.append((a, b))

    print(A)
    print(shadow_art_divide(A))
    print(shadow_art_linear(A))
    print("\n")
