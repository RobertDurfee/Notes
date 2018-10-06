import unittest
from PS3P3 import proximate_sort

tests = (
    (
        (-2, 0, -5, -3, 4, 1, 7, 10, 12, 13),
        3,
    ),
    (
        (0, 11, 14, 15, 3, 6, 10, 24, 28, 17, 20, 35, 32, 42, 41, 37, 52, 50, 48, 45),
        3,
    ),
    (
        (-82, -90, -100, -34, -63, -76, -19, -80, -52, -5, 29, 31, 12, 37, 7, 17, 28, 75, 81, 46, 61, 175, 96, 113, 131,
         195, 139, 156, 182, 198),
        5,
    ),
    (
        (
        -158, -137, -200, 15, -9, -160, -37, -108, 47, -80, -141, -155, -68, 79, 292, 159, 125, 383, 205, 117, 193, 233,
        267, 325, 483, 337, 422, 392, 413, 552, 475, 473, 348, 514, 486, 485, 445, 570, 600, 639),
        8,
    ),
    (
        (-214, -250, -249, -170, -207, -236, -222, -51, -224, -114, -12, -8, -152, -17, -33, -95, -188, 60, -75, -57,
         -133, -116, 44, 78, 7, 32, -45, 24, 202, 164, 214, 100, 222, 257, 144, 92, 283, 169, 86, 107, 119, 284, 133,
         248, 269, 185, 240, 299, 308, 317),
        10,
    )
)


def check(A, k):
    B = proximate_sort(A, k)
    if len(A) != len(B):
        return False
    A = sorted(A)
    for i in range(len(A)):
        if B[i] != A[i]:
            return False
    return True


class TestClosestPair(unittest.TestCase):
    def test_01(self):
        self.assertTrue(check(tests[0][0], tests[0][1]))

    def test_02(self):
        self.assertTrue(check(tests[1][0], tests[1][1]))

    def test_03(self):
        self.assertTrue(check(tests[2][0], tests[2][1]))

    def test_04(self):
        self.assertTrue(check(tests[3][0], tests[3][1]))

    def test_05(self):
        self.assertTrue(check(tests[4][0], tests[4][1]))


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
