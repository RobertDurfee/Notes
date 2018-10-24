import unittest
from PS6P6 import is_solved
from PS6P6 import move
from PS6P6 import solve_ksquare

tests = (
    (
        ((1, 2), (3, 4)),
        0,
    ),
    (
        ((1, -4), (3, -2)),
        1,
    ),
    (
        ((1, -4), (2, -3)),
        2,
    ),
    (
        ((1, 2, 7), (4, 5, 6), (3, -8, 9)),
        3,
    ),
    (
        ((1, 2, 3, 13), (5, 6, 7, 8), (9, 10, 11, 12), (-16, 14, 15, -4)),
        4,
    ),
)


def check(config, min_moves):
    moves = solve_ksquare(config)
    if len(moves) != min_moves:
        return False
    for mv in moves:
        config = move(config, mv)
    return is_solved(config)


class TestSolveKSquare(unittest.TestCase):
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
