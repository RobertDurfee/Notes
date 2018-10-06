import unittest
from PS4P3 import TransactionLog

tests = (
    (
        [(1, 6), (9, 6), (3, 3), (1, 8), (2, 8), (4, 5), (0, 6), (6, 4), (0, 2), (9, 7)],
        [(3, 4)],
        [8]
    ),
    (
        [(8, 0), (6, 8), (9, 3), (3, 0), (7, 3), (9, 4), (3, 0), (2, 8), (7, 0), (8, 0)],
        [(3, 3), (0, 3), (4, 5), (4, 4), (3, 6)],
        [0, 8, 0, 0, 8]
    ),
    (
        [(19, 1), (2, 14), (1, 17), (9, 18), (4, 10), (2, 9), (8, 5), (6, 13), (18, 8), (6, 10), (11, 11), (4, 7),
         (10, 16), (8, 5), (13, 15), (11, 3), (4, 15), (16, 0), (11, 12), (0, 14)],
        [(1, 3), (7, 7), (6, 12), (7, 8), (4, 6)],
        [40, 0, 93, 10, 55]
    ),
    (
        [(11, 8), (18, 11), (10, 5), (17, 19), (8, 11), (8, 8), (10, 9), (6, 3), (6, 7), (6, 13), (1, 7), (9, 8),
         (0, 6), (8, 19), (19, 6), (9, 9), (10, 5), (12, 9), (9, 11), (9, 19)],
        [(3, 8), (0, 1), (6, 7), (5, 14), (2, 2), (7, 14), (2, 7), (7, 12), (3, 5), (9, 11)],
        [61, 13, 23, 144, 0, 121, 23, 121, 0, 74]
    ),
    (
        [(9, 5), (6, 2), (19, 2), (10, 5), (1, 16), (11, 14), (5, 0), (12, 0), (7, 11), (4, 17), (8, 2), (16, 4),
         (15, 13), (16, 15), (15, 7), (18, 4), (8, 16), (9, 2), (2, 11), (19, 8)],
        [(2, 7), (1, 3), (9, 14), (4, 13), (0, 7), (5, 8), (9, 12), (0, 3), (7, 8), (9, 12), (8, 10), (7, 16), (4, 6),
         (1, 7), (0, 9), (2, 5), (1, 8), (4, 9), (6, 11), (1, 6)],
        [41, 27, 26, 74, 57, 31, 26, 27, 29, 26, 30, 94, 19, 57, 82, 28, 75, 55, 57, 46]
    ),
)


class Transaction:
    def __init__(self, t, d):
        self.key, self.d = t, d


def check(transactions, queries, answers):
    log = TransactionLog()
    for t, d in transactions:
        x = Transaction(t, d)
        log.add_transaction(x)
    for i in range(len(queries)):
        t1, t2 = queries[i]
        ans = log.interval_revenue(t1, t2)
        print('Expected: ' + str(answers[i]) + ' Actual: ' + str(ans))
        if ans != answers[i]:
            return False
    return True


class TestClosestPair(unittest.TestCase):
    def test_01(self):
        self.assertTrue(check(tests[0][0], tests[0][1], tests[0][2]))

    def test_02(self):
        self.assertTrue(check(tests[1][0], tests[1][1], tests[1][2]))

    def test_03(self):
        self.assertTrue(check(tests[2][0], tests[2][1], tests[2][2]))

    def test_04(self):
        self.assertTrue(check(tests[3][0], tests[3][1], tests[3][2]))

    def test_05(self):
        self.assertTrue(check(tests[4][0], tests[4][1], tests[4][2]))


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
