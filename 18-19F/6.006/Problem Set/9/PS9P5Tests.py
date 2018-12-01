import unittest
from PS9P5 import decode_message

tests = (
    (
        'vmtqibtyuhutfpftxgrehzzilog',  # input string
        5,                              # length of hidden message
    ),
    (
        'gciaaciihysgasgonisktukewlwedrvvkpsvqeivppnjqbomebmsepymjamvocd',
        12,
    ),
    (
        'cacdccmkdandsjkdeqssihdqrojjkcajhrckohesalassoajrakkehrwlwekiopsinfnlfzaezgriyxfaktgnrfbfippuxdvvbdiuxnnavkizuxca',
        21,
    ),
    (
        'tdretoeuhpebfnlfgrelohppsfixiullnsllruafrfeeiuamlnhellygawpnesoitnlmehcitmlhausssacwjblvcvkcqqeqavmaqosojewbqwxajkyldyqldoaoczkexoxcryaqcsixkdxikqdjsqabbhojayebxalxbukxocd',
        32,
    ),
    (
        'ttkwhxmbvefmfimccituxwrfxstckkfquhwmuesktbubtiohnontwhubxuexifucimivnctacvhlibmfwctxwibwmlbmclbbwkfebtxobwrbihteyoutmmrxnktaimihebfvuonfmwefvkickexbfrytpamgegaraspyrqoseyoazveloaanrpjolsqqezmzddanrrujoslyetyypirrwonyqtzenbollniywzdlorpaalpppnppdpinsnpfdaehzqqtproznlzodrpnjozqopitozdysaosqeyuqjtodsronjoijsrofsprehljtn',
        58,
    ),
)

def check_message(A, message):
    i, j = 0, 0
    palindrome = message[:-1] + message[::-1]
    while (i < len(A)) and (j < len(palindrome)):
        if A[i] == palindrome[j]:
            j = j + 1
        i = i + 1
    return j == len(palindrome)

def check(A, sol):
    message = decode_message(A)
    return (sol == len(message)) and check_message(A, message)

class TestCases(unittest.TestCase):
    #def test_01(self):
    #    self.assertTrue(check(tests[0][0], tests[0][1]))

    #def test_02(self):
    #    self.assertTrue(check(tests[1][0], tests[1][1]))

    #def test_03(self):
    #    self.assertTrue(check(tests[2][0], tests[2][1]))

    #def test_04(self):
    #    self.assertTrue(check(tests[3][0], tests[3][1]))
       
    def test_05(self):
        self.assertTrue(check(tests[4][0], tests[4][1]))

if __name__ == '__main__':
   res = unittest.main(verbosity = 3, exit = False)
