import unittest
from PS5P4 import count_anagrams

tests = (
    (
        ('oea', 'cbdd', 'amm', 'phbk', 'bddc', 'ckdza', 'dbdc', 'bcdd', 'xrdoq', 'skpw'),
        6,
    ),
    (
        ('qmc', 'rtao', 'faxnv', 'mahuu', 'xlj', 'iivr', 'wft', 'jknd', 'fbe', 'jnkd', 'ndkj', 'orta', 'qtind', 'atro',
         'xerk', 'nhgqy', 'rato', 'xgqb', 'mbi', 'sdj'),
        9,
    ),
    (
        ('itila', 'vwxpx', 'xnwe', 'wdv', 'pxxvw', 'qqgfs', 'fotl', 'pkbs', 'awyva', 'ayavw', 'cav', 'qlg', 'ywlob',
         'aayvw', 'irye', 'wms', 'ydmei', 'ihgc', 'vnv', 'cmn', 'aob', 'ahwfq', 'wbyol', 'ecudv', 'pfk', 'bolwy',
         'musj', 'yolbw', 'iwnb', 'ijpd'),
        10,
    ),
    (
        ('lqw', 'rtvm', 'gslq', 'mktk', 'rtvm', 'puwc', 'ayg', 'poy', 'tvmr', 'xyn', 'wbng', 'nbgw', 'obacd', 'pyyr',
         'fraxg', 'jqbar', 'gbwn', 'aobdu', 'wcpu', 'vkbcm', 'papfx', 'dubz', 'jqmg', 'vxsu', 'woy', 'upcw', 'hgc',
         'ctw', 'gpe', 'vtnb', 'imv', 'budz', 'ycyyf', 'zdbu', 'xjij', 'ldx', 'kpby', 'zudb', 'tiab', 'uphhi'),
        13,
    ),
    (
        ('cni', 'stvfn', 'jwcqf', 'aeozz', 'qvpkt', 'cvcqp', 'rhiq', 'foj', 'xmyyd', 'acvgh', 'gts', 'gpyxy', 'zlkl',
         'qsw', 'zqb', 'zozae', 'zey', 'gqnk', 'ugamn', 'cdc', 'xjoft', 'kgqn', 'rjc', 'sys', 'jfo', 'xykla', 'ysil',
         'iev', 'nkgq', 'fhqqj', 'qnkg', 'qbz', 'gitz', 'vply', 'ynb', 'hws', 'aozez', 'xebci', 'vqmha', 'mxp', 'kvtv',
         'szh', 'wdc', 'tlr', 'vply', 'hdsc', 'jof', 'ojf', 'nvlk', 'kmvy'),
        16,
    ),
)


def check(words, count):
    return count_anagrams(words) == count


class TestCountAnagrams(unittest.TestCase):
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