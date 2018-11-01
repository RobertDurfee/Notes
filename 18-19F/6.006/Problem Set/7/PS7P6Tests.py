import unittest
from PS7P6 import build_time

tests = (
    (
        (['nyxbweti.gte'], [(['nyxbweti.gte'], 'jttjdehm.npg', 45)], 'jttjdehm.npg'),
        45,
    ),
    (
        (['jnkzdwfr.esm', 'qpcvmxpv.vsy'], [(['jnkzdwfr.esm', 'qpcvmxpv.vsy'], 'ttjjyvcg.nue', 48),
                                            (['jnkzdwfr.esm', 'qpcvmxpv.vsy', 'ttjjyvcg.nue'], 'ylglssey.uff', 57)],
         'ylglssey.uff'),
        105,
    ),
    (
        (['difgrsdy.rof', 'shswjyii.ccm', 'idmjprlp.gzv'],
         [(['shswjyii.ccm'], 'mlueipcs.xva', 71), (['idmjprlp.gzv', 'mlueipcs.xva'], 'zvngqdfd.cpw', 68),
          (['difgrsdy.rof', 'zvngqdfd.cpw', 'mlueipcs.xva'], 'jghteehj.qbo', 9),
          (['shswjyii.ccm', 'idmjprlp.gzv', 'mlueipcs.xva'], 'iunqfecm.szx', 87),
          (['difgrsdy.rof', 'idmjprlp.gzv', 'shswjyii.ccm', 'zvngqdfd.cpw'], 'uokidiwg.yyx', 15)], 'uokidiwg.yyx'),
        154,
    ),
    (
        (['ffxblbjt.nte', 'qzhfkvhk.jqs', 'qmvyxorf.ytr', 'eroyowbf.miv'],
         [(['ffxblbjt.nte', 'qmvyxorf.ytr', 'qzhfkvhk.jqs'], 'rwedkusb.fqj', 42),
          (['ffxblbjt.nte', 'eroyowbf.miv', 'qmvyxorf.ytr', 'qzhfkvhk.jqs', 'rwedkusb.fqj'], 'lmrnglzx.ceu', 60),
          (['eroyowbf.miv', 'qzhfkvhk.jqs', 'lmrnglzx.ceu'], 'maazshqx.wej', 86), (
          ['qzhfkvhk.jqs', 'eroyowbf.miv', 'qmvyxorf.ytr', 'ffxblbjt.nte', 'lmrnglzx.ceu', 'maazshqx.wej'],
          'pufhgzmr.jxj', 62), (
          ['ffxblbjt.nte', 'eroyowbf.miv', 'qzhfkvhk.jqs', 'qmvyxorf.ytr', 'lmrnglzx.ceu', 'maazshqx.wej',
           'pufhgzmr.jxj'], 'vbzkxvcr.rtr', 21), (
          ['qzhfkvhk.jqs', 'pufhgzmr.jxj', 'vbzkxvcr.rtr', 'rwedkusb.fqj', 'lmrnglzx.ceu', 'maazshqx.wej'],
          'cngoxumz.ttw', 56), (['qzhfkvhk.jqs', 'ffxblbjt.nte', 'cngoxumz.ttw', 'rwedkusb.fqj'], 'mwhiiqxq.aiv', 2),
          (['ffxblbjt.nte', 'qmvyxorf.ytr', 'qzhfkvhk.jqs', 'rwedkusb.fqj'], 'gwdveiko.mlk', 76), (
          ['ffxblbjt.nte', 'eroyowbf.miv', 'qmvyxorf.ytr', 'lmrnglzx.ceu', 'vbzkxvcr.rtr', 'pufhgzmr.jxj',
           'rwedkusb.fqj', 'gwdveiko.mlk', 'cngoxumz.ttw', 'maazshqx.wej'], 'tmjxhosd.ioi', 23),
          (['qzhfkvhk.jqs', 'ffxblbjt.nte', 'mwhiiqxq.aiv'], 'cdlieiyx.xwf', 6)], 'cdlieiyx.xwf'),
        335,
    ),
    (
        (['ygccxxkx.kuo', 'etnmhxkq.bfw', 'eskxyqee.ber', 'dfayllal.ncc', 'cmwngfan.vvs'],
         [(['ygccxxkx.kuo', 'etnmhxkq.bfw', 'eskxyqee.ber', 'dfayllal.ncc'], 'wmeliwtp.lnm', 44),
          (['cmwngfan.vvs', 'wmeliwtp.lnm'], 'atspobuy.pvz', 15),
          (['ygccxxkx.kuo', 'cmwngfan.vvs', 'wmeliwtp.lnm', 'atspobuy.pvz'], 'xmypofzi.xvp', 81),
          (['eskxyqee.ber', 'atspobuy.pvz'], 'vpwzdoby.uxw', 91), (
          ['dfayllal.ncc', 'etnmhxkq.bfw', 'wmeliwtp.lnm', 'xmypofzi.xvp', 'vpwzdoby.uxw', 'atspobuy.pvz'],
          'wyhtzayx.zya', 42), (['etnmhxkq.bfw', 'atspobuy.pvz', 'wmeliwtp.lnm', 'xmypofzi.xvp'], 'ljfynrlt.zbf', 5),
          (['etnmhxkq.bfw', 'dfayllal.ncc', 'cmwngfan.vvs', 'wyhtzayx.zya', 'xmypofzi.xvp'], 'afmmkfet.jls', 76), (
          ['eskxyqee.ber', 'dfayllal.ncc', 'ygccxxkx.kuo', 'ljfynrlt.zbf', 'afmmkfet.jls', 'vpwzdoby.uxw',
           'wyhtzayx.zya'], 'qrsratpe.wkc', 33), (['eskxyqee.ber', 'xmypofzi.xvp'], 'yxvxabgd.ycj', 85), (
          ['etnmhxkq.bfw', 'dfayllal.ncc', 'cmwngfan.vvs', 'eskxyqee.ber', 'ygccxxkx.kuo', 'atspobuy.pvz'],
          'mxhuccdg.xzz', 46), (
          ['ygccxxkx.kuo', 'etnmhxkq.bfw', 'eskxyqee.ber', 'dfayllal.ncc', 'cmwngfan.vvs', 'wmeliwtp.lnm',
           'ljfynrlt.zbf'], 'ophquhiw.uby', 62), (
          ['ygccxxkx.kuo', 'cmwngfan.vvs', 'etnmhxkq.bfw', 'yxvxabgd.ycj', 'afmmkfet.jls', 'ljfynrlt.zbf',
           'wmeliwtp.lnm'], 'miicppry.qvk', 12),
          (['ygccxxkx.kuo', 'dfayllal.ncc', 'eskxyqee.ber', 'etnmhxkq.bfw', 'ophquhiw.uby'], 'sfpzqjfb.rhz', 17), (
          ['dfayllal.ncc', 'etnmhxkq.bfw', 'vpwzdoby.uxw', 'miicppry.qvk', 'mxhuccdg.xzz', 'ophquhiw.uby',
           'yxvxabgd.ycj', 'ljfynrlt.zbf'], 'oopozpnj.rax', 42),
          (['dfayllal.ncc', 'yxvxabgd.ycj', 'sfpzqjfb.rhz', 'wmeliwtp.lnm'], 'wzxzwadk.ldk', 75)], 'wzxzwadk.ldk'),
        300,
    ),
)


def check(args, sol):
    (source_files, transformations, target_file) = args
    time = build_time(source_files, transformations, target_file)
    return time == sol


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
