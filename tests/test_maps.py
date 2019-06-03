# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

from cora.foreground import galaxy, pointsource
from cora.signal import corr21cm
import matplotlib.pyplot as plt
from cora.util import units
from cora.util.cosmology import ps_nowiggle

nside = 32
fa = np.linspace(400.0, 800.0, 64)


def test_galaxy():

    cr = galaxy.ConstrainedGalaxy()
    cr.nside = nside
    cr.frequencies = fa

    unpol = cr.getsky()
    assert unpol.shape == (32, 12 * nside**2)

    pol = cr.getpolsky()
    assert pol.shape == (32, 4, 12 * nside**2)

    # Check fluctuation amplitude for unpol
    ustd = unpol.std(axis=-1)
    assert (ustd > 10.0).all()
    assert (ustd < 50.0).all()

    # Check fluctuation amplitude for pol
    pstd = pol.std(axis=-1)
    assert (pstd[:, 0] > 10.0).all()
    assert (pstd[:, 0] < 50.0).all()
    assert (pol.std(axis=-1)[:, 1:3] > 0.1).all()
    assert (pol.std(axis=-1)[:, 1:3] < 3.0).all()
    assert (pol.std(axis=-1)[:, 3] == 0.0).all()

def test_pointsource():

    cr = pointsource.CombinedPointSources()
    cr.nside = nside
    cr.frequencies = fa

    unpol = cr.getsky()
    assert unpol.shape == (32, 12 * nside**2)

    pol = cr.getpolsky()
    assert pol.shape == (32, 4, 12 * nside**2)

    # Check fluctuation amplitude for unpol
    ustd = unpol.std(axis=-1)
    assert (ustd > 3.0).all()
    assert (ustd < 15.0).all()

    # Check fluctuation amplitude for pol
    pstd = pol.std(axis=-1)
    assert (pstd[:, 0] > 3.0).all()
    assert (pstd[:, 0] < 15.0).all()
    assert (pol.std(axis=-1)[:, 1:3] > 0.01).all()
    assert (pol.std(axis=-1)[:, 1:3] < 0.05).all()
    assert (pol.std(axis=-1)[:, 3] == 0.0).all()

def test_21cm():

    cr = corr21cm.Corr21cm()
    cr.nside = nside
    cr.frequencies = fa
    # redshift = units.nu21 / fa - 1.0

    # unpol = cr.getsky()
    # assert unpol.shape == (32, 12 * nside**2)

    # pol = cr.getpolsky()
    # assert pol.shape == (32, 4, 12 * nside**2)

    # return cr.T_b(redshift)

    clms = cr.getclms(1000)
    alms = cr.getalms2(1000)
    return clms, alms

    # Check fluctuation amplitude for unpol
    # ustd = unpol.std(axis=-1)
    # assert (ustd > 3.0).all()
    # assert (ustd < 15.0).all()

    # Check fluctuation amplitude for pol
    # pstd = pol.std(axis=-1)
    # assert (pstd[:, 0] > 3.0).all()
    # assert (pstd[:, 0] < 15.0).all()
    # assert (pol.std(axis=-1)[:, 1:3] > 0.01).all()
    # assert (pol.std(axis=-1)[:, 1:3] < 0.05).all()
    # assert (pol.std(axis=-1)[:, 3] == 0.0).all()

def main():
    # x = np.linspace(2e-3,1e0,100000)
    # pk = ps_nowiggle(x)
    # plt.plot(x,pk)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # Tb = test_21cm()
    # plt.plot(Tb)
    # plt.show()

    clms, alms = test_21cm()
    # print(alms.shape)
    lmax = clms.shape[0]

    zstart = 0
    zstop = 10
    print(zstart,zstop)
    cl = clms[:,zstop,zstop]
    al = alms[zstop]

    cl2 = np.zeros(lmax)
    for l in range(lmax):
        cl2[l] += np.abs(al[l,0])**2
        for m in range(1,l+1,1):
            cl2[l] += 2*np.abs(al[l,m])**2
        cl2[l] = cl2[l]/(2.0*l+1.0)

    for l in range(len(cl)):
        cl[l] = (l*(l+1)*cl[l])/(2*np.pi)
        cl2[l] = (l*(l+1)*cl2[l])/(2*np.pi)
    
    plt.plot(cl2)
    plt.plot(cl)
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()        

if __name__ == "__main__":
    main()
