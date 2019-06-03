# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

from cora.signal import corr21cm
from cora.foreground import galaxy

def test_corr_signal():
    """Test that the signal power spectrum is being calculated correctly.

    Correct here is referenced to a specific version believed to have no errors.
    """

    cr = corr21cm.Corr21cm()

    aps1 = cr.angular_powerspectrum(np.arange(1000), 800, 800)
    #assert len(aps1) == 1000
    #assert np.allclose(aps1.sum(), 1.5963772205823096e-09, rtol=1e-7)  # Calculated for commit 02f4d1cd3f402d

    fa = np.linspace(800.0, 400.0, 64)
    aps2 = cr.angular_powerspectrum(
        np.arange(1000)[:, None, None], fa[None, :,None], fa[None, None, :]
    )
    #assert aps2.shape == (1000, 64, 64)

    # Calculated for commit 02f4d1cd3f402d
    #v1 = 8.986790805379046e-13  # l=400, fi=40, fj=40
    #v2 = 1.1939298801340165e-18  # l=200, fi=10, fj=40
    #assert np.allclose(aps2[400, 40, 40], v1, rtol=1e-7)
    #assert np.allclose(aps2[200, 10, 40], v2, rtol=1e-7)
    return aps2, fa


def test_corr_foreground():
    """Test that the foreground power spectrum is being calculated correctly.

    Correct here is referenced to a specific version believed to have no errors.
    """
    cr = galaxy.FullSkySynchrotron()

    aps1 = cr.angular_powerspectrum(np.arange(1000), 800.0, 800.0)
    assert len(aps1) == 1000
    assert np.allclose(aps1.sum(), 75.47681191093129, rtol=1e-7)  # Calculated for commit 02f4d1cd3f402d

    fa = np.linspace(400.0, 800.0, 64)
    aps2 = cr.angular_powerspectrum(
        np.arange(1000)[:, None, None], fa[None, :,None], fa[None, None, :]
    )
    assert aps2.shape == (1000, 64, 64)

    # Calculated for commit 02f4d1cd3f402d
    v1 = 9.690708728692975e-06  # l=400, fi=40, fj=40
    v2 = 0.00017630767166797886  # l=200, fi=10, fj=40
    assert np.allclose(aps2[400, 40, 40], v1, rtol=1e-7)
    assert np.allclose(aps2[200, 10, 40], v2, rtol=1e-7)

def clarray(aps, lmax, zarray, zromb=3, zwidth=None):
    """Calculate an array of C_l(z, z').
    Parameters
    ----------
    aps : function
        The angular power spectrum to calculate.
    lmax : integer
        Maximum l to calculate up to.
    zarray : array_like
        Array of z's to calculate at.
    zromb : integer
        The Romberg order for integrating over frequency samples.
    zwidth : scalar, optional
        Width of frequency channel to integrate over. If None (default),
        calculate from the separation of the first two bins.
    Returns
    -------
    aps : np.ndarray[lmax+1, len(zarray), len(zarray)]
        Array of the C_l(z,z') values.
    """


    if zromb == 0:
        return aps(np.arange(lmax + 1)[:, np.newaxis, np.newaxis],
                   zarray[np.newaxis, :, np.newaxis], zarray[np.newaxis, np.newaxis, :])

    else:
        zsort = np.sort(zarray)
        zhalf = np.abs(zsort[1] - zsort[0]) / 2.0 if zwidth is None else zwidth / 2.0
        zlen = zarray.size
        zint = 2**zromb + 1
        zspace = 2.0 * zhalf / 2**zromb

        za = (zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zint)[np.newaxis, :]).flatten()

        lsections = np.array_split(np.arange(lmax + 1), lmax // 5)

        cla = np.zeros((lmax + 1, zlen, zlen), dtype=np.float64)

        for lsec in lsections:
            clt = aps(lsec[:, np.newaxis, np.newaxis],
                      za[np.newaxis, :, np.newaxis], za[np.newaxis, np.newaxis, :])

            clt = clt.reshape(-1, zlen, zint, zlen, zint)

            clt = si.romb(clt, dx=zspace, axis=4)
            clt = si.romb(clt, dx=zspace, axis=2)

            cla[lsec] = clt / (2 * zhalf)**2  # Normalise

        return cla

def main():
    # aps2, fa = test_corr_signal()
    # print(aps2.shape)

    # plt.plot(aps2[:,0,10])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()       

    cr = corr21cm.Corr21cm()
    fa = np.linspace(400.0, 800.0, 64) 
    cr.frequencies = fa
    lmax = 1000     
    cla = clarray(cr.angular_powerspectrum, lmax, fa)
    print(cla.shape)
    cl = cla[:,10,11]
    for l in range(len(cl)):
        cl[l] = (l*(l+1)*cl[l])/(2*np.pi)
    plt.plot(cl)
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()      

if __name__ == "__main__":
    main()


