# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2020  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
This module is mainly about the calculation of the Rambo-Tainer invariant
described in:

https://dx.doi.org/10.1038%2Fnature12070  
"""
__authors__ = ["Martha E. Brennich", "J. Kieffer"]
__license__ = "MIT"
__date__ = "10/06/2020"

import numpy
from .collections import  


def calc_Vc(data, Rg, dRg, I0, dI0, imin):
    """Calculates the Rambo-Tainer invariant Vc, including extrapolation to q=0
    
    :param dat:  data in q,I,dI format, cropped to maximal q that should be used for calculation (normally 2 nm-1)
    :param Rg,dRg,I0,dI0:  results from Guinier approximation/autorg
    :param imin:  minimal index of the Guinier range, below that index data will be extrapolated by the Guinier approximation
    :returns: Vc and an error estimate based on non-correlated error propagation
    """
    dq = data[1, 0] - data[0, 0]
    qmin = data[imin, 0]
    qlow = numpy.arange(0, qmin, dq)

    lowqint = numpy.trapz((qlow * I0 * numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0)), qlow)
    dlowqint = numpy.trapz(qlow * numpy.sqrt((numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0) * dI0) ** 2 + ((I0 * 2.0 * (qlow * qlow) * Rg / 3.0) * numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0) * dRg) ** 2), qlow)
    vabs = numpy.trapz(data[imin:, 0] * data[imin:, 1], data[imin:, 0])
    dvabs = numpy.trapz(data[imin:, 0] * data[imin:, 2], data[imin:, 0])
    vc = I0 / (lowqint + vabs)
    dvc = (dI0 / I0 + (dlowqint + dvabs) / (lowqint + vabs)) * vc
    return (vc, dvc)


def calc_Rambo_Tainer(data, guinier, qmax=2):
    """calculates the invariants Vc and Qr from the Rambo & Tainer 2013 Paper,
    also the the mass estimate based on Qr for proteins
    
    :param data: data in q,I,dI format, q in nm-1
    :param Rg,dRg,I0,dI0: results from Guinier approximation
    :param imin: minimal index of the Guinier range, below that index data will be extrapolated by the Guinier approximation
    @param qmax: maximum q-value for the calculation in nm-1
    @return: dict with Vc, Qr and mass plus errors
    """
    scale_prot = 1.0 / 0.1231
    power_prot = 1.0

    imax = abs(dat[:, 0] - qmax).argmin()
    if (imax <= imin) or (imin < 0):  # unlikely but can happened
        return {}
    vc = calcVc(dat[:imax, :], Rg, dRg, I0, dI0, imin)

    qr = vc[0] ** 2 / (Rg)
    mass = scale_prot * qr ** power_prot

    dqr = qr * (dRg / Rg + 2 * ((vc[1]) / (vc[0])))
    dmass = mass * dqr / qr

    return {'Vc': vc[0], 'dVc': vc[1], 'Qr': qr, 'dQr': dqr, 'mass': mass, 'dmass': dmass}