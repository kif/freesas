# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2020-2026  European Synchrotron Radiation Facility, Grenoble, France
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
Set of namedtuples/dataclasses defined a bit everywhere
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "2020-2026 ESRF"
__date__ = "09/03/2026"

from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple
import numpy

# Used in AutoRg
RG_RESULT = namedtuple(
    "RG_RESULT",
    "Rg sigma_Rg I0 sigma_I0 start_point end_point quality aggregated",
)


def _RG_RESULT_repr(self):
    return f"Rg={self.Rg:6.4f}(±{self.sigma_Rg:6.4f}) I0={self.I0:6.4f}(±{self.sigma_I0:6.4f}) [{self.start_point}-{self.end_point}] {100.0 * self.quality:5.2f}% {'aggregated' if self.aggregated > 0.1 else ''}"


RG_RESULT.__repr__ = _RG_RESULT_repr

FIT_RESULT = namedtuple(
    "FIT_RESULT",
    "slope sigma_slope intercept sigma_intercept, R, R2, chi2, RMSD",
)
RT_RESULT = namedtuple("RT_RESULT", "Vc sigma_Vc Qr sigma_Qr mass sigma_mass")


def _RT_RESULT_repr(self):
    return f"Vc={self.Vc:6.4f}(±{self.sigma_Vc:6.4f}) Qr={self.Qr:6.4f}(±{self.sigma_Qr:6.4f}) mass={self.mass:6.4f}(±{self.sigma_mass:6.4f})"


RT_RESULT.__repr__ = _RT_RESULT_repr

# Used in BIFT
RadiusKey = namedtuple("RadiusKey", "Dmax npt")
PriorKey = namedtuple("PriorKey", "type npt")
TransfoValue = namedtuple("TransfoValue", "transfo B sum_dia")
EvidenceKey = namedtuple("EvidenceKey", "Dmax alpha npt")
EvidenceResult = namedtuple(
    "EvidenceResult", "evidence chi2r regularization radius density converged"
)

class StatsResult(NamedTuple):
    radius: numpy.ndarray|None = None
    density_avg: numpy.ndarray|None = None
    density_std: numpy.ndarray|None = None
    evidence_avg: float|None = None
    evidence_std: float|None = None
    Dmax_avg: float|None = None
    Dmax_std: float|None = None
    alpha_avg: float|None = None
    alpha_std: float|None = None
    chi2r_avg: float|None = None
    chi2r_std: float|None = None
    regularization_avg: float|None = None
    regularization_std: float|None = None
    Rg_avg: float|None = None
    Rg_std: float|None = None
    I0_avg: float|None = None
    I0_std: float|None = None

    def save(self, filename, source=None):
        "Save the results of the fit to the file"
        res = [
            f"Dmax= {self.Dmax_avg:.2f}±{self.Dmax_std:.2f}",
            f"𝛂= {self.alpha_avg:.1f}±{self.alpha_std:.1f}",
            f"S₀= {self.regularization_avg:.4f}±{self.regularization_std:.4f}",
            f"χ²= {self.chi2r_avg:.2f}±{self.chi2r_std:.2f}",
            f"logP= {self.evidence_avg:.2f}±{self.evidence_std:.2f}",
            f"Rg= {self.Rg_avg:.2f}±{self.Rg_std:.2f}",
            f"I₀= {self.I0_avg:.2f}±{self.I0_std:.2f}",
        ]
        with open(filename, "wt", encoding="utf-8") as out:
            out.write("# %s %s" % (source or filename, "\n"))
            for txt in res:
                out.write(f"# {txt} \n")
            out.write("\n# r\tp(r)\tsigma_p(r)\n")
            for r, p, s in zip(
                self.radius.astype(numpy.float32),
                self.density_avg.astype(numpy.float32),
                self.density_std.astype(numpy.float32),
            ):
                out.write("%s\t%s\t%s%s" % (r, p, s, "\n"))
        return filename + ": " + "; ".join(res)


# Used in Cormap
GOF = namedtuple("GOF", ["n", "c", "P"])


class UVJuice(NamedTuple):
    """All information of an UV-file"""
    wavelengths: numpy.ndarray
    timestamps: numpy.ndarray
    absorbance: numpy.ndarray

    @classmethod
    def from_file(cls, filename):
        """Create dataclass from filename

        :param filename: name or Path of the .dat file to read & parse
        :return: dataclass instance
        """
        with open(filename) as fd:
            header = fd.readline()
        keys = [k.strip() for k in header.split("|")]
        raw = numpy.loadtxt(filename, skiprows=1, delimiter="|", unpack=True)
        nb_time = raw.shape[1]
        nb_wl = raw.shape[0] // 3
        absorbance = numpy.empty((nb_wl, nb_time))
        timestamps = numpy.empty(nb_time)
        wavelengths = numpy.empty(nb_wl)
        for i,k in enumerate(keys):
            if k=="T0":
                timestamps = raw[i]
            elif k.startswith("w"):
                j = int(k[1])
                wavelengths[j] = raw[i,0]
            elif k.startswith("ABS"):
                j = int(k[3])
                absorbance[j] = raw[i]
        return cls(wavelengths, timestamps, absorbance)
