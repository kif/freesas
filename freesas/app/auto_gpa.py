#!/usr/bin/python3
# coding: utf-8
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

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "14/05/2020"

import sys
import os
import argparse
import logging
import glob
import platform
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("auto_gpa")

import numpy
import freesas
from freesas import autorg

if sys.version_info < (3, 6):
    logger.error("This code uses F-strings and requires Python 3.6+")


def parse():
    """ Parse input and return list of files.
    :return: list of input files
    """
    usage = "auto_gpa.py [OPTIONS] FILES "
    description = "Calculate the radius of gyration using Guinier Peak Analysis (Putnam 2016) for a set of scattering curves"
    epilog = """auto_gpa.py is an open-source implementation of
    the autorg algorithm originately part of the ATSAS suite.
    As this tool used a differnt theory, some results may differ 
    """
    version = "auto_gpa.py version %s from %s" % (freesas.version, freesas.date)
    parser = argparse.ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("file", metavar="FILE", nargs='+', help="dat files to compare")
    parser.add_argument("-o", "--output", action='store', help="Output filename", default=None, type=str)
    parser.add_argument("-f", "--format", action='store', help="Output format: native, csv, ssf", default="native", type=str)
    parser.add_argument("-v", "--verbose", default=0, help="switch to verbose mode", action='count')
    parser.add_argument("-V", "--version", action='version', version=version)
    return parser.parse_args()


def main():
    args = parse()
    if args.verbose==1:
        logging.root.setLevel(logging.INFO)
    elif args.verbose>=2:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)

    if args.output:
        dst = open(args.output, "w")
    else:
        dst = sys.stdout

    if args.format == "csv":
        dst.write("File,Rg,Rg StDev,I(0),I(0) StDev,First point,Last point,Quality,Aggregated" + os.linesep)

    for afile in files:
        logger.info("Processing %s", afile)
        try:
            data = numpy.loadtxt(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            try:
                rg = autorg.auto_gpa(data)
            except Exception as err:
                sys.stdout.write("%s, %s: %s\n" % (afile, err.__class__.__name__, err))
            else:
                if args.format == "csv":
                    res = f"{afile},{rg.Rg:6f},{rg.sigma_Rg:6f},{rg.I0:6f},{rg.sigma_I0:6f},{rg.start_point},{rg.end_point},{rg.quality:6f},{rg.aggregated}"
                elif args.format == "ssv":
                    res = f"{rg.Rg:6f} {rg.sigma_Rg:6f} {rg.I0:6f} {rg.sigma_I0:6f} {rg.start_point} {rg.end_point} {rg.quality:6f} {rg.aggregated} {afile}"
                else:
                    res = "%s %s"%(afile, rg)
                dst.write(res)
                dst.write(os.linesep)
                dst.flush()
    if args.output:
        dst.close()


if __name__ == "__main__":
    main()
