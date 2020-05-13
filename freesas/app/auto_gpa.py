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
__date__ = "13/05/2020"

import os
import argparse
import logging
import glob
import platform
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto_gpa")

import numpy
import freesas
from freesas import autorg


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
    parser.add_argument("-v", "--verbose", default=False, help="switch to verbose mode", action='store_true')
    parser.add_argument("-V", "--version", action='version', version=version)
    args = parser.parse_args()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)
    return files


def main():
    list_files = parse()
    for afile in list_files:
        try:
            data = numpy.loadtxt(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            try:
                rg = autorg.auto_gpa(data)
            except Exception as err:
                print("%s %s" % (afile, err))
            else:
                print("%s %s" % (afile, rg))


if __name__ == "__main__":
    main()