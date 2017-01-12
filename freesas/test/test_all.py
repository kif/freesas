#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "12/01/2017"

import unittest
from . import test_model
from . import test_align
from . import test_distance
from . import test_cormap


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_model.suite())
    testSuite.addTest(test_align.suite())
    testSuite.addTest(test_distance.suite())
    testSuite.addTest(test_cormap.suite())
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
