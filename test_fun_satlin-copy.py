"""
Test SatLin Class
Author: Yuntao Li
Date: 1/18/2024
"""

from StarV.fun.satlin import SatLin
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
from StarV.util.plot import plot_probstar, plot_probstar_using_Polytope
# import ipyparallel as ipp
# import subprocess


class Test(object):
    """
    Testing SatLin class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_reachExactSingleInput(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting reachExactSingleInput method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        # # case 1:
        # pred_lb = np.array([0.2, 0.2])
        # pred_ub = np.array([0.5, 0.5])
        # # case 2:
        # pred_lb = np.array([0.2, 0.2])
        # pred_ub = np.array([1.5, 1.5])
        # # case 3:
        # pred_lb = np.array([-0.2, -0.2])
        # pred_ub = np.array([0.5, 0.5])
        # # case 4:
        # pred_lb = np.array([-2.0, -2.0])
        # pred_ub = np.array([2.0, 2.0])
        # case 5:
        # pred_lb = np.array([1.2, 1.2])
        # pred_ub = np.array([3.0, 3.0])
        # # case 6:
        # pred_lb = np.array([-3.0, -3.0])
        # pred_ub = np.array([-1.2, -1.2])
        
        inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)
        print(inputSet.__str__())
        # plot_probstar(inputSet)
        # S = SatLin.reachExactSingleInput(inputSet, 'gurobi')
        S = SatLin.reachExactSingleInput(inputSet, 'gurobi')
        print('\nNumber of output set = {}'.format(len(S)))
        for S1 in S:
            print(S1.__str__())
        # try:
        #     S = SatLin.reachExactSingleInput(inputSet, 'gurobi')
        #     for S1 in S:
        #         print(S1.__str__())
        #     # plot_probstar(S)
        #     print('\nNumber of output set = {}'.format(len(S)))
        # except Exception:
        #     print('\nTest Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')


if __name__ == "__main__":

    test_SatLin = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_SatLin.test_reachExactSingleInput()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_SatLin.n_fails,
                            test_SatLin.n_tests - test_SatLin.n_fails,
                            test_SatLin.n_tests))
