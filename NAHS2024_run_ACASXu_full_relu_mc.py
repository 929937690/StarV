"""
Verify ACASXU (ReLU, LeakyReLU, SatLin, SatLins) networks and generate full table
Author: Yuntao Li
Date: 2/10/2024
"""


from StarV.verifier.verifier import quantiVerifyMC
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_piecewise import load_ACASXU_ReLU, load_ACASXU_LeakyReLU, load_ACASXU_SatLin, load_ACASXU_SatLins
import time
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util


def quantiverify_ACASXU_all_ReLU_MC(x, y, spec_ids, unsafe_mat, unsafe_vec, numSamples, nTimes, numCore):
    """Verify all ACASXU ReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for numSample in numSamples:
            print_util('h3')
            print('quanti verify using Monte Carlo of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a  = 3.0 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            netName = '{}-{}'.format(x[i], y[i])

            start = time.time()
            unsafe_prob = quantiVerifyMC(net=net, inputSet=In, unsafe_mat=unsmat, unsafe_vec=unsvec, numSamples=numSample, nTimes=nTimes, numCores=numCore)
            end = time.time()

            verifyTime = end-start
            data.append([spec_ids[i], netName, numSample, unsafe_prob, verifyTime])
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "N-samples", "UnsafeProb", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024/ACASXU/ReLU/MonteCarlo"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTableMC.tex", "w") as f:
        print(tabulate(data, headers=["Prop.", "Net", "N-samples", "UnsafeProb", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



if __name__ == "__main__":

    # x = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]
    # y = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 8, 9, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9] 
    # s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4] # property id

    # x = [1, 2, 1, 1, 1, 1, 1]
    # y = [2, 6, 7, 8, 9, 7, 9] 
    # s = [2, 2, 3, 3, 3, 4, 4] # property id

    x = [1]
    y = [2] 
    s = [2] # property id

    # numSamplesList = [10000, 100000, 1000000]
    numSamplesList = [10000000]

    quantiverify_ACASXU_all_ReLU_MC(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16)