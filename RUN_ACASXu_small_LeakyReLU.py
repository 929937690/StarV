"""
Verify ACASXU networks LeakyReLU small table
Author: Yuntao Li
Date: 2/10/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_leakyrelu import load_ACASXU_LeakyReLU
import time
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util


def quantiverify_ACASXU_all_LeakyReLU(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU LeakyReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} LeakyReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_LeakyReLU(x[i], y[i], spec_ids[i])
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
            inputSet = []
            inputSet.append(In)
            inputSetProb = inputSet[0].estimateProbability()
            netName = '{}-{}'.format(x[i], y[i])
            start = time.time()
            OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                    unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
            end = time.time()
            verifyTime = end-start
            data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/Journal/ACASXU/LeakyReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable_small.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data


if __name__ == "__main__":
   
    x = [1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 1]
    y = [6, 2, 9, 1, 6, 7, 1, 7, 3, 7, 9] 
    s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4] # property id

    x = [1]
    y = [9]
    s = [4]
    quantiverify_ACASXU_all_LeakyReLU(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])

    
