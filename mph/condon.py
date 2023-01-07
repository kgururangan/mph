import numpy as np
from math import factorial

def volap(lambda1, v1, lambda2, v2):

    dlambda = lambda2 - lambda1

    overlap = 0.0
    for k in range(min(v1, v2) + 1):
        overlap += (
                (-1.0)**(v2-k)/(factorial(v1-k) * factorial(k) * factorial(v2-k)) * 
                dlambda**(v1 + v2 - 2*k)
        )
    overlap *= np.sqrt(factorial(v1) * factorial(v2)) * np.exp(-dlambda**2/2)

    return overlap

def fcmatrix(vibmax, lambda_f, lambda_c, lambda_a):

    fcmat = {'gf' : np.zeros((vibmax, vibmax)), # ground-to-frenkel
             'gc' : np.zeros((vibmax, vibmax)), # ground-to-cation
             'ga' : np.zeros((vibmax, vibmax)), # ground-to-anion
             'cf' : np.zeros((vibmax, vibmax)), # cation-to-frenkel
             'af' : np.zeros((vibmax, vibmax))} # anion-to-frenkel

    for v1 in range(vibmax):
        for v2 in range(vibmax):
            fcmat['gf'][v1, v2] = volap(0.0, v1, lambda_f, v2)
            fcmat['gc'][v1, v2] = volap(0.0, v1, lambda_c, v2)
            fcmat['ga'][v1, v2] = volap(0.0, v1, lambda_a, v2)
            fcmat['cf'][v1, v2] = volap(lambda_c, v1, lambda_f, v2)
            fcmat['af'][v1, v2] = volap(lambda_a, v1, lambda_f, v2)

    return fcmat

