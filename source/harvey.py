import numpy as np

def noise_harvey(x, params):
    '''
        Compute the generalized Harvey model
    '''
    Nparams=len(params)
    NHarvey=int(Nparams/3.)
    if np.fix(NHarvey) != NHarvey:
        print('Error: The invalid number of parameters in params vector for function model.py::noise_harvey()')
        print('        Debug required. The program will exit now')
        exit()
    try:
        m=np.zeros(len(x))
    except:
        m=0
    nh=0
    for j in range(NHarvey):
        m=m+params[nh]/(1. + (params[nh+1]*(1e-3*x))**params[nh+2])
        nh=nh+3
    return m