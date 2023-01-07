import numpy as np

def index_1p(vibmax):

    idx_1p = -np.ones(vibmax, dtype=int)

    ct_1p = 0
    for v in range(vibmax):
        idx_1p[v] = ct_1p
        ct_1p += 1

    return idx_1p, ct_1p

def index_2p(vibmax, srange, ct_1p):

    idx_2p = -np.ones((vibmax, len(srange), vibmax), dtype=int)

    ct_2p = 0
    for v1 in range(vibmax):
        for s in srange:
            if s == 0: continue
            for v2 in range(1, vibmax):
                if v1 + v2 > vibmax - 1: continue
                idx_2p[v1, s, v2] = ct_2p + ct_1p
                ct_2p += 1

    return idx_2p, ct_2p

def index_ct(vibmax, sctrange, ct_1p2p):

    idx_ct = -np.ones((vibmax, len(sctrange), vibmax), dtype=int)

    ct_ct = 0
    for v1 in range(vibmax):
        for s in sctrange:
            if s == 0: continue
            for v2 in range(vibmax):
                if v1 + v2 > vibmax - 1: continue
                idx_ct[v1, s, v2] = ct_ct + ct_1p2p
                ct_ct += 1

    return idx_ct, ct_ct
