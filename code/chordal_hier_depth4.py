# 深度 4（b=2）分层星套星的双精度稀疏计算（§18.5）。用法：cd code && python3 chordal_hier_depth4.py
import numpy as np, time, chordal_hier_star as H
from chordal_hier_sparse import log_trel_sparse
b=2
for dep in (3,4):
    tr=H.Tree((b,)*dep); ch=tr.chain(); g=1/(b+1); e=[b*g]+[1.0]*(dep-1)+[g]
    v=[]
    for p in (1.5,2.0,2.5,3.0):
        t0=time.time(); v.append(log_trel_sparse(tr,ch,e,np.log(10**p)))
        print(f'深度{dep} Λ=1e{p}: t·c={np.exp(v[-1]):.5g} ({time.time()-t0:.0f}s)', flush=True)
    print(f'深度{dep} 局部斜率:', ' '.join('%.3f'%((v[i+1]-v[i])/np.log(10**0.5)) for i in range(len(v)-1)), flush=True)
