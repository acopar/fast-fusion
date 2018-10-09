#!/usr/bin/env python

import time

from ffusion.common import *


@fusion
def df_zitnik(engine, X, Xt, mask, GG, S, TrX, dimensions, k=None, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    locals().update(engine._locals)
    timer = Timer()
    G, Gi = GG
    
    n_objects = len(G)
    
    XG = {}
    XU = {}
    XtUSV = {}
    XGG = {}
    ZBLJ = {}
    GtG = {}
        
    tr2 = {}
    tr3 = {}
    E = {}
    GtGi = {}
    T1 = {}
    T2 = {}
    H_e = {}
    H_d = {}
    Top = {}
    Bot = {}
    Top2 = {}
    Bot2 = {}
    t1 = {}
    t2 = {}
    T = {}
    Top_neg = {}
    Top_neg2 = {}
    Bot_neg = {}
    Bot_neg2 = {}
    VSt = {}
    GS = {}
    SVVS = {}
    SGGS = {}
    tmp1p = {}
    tmp1n = {}
    tmp2p = {}
    tmp2n = {}
    tmp4p = {}
    tmp4n = {}
    tmp5p = {}
    tmp5n = {}
    n = dimensions['n']
    k = dimensions['k']
    
    for i, j in mask:
        XG[i,j] = zeros(shape=(n[i], k[j]))
        XU[i,j] = zeros(shape=(n[j], k[i]))
        XGG[i,j] = zeros(shape=(k[j], k[i]))
        XtUSV[i,j] = zeros(shape=(k[j], k[j]))
        t1[i,j] = zeros(shape=(k[j], k[i]))
        t2[i,j] = zeros(shape=(k[j], k[j]))
        ZBLJ[i,j] = zeros(shape=(k[j], k[j]))
        T[i,j] = zeros(shape=(n[i], k[j]))
        VSt[i,j] = zeros(shape=(n[j], k[i]))
        GS[i,j] = zeros(shape=(n[i], k[j]))
        SVVS[i,j] = zeros(shape=(k[i], k[i]))
        SGGS[i,j] = zeros(shape=(k[j], k[j]))
        tmp2p[i,j] = zeros_like(SVVS[i,j])
        tmp2n[i,j] = zeros_like(SVVS[i,j])
        tmp5p[i,j] = zeros_like(SGGS[i,j])
        tmp5n[i,j] = zeros_like(SGGS[i,j])

    for i in range(n_objects):
        Bot[i] = zeros_like(G[i])
        Bot2[i] = zeros_like(G[i])
        GtG[i] = zeros(shape=(k[i], k[i]))
        T1[i] = zeros(shape=(n[i], k[i]))
        T2[i] = zeros(shape=(k[i], n[i]))
        Top[i] = zeros_like(G[i])
        Top2[i] = zeros_like(G[i])
        Top_neg[i] = zeros_like(G[i])
        Top_neg2[i] = zeros_like(G[i])
        Bot_neg[i] = zeros_like(G[i])
        Bot_neg2[i] = zeros_like(G[i])
        tmp1p[i] = zeros_like(G[i])
        tmp1n[i] = zeros_like(G[i])
        tmp4p[i] = zeros_like(G[i])
        tmp4n[i] = zeros_like(G[i])
        
    m = dimensions['m']
    slices = dimensions['slices']
    rank = engine.rank
    n_proc = engine.n_proc
    
    for i, j in mask:
        X[i,j] = sparse_init(X[i,j])
        Xt[i,j] = sparse_init(Xt[i,j])
        

    for i in G:
        G[i] = togpu(G[i])
    
    if n_proc != 1:
        for i in G:
            Gi[i] = togpu(Gi[i])

    for i, j in S:
        S[i,j] = togpu(S[i,j])

    t0 = time.time()
    
    XGT = {}
    XUT = {}

    if n_proc != 1:
        for i,j in mask:
            XGT[i,j] = {}
            XUT[i,j] = {}
            for d in range(n_proc):
                XGT[i,j][d] = zeros(shape=(m[i][d], k[j]))
                XUT[i,j][d] = zeros(shape=(n[j], k[i]))

    for it in range(max_iter):
        timer.split('Bigdot')
        
        for i, j in mask:
            if n_proc != 1:
                XGT[i,j][rank] = bigdot(X[i,j], G[j])
            else:
                XG[i,j] = bigdot(X[i,j], G[j])
        
        for i, j in mask:
            if n_proc != 1:
                XUT[i,j][rank] = bigdot(Xt[i,j], Gi[i])
            else:
                XU[i,j] = bigdot(Xt[i,j], G[i])
        
        timer.split("Send/Recv & reduce")
        for i, j in mask:
            if n_proc != 1:
                if rank == 0:
                    for d in range(1,n_proc):
                        mpi_recv(XGT[i,j][d], d)
                
                    for d in slices[i]:
                        a,b = slices[i][d]
                        slice_assign(XGT[i,j][d], XG[i,j], a, b)
                else:
                    mpi_send(XGT[i,j][rank], 0)
        
        for i, j in mask:
            if n_proc != 1:
                if rank == 0:
                    XU[i,j] = XUT[i,j][rank]
                    for d in range(1,n_proc):
                        mpi_recv(XUT[i,j][d], d)
                        add(XU[i,j], XUT[i,j][d], out=XU[i,j])
                else:
                    mpi_send(XUT[i,j][rank], 0)
        
        if rank == 0:
            timer.split('Error')
            for i, j in mask:
                dot(XG[i,j], G[i], out=XGG[i,j], transa='T')
            
            for i, j in mask:
                dot(XGG[i,j], S[i,j], out=XtUSV[i,j])
            
            for i in range(n_objects):
                dot(G[i], G[i], out=GtG[i], transa='T')
            
            for i, j in mask:
                dot(S[i,j], GtG[i], out=t1[i,j], transa='T')
                dot(t1[i,j], S[i,j], out=t2[i,j])
                dot(t2[i,j], GtG[j], out=ZBLJ[i,j])
    
            for i, j in mask:
                tr2[i,j] = trace(XtUSV[i,j])
            
            for i, j in mask:
                tr3[i,j] = trace(ZBLJ[i,j])
            
            for i, j in mask:
                E[i,j] = 1 - 2*tr2[i,j]/TrX[i,j] + tr3[i,j]/TrX[i,j]
            
            Esum = 0
            for i, j in E:
                Esum += E[i,j]
            
            err_history.append(Esum)
            if verbose:
                print("Error", Esum)
                sys.stdout.flush()
    
            timer.split('S')
            #GtG = {}
            #for i in range(len(G)):
            #    GtG[i] = dot(G[i].T, G[i])
            
            
            for i in range(n_objects):
                GtGi[i] = inverse(GtG[i])
            
            for i in range(n_objects):
                dot(G[i], GtGi[i], out=T1[i])
    
            for i in range(n_objects):
                dot(GtGi[i], G[i], out=T2[i], transb='T')
            
            for i, j in mask:
                #T = bigdot(X[i,j], T1[j])
                dot(XG[i,j], GtGi[j], out=T[i,j])
                dot(T2[i], T[i,j], out=S[i,j])
    
            timer.split('G')
    
            for i in range(n_objects):
                H_e[i] = zeros_like(G[i])
                H_d[i] = zeros_like(G[i])
            
            for i, j in mask:
                timer.split('G[i]')
                dot(XG[i,j], S[i,j], out=Top[i], transb='T')
                
                pomus(Top[i], tmp1p[i], tmp1n[i])
                
                dot(G[j], S[i,j], out=VSt[i,j], transb='T')
                dot(VSt[i,j], VSt[i,j], out=SVVS[i,j], transa='T')
                pomus(SVVS[i,j], tmp2p[i,j], tmp2n[i,j])
                
                dot(G[i], tmp2p[i,j], out=Bot_neg[i])
                dot(G[i], tmp2n[i,j], out=Top_neg[i])
                
                add(tmp1p[i], Top_neg[i], out=Top[i])
                add(tmp1n[i], Bot_neg[i], out=Bot[i])
    
                add(H_e[i], Top[i], out=H_e[i])
                add(H_d[i], Bot[i], out=H_d[i])
                
                timer.split('G[j]')
                
                dot(XU[i,j], S[i,j], out=Top2[j])
                pomus(Top2[j], tmp4p[j], tmp4n[j])
                
                dot(G[i], S[i,j], out=GS[i,j])
                dot(GS[i,j], GS[i,j], out=SGGS[i,j], transa='T')
                pomus(SGGS[i,j], tmp5p[i,j], tmp5n[i,j])
                
                dot(G[j], tmp5n[i,j], out=Top_neg2[j])
                dot(G[j], tmp5p[i,j], out=Bot_neg2[j])
                
                add(tmp4p[j], Top_neg2[j], out=Top2[j])
                add(tmp4n[j], Bot_neg2[j], out=Bot2[j])
                
                add(H_e[j], Top2[j], out=H_e[j])
                add(H_d[j], Bot2[j], out=H_d[j])
                #    print("H_e[j]", fromgpu(tmp1p[i]).sum(), fromgpu(tmp1n[i]).sum())
            
            timer.split('G divide')
            
            for i in range(n_objects):
                H_e[i] = divide(H_e[i], H_d[i], out=H_e[i])
                Ts = sqrt(H_e[i])
                G[i] = multiply(G[i], Ts, out=G[i])
            
    
        
        timer.split('MPI sync')
        if n_proc != 1:
            for i in range(n_objects):
                if rank == 0:
                    for d in range(1,n_proc):
                        mpi_send(G[i], d)
                else:
                    mpi_recv(G[i], 0)
        
        if n_proc != 1:
            for i in range(n_objects):
                a = G[i].shape[0]*(rank)//n_proc
                b = G[i].shape[0]*(rank+1)//n_proc
                Gi[i] = slice_assign_back(Gi[i], G[i], a, b)
            #Gi[i] = togpu(fromgpu(G[i])[a:b,:])
        
        #if check_stop(err_history) > 0:
        #    print("Stopping after %d iterations" % it)
        #    break
    
    t1 = time.time()
    engine._locals = locals()
    timer.split('fromgpu')
    if rank == 0:
        print("Iterations time:", t1 - t0)
    
    for i in range(n_objects):
        G[i] = fromgpu(G[i])
    
    for i, j in S:
        S[i,j] = fromgpu(S[i,j])
    
    if rank == 0:
        print("Timer", str(timer))
    
    factors = G, S
    return factors, err_history


@transform
def df_transform(engine, X, Xt, mask, G, S, Gi, TrX, k=None, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    target = list(Gi.keys())[0]
    Gi = Gi[target]
    
    VSt = {}
    GS = {}
    SVVS = {}
    SGGS = {}
    tmp2p = {}
    tmp2n = {}
    tmp5p = {}
    tmp5n = {}
    
    Bot = {}
    Bot2 = {}
    Top = {}
    Top2 = {}
    Top_neg = {}
    Top_neg2 = {}
    Bot_neg = {}
    Bot_neg2 = {}
    tmp1p = {}
    tmp1n = {}
    tmp4p = {}
    tmp4n = {}
    T = {}
    
    timer.split('init')
    for i, j in mask:
        VSt[i,j] = zeros(shape=(G[j].shape[0], S[i,j].shape[0]))
        GS[i,j] = zeros(shape=(G[i].shape[0], S[i,j].shape[1]))
        SVVS[i,j] = zeros(shape=(S[i,j].shape[0], S[i,j].shape[0]))
        SGGS[i,j] = zeros(shape=(S[i,j].shape[1], S[i,j].shape[1]))
        tmp2p[i,j] = zeros_like(SVVS[i,j])
        tmp2n[i,j] = zeros_like(SVVS[i,j])
        tmp5p[i,j] = zeros_like(SGGS[i,j])
        tmp5n[i,j] = zeros_like(SGGS[i,j])
    
    
    for i in [target]:
        Bot[i] = zeros_like(Gi)
        Bot2[i] = zeros_like(Gi)
        Top[i] = zeros_like(Gi)
        Top2[i] = zeros_like(Gi)
        Top_neg[i] = zeros_like(Gi)
        Top_neg2[i] = zeros_like(Gi)
        Bot_neg[i] = zeros_like(Gi)
        Bot_neg2[i] = zeros_like(Gi)
        tmp1p[i] = zeros_like(Gi)
        tmp1n[i] = zeros_like(Gi)
        tmp4p[i] = zeros_like(Gi)
        tmp4n[i] = zeros_like(Gi)
        T[i] = zeros_like(Gi)
    
    

    for i, j in mask:
        X[i,j] = sparse_init(X[i,j])
        Xt[i,j] = sparse_init(Xt[i,j])

    for i in G:
        G[i] = togpu(G[i])
    
    for i, j in S:
        S[i,j] = togpu(S[i,j])

    Gi = togpu(Gi)
    
    timer.split('iterations')
    for it in range(max_iter):
        H_e = {}
        H_d = {}
        H_e[target] = zeros_like(Gi)
        H_d[target] = zeros_like(Gi)
        
        for i, j in mask:
            if i == target:
                XV = bigdot(X[i,j], G[j])
                dot(XV, S[i,j], out=Top[i], transb='T')
                
                pomus(Top, tmp1p[i], tmp1n[i])
                
                dot(G[j], S[i,j], out=VSt[i,j], transb='T')
                dot(VSt[i,j], VSt[i,j], out=SVVS[i,j], transa='T')
                pomus(SVVS[i,j], tmp2p[i,j], tmp2n[i,j])
                
                dot(Gi, tmp2p[i,j], out=Bot_neg[i])
                dot(Gi, tmp2n[i,j], out=Top_neg[i])
                
                add(tmp1p[i], Top_neg[i], out=Top[i])
                add(tmp1n[i], Bot_neg[i], out=Bot[i])
                
                add(H_e[i], Top[i], out=H_e[i])
                add(H_d[i], Bot[i], out=H_d[i])
            
            if j == target:
                XU = bigdot(Xt[i,j], G[i])
                dot(XU, S[i,j], out=Top2[j])
                pomus(Top2[j], tmp4p[j], tmp4n[j])
                dot(G[i], S[i,j], out=GS[i,j])
                dot(GS[i,j], GS[i,j], out=SGGS[i,j], transa='T')
                pomus(SGGS[i,j], tmp5p[i,j], tmp5n[i,j])
                
                dot(Gi, tmp5n[i,j], out=Top_neg2[j])
                dot(Gi, tmp5p[i,j], out=Bot_neg2[j])
                
                add(tmp4p[j], Top_neg2[j], out=Top2[j])
                add(tmp4n[j], Bot_neg2[j], out=Bot2[j])
                
                add(H_e[j], Top2[j], out=H_e[j])
                add(H_d[j], Bot2[j], out=H_d[j])
    
        T[target] = divide(H_e[target], H_d[target])
        Ts = sqrt(T[target])
        Gi = multiply(Gi, Ts)

        if check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    timer.split('fromgpu')
    for i in range(len(G)):
        G[i] = fromgpu(G[i])
    
    for i, j in S:
        S[i,j] = fromgpu(S[i,j])
    
    Gi = fromgpu(Gi)

    print("Timer", "iterations: %d" % it, str(timer))
    return Gi

