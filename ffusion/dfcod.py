#!/usr/bin/env python

import time

from ffusion.common import *

@fusion
def df_cod(engine, X, Xt, mask, GG, S, TrX, dimensions, k=None, max_iter=10, verbose=False):
    err_history = []
    G, Gi = GG
    globals().update(engine.methods())
    timer = Timer()
    
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

    for i, j in mask:
        X[i,j] = sparse_init(X[i,j])
        Xt[i,j] = sparse_init(Xt[i,j])

    for i in G:
        G[i] = togpu(G[i])
    
    for i, j in S:
        S[i,j] = togpu(S[i,j])

    t0 = time.time()
    
    XGT = {}
    XUT = {}
    
    m = dimensions['m']
    slices = dimensions['slices']
    rank = engine.rank
    n_proc = engine.n_proc
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
    
            timer.split('S')
            
            for i in range(n_objects):
                GtGi[i] = inverse(GtG[i])
            
            for i in range(n_objects):
                dot(G[i], GtGi[i], out=T1[i])
    
            for i in range(n_objects):
                dot(GtGi[i], G[i], out=T2[i], transb='T')
            
            for i, j in mask:
                dot(XG[i,j], GtGi[j], out=T[i,j])
                dot(T2[i], T[i,j], out=S[i,j])
            
    
            timer.split('G')
    
            dirtyG = {}
            for i in range(n_objects):
                dirtyG[i] = 0
            
            
            Sgc = {}
            SGGSi = {}
            #for i,j in mask:
            #    if (i,j) in mask:
            #        Sgc[i,j] = dot(G[j], S[i,j].T)
            #        Sgi = Sgc[i,j][:,i].reshape(-1,1)
            #        SGGSi[i,j] = dot(Sgi.T, Sgi)
            
            for I in G:
                timer.split('G')
                H = G[I]
                #XHS = {}
                #for i, j in mask:
                #    XHS[i,j] = dot(XG[I,j], S[I,j])
                
                for i in range(H.shape[1]):
                    H_j = H[:,i].reshape(-1,1)
                    
                    timer.split('Part 1')
    
                    for j in range(n_objects):
                        if (I,j) in mask:
                            Sgc[I,j] = dot(G[j], S[I,j], transb='T')
                            Sgi = Sgc[I,j][:,i].reshape(-1,1)
                            SGGSi[I,j] = dot(Sgi, Sgi, transa='T')
                    
                    #Top = dot(XG[i,j], S[i,j].T)
                    timer.split('Part 2')
    
                    XH = np.zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (I,j) in mask:
                            Si = S[I,j][i,:].reshape(1,-1)
                            XH += dot(XG[I,j], Si, transb='T')
                    
                    sHHs = 0
                    for j in range(n_objects):
                        if (I,j) in mask:
                            sHHs += SGGSi[I,j]
                    
                    below = sHHs
                    
                    timer.split('Part 3')
    
                    Q2 = zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (I,j) in mask:
                            SG = Sgc[I,j]
                            SGGS = dot(SG, SG[:,i], transa='T').reshape(-1,1)
                            Q2 += dot(H, SGGS)
                    
                    timer.split('Part 4')
    
                    Q3 = zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (I,j) in mask:
                            Q3 += dot(H_j, SGGSi[I,j])
                    
                    Top = XH - Q2 + Q3
                    
                    timer.split('Part 5')
    
                    Sgc2 = {}
                    SGGSi = {}
                    for j in range(n_objects):
                        if (j,I) in mask:
                            Sgc2[j,I] = dot(G[j], S[j,I])
                            Sgi = Sgc2[j,I][:,i].reshape(-1,1)
                            SGGSi[j,I] = dot(Sgi, Sgi, transa='T')
                    
                    timer.split('Part 6')
    
                    H_j = H[:,i].reshape(-1,1)
                    Q1 = zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (j,I) in mask:
                            Si = S[j,I][:,i].reshape(-1,1)
                            Q1 += dot(XU[j,I], Si)
                    
                    timer.split('Part 7')
    
                    Q2 = zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (j,I) in mask:
                            SG = Sgc2[j,I]
                            SGGS = dot(SG.T, SG[:,i]).reshape(-1,1)
                            
                            Q2 += dot(H, SGGS)
                    
                    timer.split('Part 8')
    
                    Q3 = zeros((H.shape[0], 1))
                    for j in range(n_objects):
                        if (j,I) in mask:
                            Q3 += dot(H_j, SGGSi[j,I])
                    
                    sHHs = 0
                    for j in range(n_objects):
                        if (j,I) in mask:
                            sHHs += SGGSi[j,I]
                    below2 = sHHs
                    
                    timer.split('Part 9')
    
                    Top2 =  Q1 - Q2 + Q3
            
                    Top += Top2
                    below += below2
                    NA33 = divide(Top, below)
                    #NA34 = project(NA33)
                    H[:,i] = NA33.ravel()
                
                dirtyG[I] = 1
        
    
        timer.split('MPI sync')
    
        if n_proc != 1:
            for i in range(n_objects):
                if rank == 0:
                    for d in range(1,n_proc):
                        mpi_send(G[i], d)
                else:
                    mpi_recv(G[i], 0)
            
            for i in range(n_objects):
                a = G[i].shape[0]*(rank)//n_proc
                b = G[i].shape[0]*(rank+1)//n_proc
                Gi[i] = slice_assign_back(Gi[i], G[i], a, b)
        
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


