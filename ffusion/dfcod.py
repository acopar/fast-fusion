#!/usr/bin/env python

import time

from ffusion.common import *

@fusion
def df_cod(engine, X, Xt, mask, G, S, TrX, k=None, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    
    N = len(G)
    
    for it in range(max_iter):
        XG = {}
        for i, j in mask:
            XG[i,j] = bigdot(X[i,j], G[j])
        XG2 = {}
        for i, j in mask:
            XG2[i,j] = bigdot(Xt[i,j], G[i])
        
        XtUSV = {}
        for i, j in mask:
            XGG = dot(XG[i,j].T, G[i])
            XtUSV[i,j] = dot(XGG, S[i,j])
        
        ZBLJ = {}
        for i, j in mask:
            GtGi = dot(G[i].T, G[i])
            GtGj = dot(G[j].T, G[j])
            
            T1 = dot(S[i,j].T, GtGi)
            T2 = dot(T1, S[i,j])
            T3 = dot(T2, GtGj)
            ZBLJ[i,j] = T3
        
        
        tr2 = {}
        for i, j in mask:
            tr2[i,j] = np.trace(XtUSV[i,j])
        
        tr3 = {}
        for i, j in mask:
            tr3[i,j] = np.trace(ZBLJ[i,j])
        
        E = {}
        for i, j in mask:
            E[i,j] = 1 - 2*tr2[i,j]/TrX[i,j] + tr3[i,j]/TrX[i,j]
        
        Esum = 0
        for i,j in E:
            Esum += E[i,j]
        
        err_history.append(Esum)
        if verbose:
            print("Error", Esum)

        GtG = {}
        for i in range(N):
            GtG[i] = dot(G[i].T, G[i])
        
        GtGi = {}
        for i in range(N):
            GtGi[i] = inverse(GtG[i])
        
        T1 = {}
        for i in range(N):
            T1[i] = dot(G[i], GtGi[i])

        T2 = {}
        for i in range(N):
            T2[i] = dot(GtGi[i], G[i].T)
        
        for i, j in mask:
            #T = bigdot(X[i,j], T1[j])
            T = dot(XG[i,j], GtGi[j])
            S[i,j] = dot(T2[i], T)

        H_e = {}
        H_d = {}
        for i in range(N):
            H_e[i] = np.zeros_like(G[i])
            H_d[i] = np.zeros_like(G[i])
        
        dirtyG = {}
        for i in range(len(G)):
            dirtyG[i] = 0
        
        for I in G:
            H = G[I]
            #XHS = {}
            #for i, j in mask:
            #    XHS[i,j] = dot(XG[I,j], S[I,j])
            
            for i in range(H.shape[1]):
                H_j = H[:,i].reshape(-1,1)
                
                Sgc = {}
                SGGSi = {}
                for j in range(N):
                    if (I,j) in mask:
                        Sgc[I,j] = dot(G[j], S[I,j].T)
                        Sgi = Sgc[I,j][:,i].reshape(-1,1)
                        SGGSi[I,j] = dot(Sgi.T, Sgi)
                
                #Top = dot(XG[i,j], S[i,j].T)
                
                XH = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (I,j) in mask:
                        Si = S[I,j][i,:].reshape(1,-1)
                        XH += dot(XG[I,j], Si.T)
                
                sHHs = 0
                for j in range(N):
                    if (I,j) in mask:
                        #Si = S[I,j][i,:].reshape(1,-1)
                        #T = dot(Si, G[j].T).reshape(1,-1)
                        #T = dot(G[j], Si.T).reshape(1,-1)
                        #T = Sgc[I,j][:,i].reshape(1,-1)
                        sHHs += SGGSi[I,j]#dot(T, T.T)
                
                below = sHHs
                
                T2 = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (I,j) in mask:
                        #Si = S[I,j][i,:].reshape(1,-1)
                        #SG = dot(S[I,j], G[j].T)
                        #SG = dot(G[j], S[I,j].T)
                        SG = Sgc[I,j]
                        SGGS = dot(SG.T, SG[:,i]).reshape(-1,1)
                        T2 += dot(H, SGGS)
                
                T3 = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (I,j) in mask:
                        #Si = S[I,j][i,:].reshape(1,-1)
                        #Sg = dot(Si, G[j].T)
                        #Sg = Sgc[I,j][:,i].reshape(-1,1)
                        #Sggs = dot(Sg.T, Sg)
                        T3 += dot(H_j, SGGSi[I,j])
                
                Top = XH - T2 + T3
                
                Sgc2 = {}
                SGGSi = {}
                for j in range(N):
                    if (j,I) in mask:
                        Sgc2[j,I] = dot(G[j], S[j,I])
                        Sgi = Sgc2[j,I][:,i].reshape(-1,1)
                        SGGSi[j,I] = dot(Sgi.T, Sgi)
                
                H_j = H[:,i].reshape(-1,1)
                T1 = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (j,I) in mask:
                        Si = S[j,I][:,i].reshape(-1,1)
                        #T1 += dot(XG2[j,I], Si)
                        Xg2 = bigdot(Xt[j,I], G[j])
                        T1 += dot(Xg2, Si)
                
                T2 = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (j,I) in mask:
                        #Si = S[j,I][:,i].reshape(-1,1)
                        #SG = dot(S[j,I].T, G[j].T)
                        #SG = dot(G[j], S[j,I])
                        SG = Sgc2[j,I]
                        SGGS = dot(SG.T, SG[:,i]).reshape(-1,1)#dot(G[j], Si))
                        
                        T2 += dot(H, SGGS)
                
                T3 = np.zeros((H.shape[0], 1))
                for j in range(N):
                    if (j,I) in mask:
                        #Si = S[j,I][:,i].reshape(-1,1)
                        #Sg = dot(Si.T, G[j].T)
                        #Sg = dot(G[j], Si)
                        #Sg = Sgc2[j,I][:,i].reshape(-1,1)
                        #Sggs = dot(Sg.T, Sg)
                        T3 += dot(H_j, SGGSi[j,I])
                
                sHHs = 0
                for j in range(N):
                    if (j,I) in mask:
                        #Si = S[j,I][:,i].reshape(-1,1)
                        #T = dot(Si.T, G[j].T).reshape(1,-1)
                        #T = dot(G[j], Si).reshape(1,-1)
                        #T = Sgc2[j,I][:,i].reshape(1,-1)
                        sHHs += SGGSi[j,I]
                below2 = sHHs
                
                Top2 = T1 - T2 + T3
        
                Top += Top2
                below += below2
                NA33 = divide(Top, below)
                #NA34 = project(NA33)
                H[:,i] = NA33.ravel()
            dirtyG[I] = 1

        if check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    print("Timer", str(timer))
    factors = G, S
    return factors, err_history