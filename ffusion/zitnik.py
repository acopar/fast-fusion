#!/usr/bin/env python

import time

from ffusion.common import *

@fusion
def df_zitnik(engine, X, Xt, mask, G, S, TrX, k=None, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    
    nobjects = len(G)
    
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
        for i in range(len(G)):
            GtG[i] = dot(G[i].T, G[i])
        
        GtGi = {}
        for i in range(len(G)):
            GtGi[i] = inverse(GtG[i])
        
        T1 = {}
        for i in range(len(G)):
            T1[i] = dot(G[i], GtGi[i])

        T2 = {}
        for i in range(len(G)):
            T2[i] = dot(GtGi[i], G[i].T)
        
        for i, j in mask:
            #T = bigdot(X[i,j], T1[j])
            T = dot(XG[i,j], GtGi[j])
            S[i,j] = dot(T2[i], T)

        H_e = {}
        H_d = {}
        for i in range(len(G)):
            H_e[i] = np.zeros_like(G[i])
            H_d[i] = np.zeros_like(G[i])
        
        for i, j in mask:
            XV = XG[i,j]#bigdot(X[i,j], G[j])
            Top = dot(XV, S[i,j].T)
            
            tmp1p, tmp1n = pomus(Top)
            
            VSt = dot(G[j], S[i,j].T)
            SVVS = dot(VSt.T, VSt)
            tmp2p, tmp2n = pomus(SVVS)
            
            Bot_neg = dot(G[i], tmp2p)
            Top_neg = dot(G[i], tmp2n)
            
            Top = add(tmp1p, Top_neg)
            Bot = add(tmp1n, Bot_neg)
            
            H_e[i] = add(H_e[i], Top)
            H_d[i] = add(H_d[i], Bot)
            
            XU = XG2[i,j]# bigdot(Xt[i,j], G[i])
            Top2 = dot(XU, S[i,j])
            tmp4p, tmp4n = pomus(Top2)
            GS = dot(G[i], S[i,j])
            SGGS = dot(GS.T, GS)
            tmp5p, tmp5n = pomus(SGGS)
            
            Top_neg2 = dot(G[j], tmp5n)
            Bot_neg2 = dot(G[j], tmp5p)
            
            Top2 = add(tmp4p, Top_neg2)
            Bot2 = add(tmp4n, Bot_neg2)
            
            H_e[j] = add(H_e[j], Top2)
            H_d[j] = add(H_d[j], Bot2)
    
        for i in range(len(G)):
            T = divide(H_e[i], H_d[i])
            Ts = sqrt(T)
            G[i] = multiply(G[i], Ts)

        if check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    print("Timer", str(timer))
    factors = G, S
    return factors, err_history
    
@transform
def df_transform(engine, X, mask, G, S, Gi, TrX, k=None, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    target = list(Gi.keys())[0]
    Gi = Gi[target]
    
    for it in range(max_iter):
        H_e = {}
        H_d = {}
        H_e[target] = np.zeros_like(Gi)
        H_d[target] = np.zeros_like(Gi)
        
        for i, j in mask:
            if i == target:
                XV = bigdot(X[i,j], G[j])
                Top = dot(XV, S[i,j].T)
                
                tmp1p, tmp1n = pomus(Top)
                
                VSt = dot(G[j], S[i,j].T)
                SVVS = dot(VSt.T, VSt)
                tmp2p, tmp2n = pomus(SVVS)
                
                Bot_neg = dot(Gi, tmp2p)
                Top_neg = dot(Gi, tmp2n)
                
                Top = add(tmp1p, Top_neg)
                Bot = add(tmp1n, Bot_neg)
                
                H_e[i] = add(H_e[i], Top)
                H_d[i] = add(H_d[i], Bot)
            
            if j == target:
                XU = bigdot(X[i,j].T, G[i])
                Top2 = dot(XU, S[i,j])
                tmp4p, tmp4n = pomus(Top2)
                GS = dot(G[i], S[i,j])
                SGGS = dot(GS.T, GS)
                tmp5p, tmp5n = pomus(SGGS)
                
                Top_neg2 = dot(Gi, tmp5n)
                Bot_neg2 = dot(Gi, tmp5p)
                
                Top2 = add(tmp4p, Top_neg2)
                Bot2 = add(tmp4n, Bot_neg2)
                
                H_e[j] = add(H_e[j], Top2)
                H_d[j] = add(H_d[j], Bot2)
    
        T = divide(H_e[target], H_d[target])
        Ts = sqrt(T)
        Gi = multiply(Gi, Ts)

        if check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    print("Timer", str(timer))
    return Gi