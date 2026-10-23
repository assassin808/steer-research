import glob, numpy as np, re
R="/home/yang0531/scratch/ideea_gemma/runs_ng/"
zs=[np.load(f,allow_pickle=True) for f in sorted(glob.glob(R+"train_L13_shard*.npz"))]
U=np.concatenate([z["U"] for z in zs]).astype(np.float64); q=np.concatenate([z["qidx"] for z in zs]); tx=np.concatenate([z["texts"] for z in zs])
ref=np.array([bool(re.match(r"\s*(A:\s*)?I have no comment",str(t))) for t in tx]); print("n",len(U),"refusal frac",ref.mean())
dif=[U[(q==i)&ref].mean(0)-U[(q==i)&~ref].mean(0) for i in np.unique(q) if ((q==i)&ref).any() and ((q==i)&~ref).any()]
rd=np.mean(dif,0)
Z=np.load(R+"directions_fix_L13.npz"); G=Z["G"]; w,V=np.linalg.eigh(G); w,V=w[::-1],V[:,::-1]
print("top eig",w[:4],"tr/d",w.mean(),"cos(ref axis, eig1..3)",[round(abs(V[:,k]@rd)/np.linalg.norm(rd),3) for k in range(3)])
I=np.load(R+"bridge/dirs/resid.npz")
for n,v in [("K_inf",Z["efsum"]),("K_10",Z["nfsum10"]),("K_1",Z["nfsum1"]),("K_0.1",Z["nfsum0.1"]),("I_inf",I["I_inf"]),("I_10",I["I_10"]),("J_inf",I["J_inf"])]:
    fn=np.sqrt(v@G@v); print("%-6s dlogodds(ref-ans)/FisherNorm %+.3f  frac energy top1 %.3f"%(n,v@rd/fn,w[0]*(V[:,0]@v)**2/fn**2))
