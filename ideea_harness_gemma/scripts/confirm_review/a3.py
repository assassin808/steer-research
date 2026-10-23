exec(open('a1.py').read().split('def sc(')[0])
import pandas as pd
df=pd.read_csv(ROOT+'/results/truthfulqa/dataset/TruthfulQA.csv')
print(df.columns.tolist())
TN=re.compile(r"^\s*(there (is|are|'s) no|there isn't|there aren't|there is not|this is a (bit of a )?(trick|hypothetical|subjective)|it'?s (impossible|difficult) to)",re.I)
for k in ['base','IDEEA_s0','C_b1','C_b2','C_b3','H_b1','K_orig']:
  kk=k+'|mid' if k!='base' else 'base|mid'
  t=[full(gen[kk][q]['response']) for q in QS]; m=np.array([bool(TN.match(x)) for x in t])
  d=ITEM[kk]; print(k,'debunk-template rate %.2f'%m.mean(),'TI(b)|tmpl %.3f'%(d['ti_b'][m].mean() if m.any() else np.nan),'TI(b)|other %.3f'%d['ti_b'][~m].mean())
typ=np.array([df.loc[int(q),'Type'] for q in QS]); best=np.array([str(df.loc[int(q),'Best Answer']) for q in QS])
neg=np.array([bool(re.match(r"^\s*(no\b|nothing|there (is|are) no|none|i have no comment|it is not|you (can't|cannot)|.*(myth|not true|doesn't|don't|does not|do not|never))",b,re.I)) for b in best])
print('neg-best frac',neg.mean())
C=np.mean([ITEM[f'C_b{b}|mid']['ti_b'] for b in (1,2,3)],0); I=ITEM['IDEEA_s0|mid']['ti_b']; Ca=np.mean([ITEM[f'C_b{b}|mid']['ti_a'] for b in (1,2,3)],0); Ia=ITEM['IDEEA_s0|mid']['ti_a']
for nm,m in [('Adversarial',typ=='Adversarial'),('Non-Adv',typ!='Adversarial'),('best=negation/debunk',neg),('best=specific',~neg)]:
  print(nm,m.sum(),'C %.3f I %.3f'%(C[m].mean(),I[m].mean()),'b',bci(C[m]-I[m]),' a',bci(Ca[m]-Ia[m])['mean'])
