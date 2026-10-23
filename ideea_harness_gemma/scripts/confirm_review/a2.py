exec(open('a1.py').read().split('def sc(')[0])
import pandas as pd
df=pd.read_csv(ROOT+'/results/truthfulqa/dataset/TruthfulQA.csv')
for a in ['C_b1','C_b2','C_b3','C_orig']:
  d=ITEM[a+'|mid']; m=(d['t_b']==0)&(d['i_b']==0); print(a,'T0I0',m.mean(),'of which dneg_rx',d['dneg_rx'][m].mean(), 'dnl',d['dnl_b'][m].mean())
I=ITEM['IDEEA_s0|mid']; C=ITEM['C_b2|mid']
rs=np.random.RandomState(1)
win=[j for j in range(len(QS)) if C['ti_b'][j]==1 and I['ti_b'][j]==0]
lose=[j for j in range(len(QS)) if C['ti_b'][j]==0 and I['ti_b'][j]==1]
print(len(win),len(lose))
pick=list(rs.choice(win,14,replace=False))+list(rs.choice(lose,8,replace=False))
oth=[j for j in range(len(QS)) if C['t_b'][j]==1 and C['i_b'][j]==1]
pick+=list(rs.choice(oth,8,replace=False))
for j in pick:
  q=QS[j]; i=int(q)
  print('=== q',q, df.loc[i,'Question']); print('  REF+:',df.loc[i,'Correct Answers'][:200]); print('  REF-:',df.loc[i,'Incorrect Answers'][:150])
  for nm,d in (('IDEEA',I),('C_b2',C)):
    print('  [%s] a=%d%d b=%d%d :: %s'%(nm,d['t_a'][j],d['i_a'][j],d['t_b'][j],d['i_b'][j], full(gen[nm.replace('IDEEA','IDEEA_s0')+'|mid'][q]['response']).replace('\n',' ')[:330]))
