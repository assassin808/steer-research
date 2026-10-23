import json,sys,re,numpy as np
sys.path.insert(0,'/home/yang0531/scratch/ideea_gemma/scripts')
import os; os.chdir('/home/yang0531/scratch/ideea_gemma/scripts')
# reuse report code by exec up to ITEM
src=open('confirm_report.py').read().split('json.dump(out, open(ROOT')[0]
__file__="/home/yang0531/scratch/ideea_gemma/scripts/confirm_report.py"; exec(src)
def sc(arm,key,lv='mid'):
    ks={"C":["C_b1","C_b2","C_b3"]}.get(arm,[arm])
    return np.mean([ITEM[f"{a}|{lv}"][key] for a in ks],0)
for c in 'ab':
  for key in ('t_','i_'):
    d=sc('C',key+c)-sc('IDEEA_s0',key+c); print(c,key,bci(d))
  # precision P(T|I)
  for arm in ['base','IDEEA_s0','IDEEA_s1','C_b1','C_b2','C_b3','C_orig','H_b1','K_orig']:
    d=ITEM[arm+'|mid'] if arm!='base' else ITEM['base|mid']
    i=d['i_'+c].astype(bool); print(c,arm,'P(T|I)=%.3f nI=%d  P(T|~I)=%.3f  false-informative=%.3f'%(d['t_'+c][i].mean(),i.sum(),d['t_'+c][~i].mean(),((1-d['t_'+c])*d['i_'+c]).mean()))
# stratify by IDEEA nocomment and C hitlim
I=ITEM['IDEEA_s0|mid']
for c in 'ab':
  dl=sc('C','ti_'+c)-I['ti_'+c]
  print(c,'IDEEA hitlim',I['hitlim'].mean(), 'delta | IDEEA hitlim', dl[I['hitlim']==1].mean(), dl[I['hitlim']==0].mean())
  for a in ['C_b1','C_b2','C_b3']:
    d=ITEM[a+'|mid']; print(a,'TI|hitlim',d['ti_'+c][d['hitlim']==1].mean(),'TI|not',d['ti_'+c][d['hitlim']==0].mean(), d['hitlim'].mean())
  print('IDEEA TI|hitlim',I['ti_'+c][I['hitlim']==1].mean(), I['ti_'+c][I['hitlim']==0].mean())
# KL ratio
for k in ITEM:
  if k.endswith('|mid'): print(k, '%.3f'%(ITEM[k]['kl'].mean()/ITEM['IDEEA_s0|mid']['kl'].mean()), 'totalKL %.1f'%(ITEM[k]['kl']*1).mean())
