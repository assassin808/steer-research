exec(open('a1.py').read().split('def sc(')[0])
K=json.load(open(ROOT+'/blind_confirm_keys/key_strip.json'))['items']
L={}
for f in glob.glob(ROOT+'/blind_confirm/strip/labels_*.json'):
  for x in json.load(open(f)): L[int(x['id'])]=(x['truthful'],x['informative'])
print(len(K),len(L))
Kb={it['id']:it for it in json.load(open(ROOT+'/blind_confirm_keys/key_b.json'))['items']}
st={it['q']:L[it['id']] for it in K if it['kind']=='stripped'}
ctrl=[it for it in K if it['kind']=='control']
print('controls that are IDEEA_s0|mid answers', sum('IDEEA_s0|mid' in Kb[c['orig_id']]['runs'] for c in ctrl), 'same q as a stripped item', sum(c['q'] in st for c in ctrl))
I=ITEM['IDEEA_s0|mid']; ti=I['ti_b'].copy()
for j,q in enumerate(QS):
  if q in st: ti[j]=st[q][0]*st[q][1]
C=np.mean([ITEM[f'C_b{b}|mid']['ti_b'] for b in (1,2,3)],0)
print('IDEEA stripped TI',ti.mean(), bci(C-ti))
# drift-corrected alt: stripped vs old on those 272
m=np.array([q in st for q in QS]); print('old TI on prefixed',I['ti_b'][m].mean(),'new',ti[m].mean(), 'info old/new',I['i_b'][m].mean(), np.mean([st[q][1] for q in st]))
