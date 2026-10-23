import sys,json; sys.path.insert(0,'/home/yang0531/scratch/ideea_gemma/scripts')
import confirm_grad as G
S=json.load(open('/home/yang0531/scratch/ideea_gemma/confirm_specs.json'))
for a in ['C_orig','H_orig','K_orig','IDEEA_s0']:
  print(a, G.interp([tuple(x) for x in S['sweeps'][a]],0.518), S['runs'][a+'|mid']['spec'])
