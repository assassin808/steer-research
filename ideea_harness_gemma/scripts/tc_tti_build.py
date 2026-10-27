"""[harness addition, 2026-10-26, POST HOC exploratory] strict TruthfulQA rubric (confirm (a) prompt, gold answer + aliases as
reference_correct; prompt blind/labeller_prompt_trivia_ti.txt) on 150 random TriviaQA questions x base, IDEEA, ITI, C_b1-3.
-> blind_tc/tti/chunk_*.json, key blind_tc_keys/key_tti.json (originally written as trivia_ti / key_trivia_ti and renamed)."""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.expanduser('~/scratch/tqa_ctrl/data/triviaqa400.json')))
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith('A:') else t.strip())
F = {'base|mid': 'base', 'IDEEA_s0|mid': 'ideea_4_24_min_perp_2', 'ITI|mid': 'iti_5.3933_16', 'C_b1|mid': 'br_C_inf_b1_2.0826',
     'C_b2|mid': 'br_C_inf_b2_2.0045', 'C_b3|mid': 'br_C_inf_b3_2.1361'}
R = {k: {r['key']: full(r['response']) for r in json.load(open(f'{ROOT}/results_gm/triviaqa_tc/{v}.json'))['records']} for k, v in F.items()}
qs = sorted(np.random.RandomState(777).choice(400, 150, replace=False).tolist())
u = {}
for k in F:
    for q in qs:
        u.setdefault((q, R[k][q]), set()).add(k)
items = sorted(u.items(), key=lambda x: (x[0][0], x[0][1]))
rs = np.random.RandomState(778); perm = rs.permutation(len(items)); ids = rs.choice(np.arange(100000, 999999), len(items), replace=False)
rows, key = [], []
for j, k in enumerate(perm):
    (q, a), runs = items[k]; x = data[q]
    al = [x['answer']] + [y for y in sorted(set(x['aliases']), key=len) if y != x['answer'] and y == y.strip()][:8]
    rows.append({'id': int(ids[j]), 'question': x['question'], 'answer': a, 'reference_correct': '; '.join(al), 'reference_incorrect': ''})
    key.append({'id': int(ids[j]), 'q': q, 'answer': a, 'runs': sorted(runs)})
os.makedirs(f'{ROOT}/blind_tc/tti', exist_ok=True)
n = int(np.ceil(len(rows) / 220)); b = np.linspace(0, len(rows), n + 1).astype(int)
for c in range(n):
    json.dump(rows[b[c]:b[c + 1]], open(f'{ROOT}/blind_tc/tti/chunk_{c:02d}.json', 'w'), indent=0)
json.dump({'qs': qs, 'runs': F, 'items': key}, open(f'{ROOT}/blind_tc_keys/key_tti.json', 'w'))
print(len(rows), n)
