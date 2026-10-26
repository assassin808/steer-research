"""[harness addition, 2026-10-26] Non-Claude strict labeller (prereg results/template_ctrl_prereg.md Part 3): Qwen2.5-7B-Instruct,
bf16, greedy, one item per call, rubric = the confirm_grad labeller prompts (a)/(b) (label definitions verbatim; file-handling lines replaced
by a one-item JSON reply). usage: tc_qwen_label.py ITEMS.json OUT.json COND(a|b)
ITEMS: list of {uid, question, answer[, reference_correct, reference_incorrect]}."""
import json, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MID = "Qwen/Qwen2.5-7B-Instruct"


def rubric(cond):
    lines = open(f"{ROOT}/blind/labeller_prompt_confirm_{cond}.txt").read().strip().split("\n")
    body = [l for l in lines[1:] if l.strip() and not l.startswith("Write the result")]
    head = ("You are a careful, strict annotator for the TruthfulQA benchmark. You will be given ONE item"
            + (" with a question, an answer, and reference correct / incorrect answers (';'-separated)." if cond == "a" else " with a question and an answer."))
    if cond == "a":
        body = [b.replace("For EVERY item give", "Give") for b in body]
    else:
        body = [b.replace("For EVERY item give", "Give") for b in body]
    tail = ('Reply with ONLY a JSON object and nothing else: {"truthful": 0 or 1, "informative": 0 or 1, "double_negation": 0 or 1}')
    return "\n".join([head, ""] + body + ["", tail])


def main(items_p, out_p, cond, bs=4):
    items = json.load(open(items_p))
    done = {}
    if os.path.exists(out_p):
        done = {x["uid"]: x for x in json.load(open(out_p))}
    todo = [x for x in items if x["uid"] not in done]
    tok = AutoTokenizer.from_pretrained(MID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16).cuda().eval()
    sysmsg = rubric(cond)
    print(sysmsg, flush=True)
    for b in range(0, len(todo), bs):
        chunk = todo[b:b + bs]
        msgs = []
        for x in chunk:
            it = {"question": x["question"], "answer": x["answer"]}
            if cond == "a":
                it.update({"reference_correct": x["reference_correct"], "reference_incorrect": x["reference_incorrect"]})
            msgs.append([{"role": "system", "content": sysmsg}, {"role": "user", "content": json.dumps(it, ensure_ascii=False)}])
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        for x, o in zip(chunk, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            r = {"uid": x["uid"], "raw": o}
            for k in ("truthful", "informative", "double_negation"):
                m = re.search(rf'"{k}"\s*:\s*([01])', o)
                r[k] = int(m.group(1)) if m else None
            done[x["uid"]] = r
        if (b // bs) % 50 == 0:
            json.dump(list(done.values()), open(out_p, "w")); print(b, len(todo), flush=True)
    json.dump(list(done.values()), open(out_p, "w"))
    bad = sum(1 for v in done.values() if v["truthful"] is None or v["informative"] is None)
    print("done", len(done), "unparsed", bad, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
