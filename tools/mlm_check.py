#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# mlm_check: does a .mlm image behave like the original model?
# Run on a PC with torch and transformers (pip install torch transformers).
#
#   mlm_check.py HF_DIR model.mlm "prompt" [steps]
#
# 1. transformers generates greedily from the prompt (float32);
# 2. the same token ids go through ref/mlm_run (the board's arithmetic);
# 3. per position: does the top token agree, and how far apart are the
#    logits of the top five?
# A converted model that agrees on most positions and stays within a few
# tenths of a logit is behaving; quantisation (--bits 4) costs agreement
# first on near-ties.

import os, re, subprocess, sys

def main():
    hf, mlm, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32)
    model.eval()
    ids = tok(prompt, return_tensors='pt').input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=steps, do_sample=False)
        logits = model(out).logits[0]
    seq = out[0].tolist()
    ref_lines = []
    for p in range(len(seq) - 1):
        lg = logits[p]
        top = torch.topk(lg, 5)
        ref_lines.append('pos %d in %d:' % (p, seq[p]) +
                         ''.join(' %d:%.4f' % (int(i), float(v)) for v, i in zip(top.values, top.indices)))
    here = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(here, '..', 'ref', 'mlm_run')
    # mlm_run adds BOS itself when the image says so; pass the rest
    feed = seq[1:] if (tok.bos_token_id is not None and seq[0] == tok.bos_token_id) else seq
    r = subprocess.run([run, mlm, '-x', ' '.join(str(t) for t in feed), '-n', '0', '-d'],
                       capture_output=True, text=True)
    mine = [l for l in r.stderr.splitlines() if l.startswith('pos')]
    open('/tmp/mlm_check_hf.txt', 'w').write('\n'.join(ref_lines) + '\n')
    open('/tmp/mlm_check_mlm.txt', 'w').write('\n'.join(mine) + '\n')
    print('transformers:', tok.decode(seq))
    subprocess.run([sys.executable, os.path.join(here, '..', 'ref', 'check', 'cmp_logits.py'),
                    '/tmp/mlm_check_hf.txt', '/tmp/mlm_check_mlm.txt'])

if __name__ == '__main__':
    main()
