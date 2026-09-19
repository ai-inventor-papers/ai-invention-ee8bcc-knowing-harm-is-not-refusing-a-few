import sys, json
sys.path.insert(0, '.')
from behavior import strip_thinking, classify_refusal

j = json.load(open('caches/per_model/Qwen__Qwen3-0.6B.json'))
t = j['behavior']['exemplars'][0]['text']
print('len', len(t))
print('bytes8:', t[:8].encode('utf-8'))
print('chars:', [(i, ord(c), ascii(c)) for i, c in enumerate(t[:10])])
head = t[:40].lstrip()
print('head chars:', [(i, ord(c), ascii(c)) for i, c in enumerate(head[:10])])
ts, was = strip_thinking(t)
print('was', was, 'ts_len', len(ts))
print('strip result head:', ascii(ts[:40]))