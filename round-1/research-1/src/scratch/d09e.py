# d09e: sources metadata part 5 (index 22-24) + follow_ups
SOURCES_META5 = [
 {"index": 22, "page": None, "url": "https://huggingface.co/api/models?search=Qwen3&sort=downloads&limit=60",
  "title": "Hugging Face API model search: Qwen3 (plus abliterated / Llama / Gemma searches)",
  "authors": None, "year": 2026,
  "summary": "Live HF API enumerations on 2026-09-19: Qwen3 family (Qwen3-0.6B/1.7B/4B, Qwen3-4B-Instruct-2507, Qwen3.5/3.6/3.8 families), the abliterated search (huihui-ai, mlabonne, bartowski, mylesgoose, Goekdeniz-Guelmez etc.), per-repo metadata and tree sizes, and tokenizer_config.json chat-template checks (huihui-ai abliterated repos keep separate chat_template.jinja files).",
  "passages": []},
 {"index": 23, "page": None, "url": "https://export.arxiv.org/api/query?search_query=all:%22refusal%20direction%22%20AND%20all:%22score%22&max_results=10",
  "title": "arXiv API saturation queries (2026-09-19)",
  "authors": None, "year": 2026,
  "summary": "arXiv API saturation runs (this URL is the representative first query): refusal-direction+score, safety-metric+few-shot, compliance+activation+refusal, refusal-rate+activation, knowledge-action-gap, label-free+safety, refusal+abliteration, activation-scanner; plus the S3 prior-art arXiv ID verification (RepE 2310.01405, ITI 2306.03341, Tuned Lens 2303.08112, Patchscopes 2401.06102, HarmBench 2402.04249, WMDP 2403.03218, Circuit Breakers 2406.04313, LAT 2403.05030).",
  "passages": []},
 {"index": 24, "page": None, "url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf",
  "title": "Arditi et al. 2024, NeurIPS 2024 proceedings version (search-listing evidence)",
  "authors": ["Andy Arditi", "Oscar Obeso", "Aaquib Syed", "Daniel Paleka", "Nina Panickssery", "Wes Gurnee", "Neel Nanda"], "year": 2024,
  "summary": "Surfaced in general-web search as the NeurIPS 2024 proceedings PDF path: peer-reviewed publication evidence for Arditi et al. 2024 (snippet-level evidence only; page not fetched).",
  "passages": []}
]

FOLLOW_UP_QUESTIONS = [
 "How should the refusal-vs-compliance logit difference at the first generated position be defined per family (vocabulary-specific first tokens of refusal vs compliance phrases)?",
 "Qwen3 instruct defaults to thinking mode (enable_thinking=True per README): should the experiment disable thinking or shift the readout position, and do the base Qwen3 models need the same handling?",
 "Leakage control: the index and the latent probe share the 12+12 probe prompts - what split or CV protocol keeps the probe honest?",
 "Should the held-out behavioral set reuse HarmBench behaviors or custom prompts, and which keyword heuristic (Zou-2023-style list) plus spot-checking protocol labels refusal reliably?",
 "Is a self-abliterated Gemma (Arditi recipe) acceptable as a community-ablation proxy if no safetensors community-abliterated Gemma-2-2B exists (only GGUF bartowski repo found)?",
 "HF_TOKEN exists in the environment but meta-llama / google repos are gated manual (license acceptance required): use unsloth mirrors or attempt gated downloads, and does gating block the cross-family rank-correlation claim?",
 "Which Qwen3 revision to standardize on (original vs -2507), given two parallel abliterated variants (huihui-2507 vs mlabonne) with different template handling?",
 "How many contrast pairs does the scanner baseline need for stable sigma on a 16GB CPU (k, batch size, layer subset)?",
 "Is the 4-prompt index stable (bootstrap CIs over prompt subsets), and what is the minimum zoo size for the rank-correlation claim at rho >= 0.75?",
 "Should bootstrap intervals be over prompts, models, or both, and should RAS (reference-model-based) and LatentBiopsy (200-safe-prompt) be added as additional baselines with their prompt sets reproduced from their repos?"
]