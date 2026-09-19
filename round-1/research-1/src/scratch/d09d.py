# d09d: sources metadata part 4 (index 16-21)
SOURCES_META4 = [
 {"index": 16, "page": "hildebrant_abs", "url": "https://arxiv.org/abs/2501.08145",
  "title": "Refusal Behavior in Large Language Models: A Nonlinear Perspective",
  "authors": ["Fabian Hildebrandt", "Andreas Maier", "Patrick Krauss", "Achim Schilling"], "year": 2025,
  "summary": "Six LLMs / three families; PCA, t-SNE, UMAP; refusal mechanisms nonlinear, multidimensional, vary by architecture and layer; no scalar metric, no abliterated class.",
  "passages": [
   {"start": "This paper investigates refusal behavior across six LLMs from three architectural families", "end": "We challenge the assumption"},
   {"start": "Our results reveal that refusal mechanisms exhibit nonlinear, multidimensional characteristics", "end": "AI deployment strategies."}]},
 {"index": 17, "page": "jiang_abs", "url": "https://arxiv.org/abs/2606.08044",
  "title": "When Behavioral Safety Evaluation Fails: A Representation-Level Perspective",
  "authors": ["Enyi Jiang", "Anders Gjolbye", "Yibo Jacky Zhang", "Sanmi Koyejo"], "year": 2026,
  "summary": "Audit gap + dissociated models (Gemma 2 2B, Llama 3.2 3B, Qwen 2.5 3B); LVS 2.5-3.1x higher; bounded latent attack 54-86% compliance vs 3-48% bases; random perturbations <=12%; harmful fine-tuning reaches high compliance in 5 gradient steps vs 10-25 for bases.",
  "passages": [
   {"start": "Every static audit we run gives the dissociated model the same verdict as its base", "end": "cannot tell it from the base."},
   {"start": "At the targeted mid layer the dissociated models score 2.5 to 3.1 times higher LVS than their bases", "end": "10 to 25."}]},
 {"index": 18, "page": "refusal_tokens_abs", "url": "https://arxiv.org/abs/2412.06748",
  "title": "Refusal Tokens: A Simple Way to Calibrate Refusals in Large Language Models",
  "authors": ["Neel Jain", "Aditya Shrivastava", "Chenyang Zhu", "Daben Liu", "Alfy Samuel", "Ashwinee Panda", "Anoop Kumar", "Micah Goldblum", "Tom Goldstein"], "year": 2024,
  "summary": "Training-time calibration tokens prepended to responses; inference-time probability steering controls refusal rates; NOT a per-model refusal detection baseline.",
  "passages": [
   {"start": "we propose refusal tokens, one such token for each refusal category", "end": "selectively intervening during generation."}]},
 {"index": 19, "page": "refusal_before_decoding_abs", "url": "https://arxiv.org/abs/2605.28553",
  "title": "Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations",
  "authors": ["Matteo Gioele Collu", "Riccardo Conte", "Alberto Giaretta", "Denis Kleyko", "Mauro Conti", "Matteo Zavatteri", "Roberto Confalonieri"], "year": 2026,
  "summary": "Refusal linearly decodable at every transformer block before the final layer; probe-guided Mechanistic AutoDAN cuts per-iteration search time up to 72%; per-prompt jailbreak search, not a per-model safety metric.",
  "passages": [
   {"start": "we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding", "end": "before output generation."}]},
 {"index": 20, "page": "joad_abs", "url": "https://arxiv.org/abs/2602.02132",
  "title": "There Is More to Refusal in Large Language Models than a Single Direction",
  "authors": ["Faaiz Joad", "Majd Hawasly", "Sabri Boughorbel", "Nadir Durrani", "Husrev Taha Sencar"], "year": 2026,
  "summary": "EMNLP-2026 counterpoint: refusal categories have geometrically distinct directions; directions affect HOW not WHETHER the model refuses; SAE analysis finds a reusable core of shared refusal latents plus style/domain-specific latents.",
  "passages": [
   {"start": "refusal behaviors correspond to geometrically distinct directions in activation space", "end": "how it refuses."}]},
 {"index": 21, "page": "qwen3_readme", "url": "https://huggingface.co/Qwen/Qwen3-4B",
  "title": "Qwen/Qwen3-4B model card (README)",
  "authors": None, "year": 2025,
  "summary": "README confirms enable_thinking=True is the DEFAULT ('Switches between thinking and non-thinking modes. Default is True.') - the first-generated-token pitfall for position-0 refusal logit reads. Qwen3-4B is a 3.9B-parameter model in the 4B class.",
  "passages": [
   {"start": "enable_thinking=True", "end": "Default is True."}]}
]