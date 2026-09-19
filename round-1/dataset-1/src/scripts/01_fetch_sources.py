#!/usr/bin/env python3
"""01_fetch_sources.py — download all source datasets for the safety-screening bundle.

Sources (HuggingFace Hub + GitHub), pinned revisions, raw files only
(no datasets-server, no script execution), metadata cached to
sources/hf_api_cache.json, provenance written to sources/provenance.json.

Zero LLM API calls. Deterministic. uv venv + loguru conventions.
"""

import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from loguru import logger

WORKSPACE = Path(__file__).resolve().parents[1]
RAW_DIR = WORKSPACE / "sources" / "raw"
CACHE_FILE = WORKSPACE / "sources" / "hf_api_cache.json"
PROV_FILE = WORKSPACE / "sources" / "provenance.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

HF_API = "https://huggingface.co/api"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "aii-dataset-curation/1.0",
        **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
    }
)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            logger.warning("cache corrupted; starting fresh")
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=1))


def hf_metadata(path: str, cache_key: str, retries: int = 5) -> dict:
    """GET an HF API path with cache + exponential backoff."""
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]
    url = f"{HF_API}/{path}"
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                cache[cache_key] = data
                _save_cache(cache)
                return data
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                logger.warning(f"rate-limited on {url}; sleeping {wait}s")
                time.sleep(wait)
                continue
            logger.warning(f"HTTP {resp.status_code} on {url}")
            if resp.status_code == 404:
                cache[cache_key] = {}
                _save_cache(cache)
                return {}
            time.sleep(2**attempt)
        except requests.RequestException as exc:
            logger.warning(f"request error {url}: {exc}")
            time.sleep(2**attempt)
    logger.error(f"giving up on {url}")
    return {}


def hf_dataset_info(dataset_id: str) -> dict:
    """Full dataset repo metadata: sha, tags, siblings, gated, downloads."""
    info = hf_metadata(f"datasets/{dataset_id}?full=true", f"ds:{dataset_id}")
    if not info:
        return {}
    siblings = [s for s in info.get("siblings", []) if not s.get("rfilename", "").startswith(".")]
    return {
        "id": dataset_id,
        "sha": info.get("sha"),
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
        "gated": info.get("gated", False),
        "tags": info.get("tags", []),
        "siblings": siblings,
    }


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with SESSION.get(url, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        total = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
    return total


def download_hf_file(dataset_id: str, sha: str, rfilename: str) -> Path:
    safe = dataset_id.replace("/", "__")
    dest = RAW_DIR / safe / rfilename
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"exists, skip: {dest.name}")
        return dest
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{sha}/{rfilename}"
    logger.info(f"downloading {dataset_id}/{rfilename}")
    n = _download(url, dest)
    logger.info(f"  {dest.name}: {n / 1e6:.1f} MB")
    return dest


def download_github_file(repo: str, commit: str, path: str) -> Path:
    dest = RAW_DIR / f"{repo.replace('/', '__')}__{Path(path).name}"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"exists, skip: {dest.name}")
        return dest
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    logger.info(f"downloading github {repo}/{path}")
    n = _download(url, dest)
    logger.info(f"  {dest.name}: {n / 1e6:.1f} MB")
    return dest


def pick_files(key: str, names: list) -> list:
    """Choose which repo files to download per dataset (size-aware, observed names)."""
    if key == "PKU-Alignment/BeaverTails":
        return [n for n in names if n.startswith("round0/30k/")]  # 30k train+test gz
    if key == "databricks/databricks-dolly-15k":
        return [n for n in names if n.endswith(".jsonl")][:1]
    if key == "tatsu-lab/alpaca_eval":
        return [n for n in names if n == "alpaca_eval_gpt4_baseline.json"]
    if key == "yahma/alpaca-cleaned":
        return [n for n in names if n == "alpaca_data_cleaned.json"]
    if key == "Anthropic/hh-rlhf":
        # helpful side = benign prompts; red-team side = harmful attack prompts.
        return [n for n in names if n in ("helpful-base/train.jsonl.gz", "red-team-attempts/red_team_attempts.jsonl.gz")]
    if key == "fka/prompts.chat":
        return [n for n in names if n == "prompts.csv"]
    if key == "mlabonne/harmful_behaviors":
        return [n for n in names if n == "data/train-00000-of-00001.parquet"]
    return []


@logger.catch(reraise=True)
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    provenance = {
        "curation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": [],
    }

    hf_ids = [
        "PKU-Alignment/BeaverTails",
        "databricks/databricks-dolly-15k",
        "tatsu-lab/alpaca_eval",
        "yahma/alpaca-cleaned",
        "Anthropic/hh-rlhf",
        "fka/prompts.chat",
        "mlabonne/harmful_behaviors",
    ]

    dl_jobs = []
    for key in hf_ids:
        info = hf_dataset_info(key)
        if not info:
            provenance["sources"].append({"id": key, "kind": "hf_dataset", "status": "NOT_FOUND"})
            logger.error(f"{key}: NOT FOUND on HF API")
            continue
        license_tags = [t for t in info.get("tags", []) if "license" in t]
        logger.info(
            f"{key}: sha={str(info.get('sha'))[:12]} downloads={info.get('downloads')} "
            f"likes={info.get('likes')} gated={info.get('gated')} license={license_tags}"
        )
        picks = pick_files(key, [s["rfilename"] for s in info["siblings"]])
        logger.info(f"  picked files: {picks}")
        for p in picks:
            dl_jobs.append(("hf", key, info["sha"], p))
        provenance["sources"].append(
            {
                "id": key,
                "kind": "hf_dataset",
                "status": "downloaded" if picks else "metadata_only",
                "revision_sha": info["sha"],
                "downloads": info["downloads"],
                "likes": info["likes"],
                "gated": info.get("gated"),
                "license_tags": license_tags,
                "files": picks,
            }
        )

    # GitHub sources (pinned commit)
    github_plan = {
        "verazuo/jailbreak_llms": [
            ("data/prompts/jailbreak_prompts_2023_05_07.csv", "attack prompts (jailbreak=true subset)"),
            ("data/forbidden_question/forbidden_question_set.csv", "forbidden questions with categories"),
            ("data/prompts/regular_prompts_2023_05_07.csv", "regular (benign) user prompts"),
        ],
    }
    for repo, file_plan in github_plan.items():
        try:
            rsp = SESSION.get(f"https://api.github.com/repos/{repo}", timeout=30)
            repo_meta = rsp.json() if rsp.status_code == 200 else {}
            branch = repo_meta.get("default_branch", "main")
            rsp2 = SESSION.get(f"https://api.github.com/repos/{repo}/commits?per_page=1", timeout=30)
            commit = rsp2.json()[0]["sha"] if rsp2.status_code == 200 else branch
        except Exception as exc:
            logger.warning(f"github metadata lookup failed for {repo}: {exc}")
            branch, commit = "main", "main"
        for path, note in file_plan:
            dl_jobs.append(("github", repo, commit, path))
        provenance["sources"].append(
            {
                "id": repo,
                "kind": "github_repo",
                "status": "downloaded",
                "revision_sha": commit,
                "branch": branch,
                "license": "MIT (verazuo/jailbreak_llms LICENSE file)",
                "files": [p for p, _ in file_plan],
                "note": "JailbreakBench jailbreak_llms dataset (official GitHub mirror)",
            }
        )

    # parallel CDN downloads
    ok = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = []
        for kind, key, rev, path in dl_jobs:
            if kind == "hf":
                futs.append(pool.submit(download_hf_file, key, rev, path))
            else:
                futs.append(pool.submit(download_github_file, key, rev, path))
        for fut in as_completed(futs):
            try:
                p = fut.result()
                ok += 1
            except Exception as exc:
                logger.error(f"download failed: {exc}")

    PROV_FILE.write_text(json.dumps(provenance, indent=1))
    logger.info(
        f"done in {time.time() - t0:.0f}s; {ok}/{len(dl_jobs)} files OK; provenance -> {PROV_FILE}"
    )


if __name__ == "__main__":
    main()