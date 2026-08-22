import io
import os
import urllib.parse

import pandas as pd
import requests
from huggingface_hub import HfApi

API = HfApi()
BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SESSION = requests.Session()


def detail_repo(model):
    return "open-llm-leaderboard-old/details_" + model.replace("/", "__")


def list_files(repo):
    info = API.dataset_info(repo)
    return [s.rfilename for s in (info.siblings or [])]


def pick(files, task, min_year=2024):
    hits = [f for f in files
            if f.endswith(".parquet") and ("|%s|" % task) in f]
    if not hits:
        return None
    newest = sorted(hits)[-1]
    year = newest[:4]
    if year.isdigit() and int(year) < min_year:
        return None
    return newest


def fetch_parquet(repo, path, timeout=120):
    url = BASE.format(repo=repo, path=urllib.parse.quote(path))
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))
