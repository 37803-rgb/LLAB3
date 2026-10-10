"""
Генератор учебного (синтетического) текстового датасета HTTP-запросов.

Создаёт data/http_requests.csv со столбцами request, label, timestamp,
label ∈ {Normal, SQL Injection, XSS}. Нужен, чтобы панель работала «из коробки»
(например, при запуске с GitHub), пока не подготовлен реальный открытый датасет
через scripts/prepare_text_dataset.py.

Данные полностью синтетические: шаблоны классических учебных примеров
из документации по безопасности, без реальных целей. Скрипт ничего не
отправляет в сеть.

Запуск:  python scripts/generate_text_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

WORDS = ["python", "book", "shoes", "laptop", "coffee", "travel", "guide", "report",
         "music", "garden", "phone", "course", "recipe", "camera", "winter", "bike",
         "select", "union", "drop", "script", "javascript", "alert", "table", "order",
         "update", "insert", "delete", "from", "where", "or", "and"]
PATHS = ["/search", "/products", "/blog", "/users", "/api/items", "/login", "/cart", "/news"]
PARAMS = ["q", "id", "name", "page", "category", "user", "sort", "item"]
SURNAMES = ["o'reilly", "d'angelo", "o'connor", "mc'donald"]

SQLI_TAILS = [
    "' OR '1'='1", "' OR 1=1--", "' OR 1=1#", "\" OR \"1\"=\"1", "') OR ('1'='1",
    "'; DROP TABLE users;--", "1; DROP TABLE {w};--",
    "' UNION SELECT username, password FROM users--",
    "' UNION SELECT NULL, NULL, NULL--", "1 UNION ALL SELECT 1,2,3",
    "' AND 1=1--", "' AND 1=2--", "1' AND SLEEP(5)--", "1 AND (SELECT COUNT(*) FROM {w})>0",
    "admin'--", "' OR 'a'='a", "1' ORDER BY 3--", "' HAVING 1=1--",
    "'; EXEC xp_cmdshell('dir');--", "1 OR 1=1", "x' OR name LIKE '%{w}%",
]
XSS_PAYLOADS = [
    "<script>alert(1)</script>", "<script>alert('xss')</script>",
    "<img src=x onerror=alert(1)>", "<svg onload=alert(1)>",
    "<body onload=alert(1)>", "<iframe src=\"javascript:alert(1)\"></iframe>",
    "\"><script>alert(document.cookie)</script>", "'><img src=x onerror=alert({n})>",
    "<a href=\"javascript:alert(1)\">click</a>", "<input onfocus=alert(1) autofocus>",
    "<details open ontoggle=alert(1)>", "<div onmouseover=\"alert({n})\">hover</div>",
    "<ScRiPt>alert(1)</ScRiPt>", "javascript:alert(1)", "<marquee onstart=alert(1)>",
]


def _benign(rng: np.random.Generator) -> str:
    kind = rng.integers(0, 5)
    path, param = rng.choice(PATHS), rng.choice(PARAMS)
    w1, w2 = rng.choice(WORDS), rng.choice(WORDS)
    if kind == 0:
        return f"{path}?{param}={w1}+{w2}"
    if kind == 1:
        return f"{path}?{param}={rng.integers(1, 5000)}&page={rng.integers(1, 40)}"
    if kind == 2:
        return f"{path}?{param}={quote(rng.choice(SURNAMES))}+{w1}"
    if kind == 3:
        return f"{path}/{w1}-{w2}/{rng.integers(1, 999)}"
    return f"{path}?{param}={w1}%20{w2}&sort={rng.choice(['asc', 'desc', 'price', 'date'])}"


def _sqli(rng: np.random.Generator) -> str:
    tail = rng.choice(SQLI_TAILS).format(w=rng.choice(["users", "orders", "items"]))
    path, param = rng.choice(PATHS), rng.choice(PARAMS)
    prefix = str(rng.integers(1, 500)) if rng.random() < 0.5 else rng.choice(WORDS)
    payload = prefix + tail
    if rng.random() < 0.4:  # часть запросов в URL-кодировке
        payload = quote(payload)
    return f"{path}?{param}={payload}"


def _xss(rng: np.random.Generator) -> str:
    payload = rng.choice(XSS_PAYLOADS).format(n=rng.integers(1, 99))
    path, param = rng.choice(PATHS), rng.choice(PARAMS)
    if rng.random() < 0.4:
        payload = quote(payload)
    return f"{path}?{param}={payload}"


def make_text(n_per_class: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = ([(_benign(rng), "Normal") for _ in range(n_per_class)]
            + [(_sqli(rng), "SQL Injection") for _ in range(n_per_class)]
            + [(_xss(rng), "XSS") for _ in range(n_per_class)])
    df = pd.DataFrame(rows, columns=[config.TEXT_COL, config.TEXT_LABEL_COL])
    df = df.drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)
    ts = pd.Timestamp("2026-09-01 08:00") + pd.to_timedelta(
        np.sort(rng.uniform(0, 8 * 3600, len(df))), unit="s")
    df[config.TEXT_TIME_COL] = ts.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = make_text()
    df.to_csv(config.TEXT_DATA_PATH, index=False)
    print(f"Записано {len(df)} строк в {config.TEXT_DATA_PATH}")
    print(df[config.TEXT_LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
