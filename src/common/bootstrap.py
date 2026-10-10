"""
Автоматическая подготовка данных и моделей при первом запуске.

Зачем: в GitHub-репозиторий не кладут data/*.csv, models/*.joblib и outputs/
(они большие и воспроизводимые). Поэтому при запуске панели на хостинге
(Streamlit Community Cloud) этих файлов нет. ensure_artifacts() генерирует
синтетические учебные данные, обучает модели и сохраняет результаты —
один раз; при следующих запусках ничего не пересчитывается.

Если вы положили собственные data/network_traffic.csv или data/http_requests.csv,
они НЕ перезаписываются.
"""
from __future__ import annotations

import config


def _missing() -> dict[str, bool]:
    return {
        "network": not (config.ISOFOREST_MODEL_PATH.exists()
                        and config.RF_MODEL_PATH.exists()
                        and config.NETWORK_EVENTS_PATH.exists()),
        "nlp": not (config.NLP_MODEL_PATH.exists()
                    and config.NLP_NB_MODEL_PATH.exists()
                    and config.HTTP_EVENTS_PATH.exists()),
    }


def artifacts_ready() -> bool:
    return not any(_missing().values())


def ensure_artifacts() -> list[str]:
    """Создаёт недостающие данные/модели. Возвращает список выполненных шагов."""
    import json

    todo = _missing()
    steps: list[str] = []
    if not any(todo.values()):
        return steps

    import train_all
    for d in (config.DATA_DIR, config.MODELS_DIR, config.OUTPUTS_DIR, config.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    if todo["network"]:
        if not config.NETWORK_DATA_PATH.exists():
            from scripts.generate_network_data import make_network
            make_network(12_000, seed=1, start="2026-09-01 08:00").to_csv(
                config.NETWORK_DATA_PATH, index=False)
            steps.append("сгенерирован синтетический сетевой набор")
        metrics["network"] = train_all.run_network()
        steps.append("обучен сетевой модуль")

    if todo["nlp"]:
        if not config.TEXT_DATA_PATH.exists():
            from scripts.generate_text_data import make_text
            make_text().to_csv(config.TEXT_DATA_PATH, index=False)
            steps.append("сгенерирован синтетический текстовый набор")
        metrics["nlp"] = train_all.run_nlp()
        steps.append("обучен NLP-модуль")

    config.METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return steps
