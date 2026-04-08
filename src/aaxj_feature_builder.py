from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"

FF3_PATH = DATASET_DIR / "ff_asia_pacific_3factors_daily.csv"
MOM_PATH = DATASET_DIR / "mom_asia_pacific_daily.csv"
OUTPUT_PATH = DATASET_DIR / "aaxj_features_final_test.csv"

AAXJ_PAGE_URL = "https://www.ishares.com/us/products/239601/ishares-msci-all-country-asia-ex-japan-etf"
AAXJ_NAME = "iShares MSCI All Country Asia ex Japan ETF"
AAXJ_ISIN = "US4642881829"
AAXJ_ALLFUNDS_ID = "AAXJ_ETF"
PRACTICE_START = date(2016, 1, 5)
PRACTICE_END = date(2021, 7, 16)
SQRT_252 = math.sqrt(252.0)


@dataclass
class SeriesPoint:
    dt: date
    value: float


def parse_mixed_date(raw: str) -> date:
    raw = str(raw).strip()
    if "-" in raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return datetime.strptime(raw, "%Y%m%d").date()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    avg = mean(values)
    var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    if not returns:
        return float("nan")

    equity = 1.0
    peak = 1.0
    min_drawdown = 0.0

    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        min_drawdown = min(min_drawdown, drawdown)

    return min_drawdown


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Sistema singular en la regresion OLS.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def ols_coefficients(feature_rows: Sequence[Sequence[float]], y_values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if len(y_values) < 60:
        nan = float("nan")
        return (nan, nan, nan, nan, nan)

    cols = 5  # intercepto + 4 factores
    xtx = [[0.0] * cols for _ in range(cols)]
    xty = [0.0] * cols

    for factors, y in zip(feature_rows, y_values):
        row = [1.0] + list(factors)
        for i in range(cols):
            xty[i] += row[i] * y
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]

    try:
        coef = solve_linear_system(xtx, xty)
    except ValueError:
        nan = float("nan")
        return (nan, nan, nan, nan, nan)

    return tuple(float(v) for v in coef)


def load_factors() -> Dict[date, Dict[str, float]]:
    ff3_rows: Dict[date, Dict[str, float]] = {}
    mom_rows: Dict[date, float] = {}

    with FF3_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dt = parse_mixed_date(row["Date"])
            ff3_rows[dt] = {
                "Mkt-RF": float(row["Mkt-RF"]) / 100.0,
                "SMB": float(row["SMB"]) / 100.0,
                "HML": float(row["HML"]) / 100.0,
                "RF": float(row["RF"]) / 100.0,
            }

    with MOM_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dt = parse_mixed_date(row["Date"])
            mom_rows[dt] = float(row["WML"]) / 100.0

    factors: Dict[date, Dict[str, float]] = {}
    for dt, ff3 in ff3_rows.items():
        if dt in mom_rows:
            merged = ff3.copy()
            merged["WML"] = mom_rows[dt]
            factors[dt] = merged

    return dict(sorted(factors.items()))


def fetch_aaxj_total_return_series() -> List[SeriesPoint]:
    response = requests.get(
        AAXJ_PAGE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    match = re.search(r"var performanceData = \[(.*?)\];", response.text, re.S)
    if not match:
        raise RuntimeError("No se ha encontrado performanceData en la pagina oficial de AAXJ.")

    payload = match.group(1)
    points = re.findall(
        r"Date\.UTC\((\d+),(\d+),(\d+)\),y:Number\(\(([^)]+)\)\.toFixed\(2\)\)",
        payload,
    )

    if not points:
        raise RuntimeError("performanceData existe, pero no se han podido extraer observaciones.")

    series = [
        SeriesPoint(
            dt=date(int(year), int(month) + 1, int(day)),
            value=float(value),
        )
        for year, month, day, value in points
    ]

    filtered = [point for point in series if PRACTICE_START <= point.dt <= PRACTICE_END]
    if len(filtered) < 252:
        raise RuntimeError("La serie oficial de AAXJ no cubre suficientes observaciones en la ventana de la practica.")

    return filtered


def compute_daily_returns(series: Sequence[SeriesPoint]) -> List[Tuple[date, float]]:
    returns: List[Tuple[date, float]] = []
    for prev, curr in zip(series[:-1], series[1:]):
        if prev.value == 0:
            continue
        ret = curr.value / prev.value - 1.0
        returns.append((curr.dt, ret))
    return returns


def format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return repr(value)


def build_aaxj_features(output_path: Path = OUTPUT_PATH) -> Dict[str, object]:
    factors = load_factors()
    series = fetch_aaxj_total_return_series()
    daily_returns = compute_daily_returns(series)

    aligned_returns: List[float] = []
    aligned_excess: List[float] = []
    regressor_rows: List[Tuple[float, float, float, float]] = []
    matched_dates: List[date] = []

    for dt, ret in daily_returns:
        factor_row = factors.get(dt)
        if factor_row is None:
            continue

        matched_dates.append(dt)
        aligned_returns.append(ret)
        aligned_excess.append(ret - factor_row["RF"])
        regressor_rows.append(
            (
                factor_row["Mkt-RF"],
                factor_row["SMB"],
                factor_row["HML"],
                factor_row["WML"],
            )
        )

    if len(aligned_returns) < 120:
        raise RuntimeError("AAXJ no tiene suficientes observaciones alineadas con los factores para calcular features.")

    mean_r = mean(aligned_returns)
    std_r = sample_std(aligned_returns)
    ann_return = (1.0 + mean_r) ** 252 - 1.0
    ann_vol = std_r * SQRT_252 if not math.isnan(std_r) else float("nan")
    ann_excess = mean(aligned_excess) * 252

    negative_returns = [ret for ret in aligned_returns if ret < 0]
    downside_std = sample_std(negative_returns) * SQRT_252 if len(negative_returns) > 1 else float("nan")

    sharpe = ann_excess / ann_vol if ann_vol and not math.isnan(ann_vol) else float("nan")
    sortino = ann_excess / downside_std if downside_std and not math.isnan(downside_std) else float("nan")
    max_drawdown = max_drawdown_from_returns(aligned_returns)

    if not math.isnan(std_r) and std_r > 0:
        z_scores = [(ret - mean_r) / std_r for ret in aligned_returns]
        ret_skew = mean([z ** 3 for z in z_scores])
        ret_kurt = mean([z ** 4 for z in z_scores]) - 3.0
    else:
        ret_skew = float("nan")
        ret_kurt = float("nan")

    alpha_daily, beta_mkt, beta_smb, beta_hml, beta_wml = ols_coefficients(regressor_rows, aligned_excess)

    name_lc = AAXJ_NAME.lower()
    feature_row: Dict[str, object] = {
        "allfunds_id": AAXJ_ALLFUNDS_ID,
        "isin": AAXJ_ISIN,
        "name_last": AAXJ_NAME,
        "ticker": "AAXJ",
        "source_url": AAXJ_PAGE_URL,
        "source_series": "iShares performanceData (Growth of Hypothetical $10,000)",
        "window_start_requested": PRACTICE_START.isoformat(),
        "window_end_requested": PRACTICE_END.isoformat(),
        "n_obs": len(aligned_returns),
        "start_date": matched_dates[0].isoformat(),
        "end_date": matched_dates[-1].isoformat(),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "ret_skew": ret_skew,
        "ret_kurt": ret_kurt,
        "alpha_daily": alpha_daily,
        "beta_mkt": beta_mkt,
        "beta_smb": beta_smb,
        "beta_hml": beta_hml,
        "beta_wml": beta_wml,
        "asia_name_flag": int("asia" in name_lc),
        "japan_name_flag": int("japan" in name_lc),
        "pacific_name_flag": int("pacific" in name_lc),
    }

    fieldnames = list(feature_row.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                key: format_float(value) if isinstance(value, float) else value
                for key, value in feature_row.items()
            }
        )

    return feature_row


def main() -> None:
    feature_row = build_aaxj_features()
    print("Archivo generado:", OUTPUT_PATH)
    print("Observaciones alineadas:", feature_row["n_obs"])
    print("Ventana efectiva:", feature_row["start_date"], "->", feature_row["end_date"])
    print("beta_mkt:", feature_row["beta_mkt"])
    print("sharpe:", feature_row["sharpe"])


if __name__ == "__main__":
    main()
