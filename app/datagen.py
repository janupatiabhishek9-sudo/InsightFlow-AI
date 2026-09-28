"""Deterministic synthetic sales dataset generator.

The data contains an engineered story so the demo has a real answer to find:
European revenue drops in Q3 2024 versus Q2 2024, concentrated in Germany and France,
driven mostly by lower Enterprise volume of Laptop Pro and Standing Desk, plus heavier discounting.
It also contains a few deliberate data-quality problems (missing discounts, duplicate rows,
returns with negative revenue, a whitespace-variant country name) for the profiler to catch.

Usage:  python -m app.datagen [--out data/examples/sales.csv] [--seed 7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT

COUNTRIES = {
    "Europe": {"Germany": 3.0, "France": 2.2, "Netherlands": 1.2, "Spain": 1.3, "Italy": 1.4, "United Kingdom": 2.0},
    "North America": {"United States": 4.0, "Canada": 1.5, "Mexico": 1.0},
    "APAC": {"Japan": 2.0, "Australia": 1.2, "India": 1.6, "Singapore": 0.8},
}
PRODUCTS = {
    # product: (category, base price, cost ratio, popularity)
    "Laptop Pro": ("Electronics", 1200.0, 0.72, 1.0),
    "Tablet X": ("Electronics", 450.0, 0.65, 1.2),
    "Wireless Headphones": ("Electronics", 150.0, 0.55, 1.6),
    "Ergonomic Chair": ("Office", 300.0, 0.60, 1.0),
    "Standing Desk": ("Office", 550.0, 0.62, 0.8),
    "Monitor 27in": ("Office", 280.0, 0.66, 1.1),
    "Analytics Suite": ("Software", 900.0, 0.25, 0.6),
    "Security Pack": ("Software", 400.0, 0.20, 0.7),
    "USB-C Hub": ("Accessories", 60.0, 0.45, 1.8),
    "Keyboard": ("Accessories", 90.0, 0.50, 1.5),
}
SEGMENTS = {"Consumer": (0.50, 1, 3), "SMB": (0.32, 2, 8), "Enterprise": (0.18, 5, 30)}
DISCOUNTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20])


def _in_q3_2024_europe(dates: pd.Series, region: str) -> np.ndarray:
    return ((dates >= "2024-07-01") & (dates < "2024-10-01")).to_numpy() & (region == "Europe")


def generate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(len(days)) / 365.0
    seasonality = 1 + 0.08 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() / 365.0))
    growth = 1 + 0.06 * t
    product_names = list(PRODUCTS)
    popularity = np.array([PRODUCTS[p][3] for p in product_names])

    frames = []
    for region, countries in COUNTRIES.items():
        for country, weight in countries.items():
            lam = 0.75 * weight * seasonality * growth
            in_q3_24 = (days >= "2024-07-01") & (days < "2024-10-01")
            if region == "Europe":
                shock = {"Germany": 0.62, "France": 0.78, "Netherlands": 0.90}.get(country, 0.97)
                lam = np.where(in_q3_24, lam * shock, lam)
            counts = rng.poisson(lam)
            dates = np.repeat(days.to_numpy(), counts)
            n = len(dates)
            seg_names = list(SEGMENTS)
            seg_p = np.array([SEGMENTS[s][0] for s in seg_names])
            segments = rng.choice(seg_names, size=n, p=seg_p / seg_p.sum())
            prod_p = popularity / popularity.sum()
            products = rng.choice(product_names, size=n, p=prod_p)
            lo = np.array([SEGMENTS[s][1] for s in segments])
            hi = np.array([SEGMENTS[s][2] for s in segments])
            quantity = rng.integers(lo, hi + 1)
            base = np.array([PRODUCTS[p][1] for p in products])
            unit_price = np.round(base * rng.uniform(0.97, 1.03, size=n), 2)
            disc_p = np.array([0.45, 0.25, 0.18, 0.08, 0.04])
            discount = rng.choice(DISCOUNTS, size=n, p=disc_p)

            date_s = pd.Series(pd.to_datetime(dates))
            q3_eu = _in_q3_2024_europe(date_s, region)
            # Enterprise buyers of the two hero products cut volume hardest in Q3 2024 Europe.
            hero = np.isin(products, ["Laptop Pro", "Standing Desk"]) & (segments == "Enterprise")
            quantity = np.where(q3_eu & hero, np.maximum(1, (quantity * 0.45).astype(int)), quantity)
            # ...and discounting got heavier across Europe.
            heavier = rng.choice(DISCOUNTS, size=n, p=[0.25, 0.25, 0.25, 0.15, 0.10])
            discount = np.where(q3_eu, heavier, discount)

            frames.append(
                pd.DataFrame(
                    {
                        "order_date": date_s.dt.date,
                        "region": region,
                        "country": country,
                        "customer_segment": segments,
                        "product_category": [PRODUCTS[p][0] for p in products],
                        "product": products,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount": discount,
                        "cost_ratio": [PRODUCTS[p][2] for p in products],
                        "base_price": base,
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True).sort_values(["order_date", "country"], kind="stable")
    df = df.reset_index(drop=True)
    df["revenue"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount"]), 2)
    df["cost"] = np.round(df["quantity"] * df["base_price"] * df["cost_ratio"], 2)
    df["profit"] = np.round(df["revenue"] - df["cost"], 2)
    df = df.drop(columns=["cost_ratio", "base_price"])
    df.insert(0, "order_id", [f"ORD-{i:06d}" for i in range(1, len(df) + 1)])
    return _inject_quality_issues(df, rng)


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic, known data-quality problems outside the demo's key slice."""
    apac_2023 = df.index[(df["region"] == "APAC") & (pd.to_datetime(df["order_date"]) < pd.Timestamp("2023-04-01"))]
    # 1. Missing discounts (revenue was booked at list price).
    missing = rng.choice(apac_2023, size=14, replace=False)
    df.loc[missing, "discount"] = np.nan
    df.loc[missing, "revenue"] = np.round(df.loc[missing, "quantity"] * df.loc[missing, "unit_price"], 2)
    df.loc[missing, "profit"] = np.round(df.loc[missing, "revenue"] - df.loc[missing, "cost"], 2)
    # 2. Returns: negative quantity and revenue.
    returns = rng.choice(np.setdiff1d(apac_2023, missing), size=3, replace=False)
    for col in ("quantity", "revenue", "cost", "profit"):
        df.loc[returns, col] = -df.loc[returns, col].abs()
    # 3. Whitespace-variant category value.
    japan = df.index[(df["country"] == "Japan") & (pd.to_datetime(df["order_date"]) < pd.Timestamp("2023-02-01"))][:2]
    df.loc[japan, "country"] = "Japan "
    # 4. Duplicate rows (double-loaded batch).
    dupes = df.loc[rng.choice(apac_2023, size=25, replace=False)]
    return pd.concat([df, dupes], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "examples" / "sales.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    df = generate(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
