import glob
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass
class Params:
    spread_threshold: float
    imbalance_target: float
    cadence_sec: int
    price_range: Tuple[float, float]
    min_depth: float
    initial_chunk_usd: float
    order_chunk_usd: float
    budget_usd: float = 1000.0
    start_offset_min: float = None  # None = весь рынок
    limit_offset: float = 0.0  # если >0, ставим лимитку лучше ask на offset
    require_touch: bool = False  # если True: лимит исполняется только при касании цены


def list_market_files(log_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    return files


class GammaClient:
    def __init__(self):
        self.session = requests.Session()
        self.cache: Dict[str, Optional[int]] = {}

    def get_winner(self, slug: str) -> Optional[int]:
        if slug in self.cache:
            return self.cache[slug]
        try:
            resp = self.session.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                self.cache[slug] = None
                return None
            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                self.cache[slug] = None
                return None
            market = markets[0]
            prices = market.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices)
            winner = None
            if isinstance(prices, list) and len(prices) >= 2:
                try:
                    price_up = float(prices[0])
                    price_down = float(prices[1])
                    if price_up > 0.99:
                        winner = 0
                    elif price_down > 0.99:
                        winner = 1
                except Exception:
                    winner = None
            self.cache[slug] = winner
            return winner
        except Exception:
            self.cache[slug] = None
            return None


def mark_to_market(row, up_contracts: float, down_contracts: float, total_invested: float) -> float:
    up_bid = row["up_bid_1_price"]
    up_ask = row["up_ask_1_price"]
    down_bid = row["down_bid_1_price"]
    down_ask = row["down_ask_1_price"]
    up_mid = (up_bid + up_ask) / 2 if not math.isnan(up_bid) and not math.isnan(up_ask) else up_ask
    down_mid = (down_bid + down_ask) / 2 if not math.isnan(down_bid) and not math.isnan(down_ask) else down_ask
    current_value = up_contracts * up_mid + down_contracts * down_mid
    return current_value - total_invested


def simulate_market(df: pd.DataFrame, params: Params) -> Dict:
    up_contracts = 0.0
    down_contracts = 0.0
    invested = 0.0
    max_drawdown = 0.0
    actions = 0
    last_eval_ms = None

    min_price, max_price = params.price_range

    # Ограничим окно торгов: последние start_offset_min минут до конца
    if params.start_offset_min is not None and "seconds_till_end" in df.columns:
        df = df[df["seconds_till_end"] <= params.start_offset_min * 60].copy()
        if df.empty:
            return {
                "actions": 0,
                "invested": 0.0,
                "up_contracts": 0.0,
                "down_contracts": 0.0,
                "max_drawdown": 0.0,
            }

    for _, row in df.iterrows():
        ts = row["timestamp_ms"]
        if last_eval_ms is None:
            last_eval_ms = ts
        if ts - last_eval_ms < params.cadence_sec * 1000:
            continue
        last_eval_ms = ts

        up_ask = row["up_ask_1_price"]
        down_ask = row["down_ask_1_price"]
        up_bid = row["up_bid_1_price"]
        down_bid = row["down_bid_1_price"]

        # Basic sanity checks
        if any(math.isnan(x) for x in [up_ask, down_ask, up_bid, down_bid]):
            continue
        if up_ask <= 0 or down_ask <= 0:
            continue
        if up_ask + down_ask > params.spread_threshold:
            continue
        if not (min_price <= up_ask <= max_price and min_price <= down_ask <= max_price):
            continue

        # Liquidity filter
        up_depth = row.get("pm_up_ask_depth5", 0) + row.get("pm_up_bid_depth5", 0)
        down_depth = row.get("pm_down_ask_depth5", 0) + row.get("pm_down_bid_depth5", 0)
        if (up_depth < params.min_depth) or (down_depth < params.min_depth):
            continue

        remaining = params.budget_usd - invested
        if remaining <= 0:
            # Still track drawdown even if no further buys
            unreal = mark_to_market(row, up_contracts, down_contracts, invested)
            if unreal < max_drawdown:
                max_drawdown = unreal
            continue

        imbalance = 0.0
        if max(up_contracts, down_contracts) > 0:
            imbalance = abs(up_contracts - down_contracts) / max(up_contracts, down_contracts)

        action_done = False
        # Определяем фаворита (дешевле ask) и размеры шагов
        fav_is_up = up_ask < down_ask
        if fav_is_up:
            fav_price, dog_price = up_ask, down_ask
        else:
            fav_price, dog_price = down_ask, up_ask

        # Шаги лотов (приближено к стилю: фаворит крупнее, андердог лесенка мелких)
        fav_lot_usd = params.initial_chunk_usd if up_contracts == 0 and down_contracts == 0 else params.order_chunk_usd
        dog_lot_usd = params.order_chunk_usd * 0.2  # мелкие на андердоге

        # Целевой уклон 1.5–2x по контрактам в пользу фаворита
        target_ratio = 1.5
        if fav_is_up:
            ratio = (up_contracts + 1e-9) / (down_contracts + 1e-9)
        else:
            ratio = (down_contracts + 1e-9) / (up_contracts + 1e-9)

        # Покупки (с учётом лимитки)
        def fill(side_price, usd):
            # лимитка: ставим на (side_price - offset); если require_touch=True и ask не дошёл — не исполняется
            if usd <= 0:
                return 0.0, 0.0
            limit_price = side_price
            if params.limit_offset > 0:
                limit_price = max(0.001, side_price - params.limit_offset)
                if params.require_touch and side_price > limit_price:
                    return 0.0, 0.0
            return usd / limit_price, usd

        if up_contracts == 0 and down_contracts == 0:
            fav_buy = min(remaining, fav_lot_usd)
            dog_buy = min(remaining - fav_buy, dog_lot_usd)
            if fav_is_up:
                add_c, add_cost = fill(fav_price, fav_buy)
                up_contracts += add_c
                invested += add_cost
                add_c, add_cost = fill(dog_price, dog_buy)
                down_contracts += add_c
                invested += add_cost
            else:
                add_c, add_cost = fill(fav_price, fav_buy)
                down_contracts += add_c
                invested += add_cost
                add_c, add_cost = fill(dog_price, dog_buy)
                up_contracts += add_c
                invested += add_cost
            remaining = params.budget_usd - invested
            action_done = True
        else:
            if ratio < target_ratio and remaining > 0:
                fav_buy = min(remaining, fav_lot_usd)
                if fav_buy > 0:
                    add_c, add_cost = fill(fav_price, fav_buy)
                    if fav_is_up:
                        up_contracts += add_c
                    else:
                        down_contracts += add_c
                    invested += add_cost
                    remaining -= add_cost
                    action_done = True
            if remaining > 0:
                dog_buy = min(remaining, dog_lot_usd)
                if dog_buy > 0:
                    add_c, add_cost = fill(dog_price, dog_buy)
                    if fav_is_up:
                        down_contracts += add_c
                    else:
                        up_contracts += add_c
                    invested += add_cost
                    remaining -= add_cost
                    action_done = True

        if action_done:
            actions += 1
        # Update drawdown
        unreal = mark_to_market(row, up_contracts, down_contracts, invested)
        if unreal < max_drawdown:
            max_drawdown = unreal

    return {
        "actions": actions,
        "invested": invested,
        "up_contracts": up_contracts,
        "down_contracts": down_contracts,
        "max_drawdown": max_drawdown,
    }


def final_pnl(result: Dict, winner: int) -> float:
    invested = result["invested"]
    if winner == 0:  # Up wins
        return result["up_contracts"] * 1.0 - invested
    elif winner == 1:  # Down wins
        return result["down_contracts"] * 1.0 - invested
    else:
        return float("nan")


def load_market(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def run_all(log_dir: str, params_grid: List[Params], limit_markets: Optional[int] = None) -> pd.DataFrame:
    files = list_market_files(log_dir)
    if limit_markets:
        files = files[:limit_markets]
    gamma = GammaClient()
    rows = []

    total = len(files)
    for idx, file_path in enumerate(files):
        df = load_market(file_path)
        if df.empty:
            continue
        slug = df.iloc[0]["market_slug"]
        winner = gamma.get_winner(slug)
        if winner is None:
            continue
        for params in params_grid:
            sim = simulate_market(df, params)
            pnl = final_pnl(sim, winner)
            rows.append(
                {
                    "market": slug,
                    "config": f"sp{params.spread_threshold}_imb{params.imbalance_target}_cad{params.cadence_sec}_pr{params.price_range}_depth{params.min_depth}",
                    "pnl": pnl,
                    "max_drawdown": sim["max_drawdown"],
                    "invested": sim["invested"],
                    "actions": sim["actions"],
                    "winner": winner,
                }
            )
        # Progress hint
        if total > 0 and (idx + 1) % 10 == 0:
            pct = (idx + 1) / total * 100
            print(f"Progress: {idx + 1}/{total} markets ({pct:.1f}%)")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("config")
        .agg(
            markets=("market", "nunique"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            max_loss=("pnl", "min"),
            max_gain=("pnl", "max"),
            worst_dd=("max_drawdown", "min"),
            avg_dd=("max_drawdown", "mean"),
            actions=("actions", "mean"),
            invested_avg=("invested", "mean"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )
    return agg


def main():
    log_dir = "./logs"
    params_grid = [
        # Лимитные вариации (touch снят, чтобы были исполнения), мелкие лоты, строгий спред
        Params(1.010, 0.03, 1, (0.05, 0.95), 600, 30, 10, budget_usd=300, limit_offset=0.0025, require_touch=False),
        Params(1.010, 0.03, 1, (0.05, 0.95), 600, 30, 10, budget_usd=300, limit_offset=0.005, require_touch=False),
        Params(1.010, 0.03, 1, (0.05, 0.95), 600, 30, 10, budget_usd=300, limit_offset=0.01, require_touch=False),
        Params(1.010, 0.03, 2, (0.05, 0.95), 700, 30, 10, budget_usd=300, limit_offset=0.005, require_touch=False),
    ]

    start = time.time()
    results = run_all(log_dir, params_grid)
    if results.empty:
        print("No results produced.")
        return
    summary = summarize(results)

    results_path = "./balance_sim_results.csv"
    summary_path = "./balance_sim_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    duration = time.time() - start
    print(f"Done. Markets: {results['market'].nunique()}, configs: {summary.shape[0]}, time: {duration:.1f}s")
    print(f"Top configs:\n{summary.head(5)}")
    print(f"Saved detailed to {results_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

