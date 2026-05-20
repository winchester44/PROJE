#!/usr/bin/env python3
"""
Fade Impulse Strategy Backtest
Detect market impulses (price spikes) and fade them with limits on underdog
"""
import pandas as pd
import numpy as np
import glob
import os
import sys
from collections import defaultdict

# Optional: import from btceth if available, otherwise use fallback
try:
    sys.path.insert(0, '../btceth')
    from polymarket_api import get_market_outcome
except ImportError:
    # Fallback: fetch outcome from Gamma API
    def get_market_outcome(slug):
        """Fetch market outcome from Polymarket Gamma API."""
        if not slug:
            return None
        try:
            import requests
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None
            market = events[0].get("markets", [{}])[0]
            prices = market.get("outcomePrices", [])
            if isinstance(prices, str):
                import json
                prices = json.loads(prices)
            if len(prices) >= 2:
                price_up = float(prices[0])
                price_down = float(prices[1])
                if price_up > 0.99:
                    return {"winner": "UP"}
                elif price_down > 0.99:
                    return {"winner": "DOWN"}
            return None
        except Exception:
            return None

def detect_impulse(df, idx, config):
    """Detect if there's an impulse at this moment"""
    if idx < 5:
        return None, None
    
    # Current row
    row = df.iloc[idx]
    prev = df.iloc[idx-1]
    
    # Calculate price changes
    up_ask_change = abs(row['up_ask_1_price'] - prev['up_ask_1_price']) if pd.notnull(row['up_ask_1_price']) and pd.notnull(prev['up_ask_1_price']) else 0
    down_ask_change = abs(row['down_ask_1_price'] - prev['down_ask_1_price']) if pd.notnull(row['down_ask_1_price']) and pd.notnull(prev['down_ask_1_price']) else 0
    
    # Check lat_dir
    lat_dir = abs(row.get('lat_dir_norm_x1000', 0))
    
    # Check vol spike
    vol_spike = row.get('binance_volume_spike', 0)
    
    # Impulse detected?
    impulse_detected = False
    
    if up_ask_change >= config['impulse_price_thresh'] or down_ask_change >= config['impulse_price_thresh']:
        impulse_detected = True
    
    if lat_dir >= config['impulse_latdir_thresh']:
        impulse_detected = True
        
    if vol_spike >= config['impulse_volspike_thresh']:
        impulse_detected = True
    
    if not impulse_detected:
        return None, None
    
    # Determine underdog (cheaper side after impulse)
    if pd.isnull(row['up_ask_1_price']) or pd.isnull(row['down_ask_1_price']):
        return None, None
    
    sum_ask = row['up_ask_1_price'] + row['down_ask_1_price']
    if sum_ask > config['max_spread_after_impulse']:
        return None, None
    
    # Underdog is cheaper side
    if row['up_ask_1_price'] < row['down_ask_1_price']:
        underdog = 'up'
        favorite = 'down'
    else:
        underdog = 'down'
        favorite = 'up'
    
    return underdog, favorite

def simulate_market(df, config):
    """Simulate fade impulse strategy on one market"""
    up_contracts = 0.0
    down_contracts = 0.0
    total_invested = 0.0
    actions = 0
    min_pnl = 0
    
    pending_limits = []  # list of (side, price, size_usd, timestamp_placed)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        if pd.isnull(row['up_ask_1_price']) or pd.isnull(row['down_ask_1_price']):
            continue
        
        up_ask = row['up_ask_1_price']
        down_ask = row['down_ask_1_price']
        timestamp = row['timestamp_ms']
        
        # Check pending limits for execution
        executed = []
        for i, (side, limit_price, size_usd, ts_placed) in enumerate(pending_limits):
            # Cancel if too old (2 seconds)
            if timestamp - ts_placed > 2000:
                executed.append(i)
                continue
            
            # Check if touched
            current_ask = up_ask if side == 'up' else down_ask
            if current_ask <= limit_price:
                # Execute
                contracts = size_usd / limit_price
                if side == 'up':
                    up_contracts += contracts
                else:
                    down_contracts += contracts
                total_invested += size_usd
                actions += 1
                executed.append(i)
        
        # Remove executed/cancelled
        for i in reversed(executed):
            pending_limits.pop(i)
        
        # Update PnL
        current_value = (up_contracts * up_ask) + (down_contracts * down_ask)
        unrealized_pnl = current_value - total_invested
        if unrealized_pnl < min_pnl:
            min_pnl = unrealized_pnl
        
        # Check stop loss
        if unrealized_pnl < config['stop_loss']:
            break
        
        # Detect impulse
        underdog, favorite = detect_impulse(df, idx, config)
        
        if underdog is None:
            continue
        
        # Check if we have budget
        if total_invested >= config['max_budget']:
            continue
        
        # Place limit on underdog
        underdog_ask = up_ask if underdog == 'up' else down_ask
        limit_price = underdog_ask - config['limit_offset']
        
        if limit_price > 0.01:  # sanity
            order_size = config['order_size']
            pending_limits.append((underdog, limit_price, order_size, timestamp))
        
        # Aggressive entry on favorite
        fav_ask = up_ask if favorite == 'up' else down_ask
        fav_size = config['order_size'] * config['favorite_multiplier']
        
        if total_invested + fav_size <= config['max_budget']:
            contracts = fav_size / fav_ask
            if favorite == 'up':
                up_contracts += contracts
            else:
                down_contracts += contracts
            total_invested += fav_size
            actions += 1
    
    # Final PnL
    if up_contracts == 0 and down_contracts == 0:
        return {
            'total_invested': 0,
            'pnl': 0,
            'max_drawdown': 0,
            'actions': 0,
            'up_contracts': 0,
            'down_contracts': 0,
        }
    
    # Get outcome from API
    slug = df['market_slug'].iloc[0] if 'market_slug' in df.columns else None
    outcome = get_market_outcome(slug) if slug else None
    
    if outcome and outcome['winner']:
        winner = outcome['winner']
        if winner == 'UP':
            final_value = up_contracts * 1.0
        elif winner == 'DOWN':
            final_value = down_contracts * 1.0
        else:
            final_value = 0
    else:
        # Fallback: assume no resolution
        final_value = 0
    
    final_pnl = final_value - total_invested
    
    return {
        'total_invested': total_invested,
        'pnl': final_pnl,
        'max_drawdown': min_pnl,
        'actions': actions,
        'up_contracts': up_contracts,
        'down_contracts': down_contracts,
        'winner': outcome['winner'] if outcome else None,
    }

def main():
    log_dir = './logs'
    files = sorted(glob.glob(os.path.join(log_dir, '*.csv')))[:330]  # limit to ~300+ markets
    
    # Config grid
    configs = [
        {
            'name': 'impulse_tight_small',
            'impulse_price_thresh': 0.03,
            'impulse_latdir_thresh': 30,
            'impulse_volspike_thresh': 2.0,
            'max_spread_after_impulse': 1.02,
            'limit_offset': 0.005,
            'order_size': 10,
            'favorite_multiplier': 1.5,
            'max_budget': 300,
            'stop_loss': -50,
        },
        {
            'name': 'impulse_medium',
            'impulse_price_thresh': 0.05,
            'impulse_latdir_thresh': 50,
            'impulse_volspike_thresh': 3.0,
            'max_spread_after_impulse': 1.015,
            'limit_offset': 0.003,
            'order_size': 15,
            'favorite_multiplier': 1.2,
            'max_budget': 400,
            'stop_loss': -100,
        },
        {
            'name': 'impulse_aggressive',
            'impulse_price_thresh': 0.04,
            'impulse_latdir_thresh': 40,
            'impulse_volspike_thresh': 2.5,
            'max_spread_after_impulse': 1.02,
            'limit_offset': 0.01,
            'order_size': 20,
            'favorite_multiplier': 1.0,
            'max_budget': 500,
            'stop_loss': -75,
        },
    ]
    
    results = defaultdict(list)
    
    total_markets = len(files)
    for midx, fpath in enumerate(files):
        if midx % 10 == 0:
            print(f"Progress: {midx}/{total_markets} markets ({100*midx/total_markets:.1f}%)")
        
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        
        if len(df) < 10:
            continue
        
        for config in configs:
            result = simulate_market(df, config)
            result['market'] = os.path.basename(fpath).replace('.csv', '')
            result['config'] = config['name']
            results[config['name']].append(result)
    
    # Summary
    print(f"\nDone. Markets: {total_markets}, configs: {len(configs)}")
    print("\nResults by config:")
    
    summary_rows = []
    for cfg_name in results:
        res_list = results[cfg_name]
        pnls = [r['pnl'] for r in res_list]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0
        median_pnl = np.median(pnls) if pnls else 0
        max_dd = min([r['max_drawdown'] for r in res_list]) if res_list else 0
        actions_avg = np.mean([r['actions'] for r in res_list]) if res_list else 0
        invested_avg = np.mean([r['total_invested'] for r in res_list]) if res_list else 0
        
        print(f"\n{cfg_name}:")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Avg PnL: ${avg_pnl:.2f}")
        print(f"  Median PnL: ${median_pnl:.2f}")
        print(f"  Worst DD: ${max_dd:.2f}")
        print(f"  Avg actions: {actions_avg:.1f}")
        print(f"  Avg invested: ${invested_avg:.1f}")
        
        summary_rows.append({
            'config': cfg_name,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'worst_dd': max_dd,
            'actions_avg': actions_avg,
            'invested_avg': invested_avg,
            'markets': len(res_list),
        })
    
    # Save results
    detail_rows = []
    for cfg_name in results:
        detail_rows.extend(results[cfg_name])
    
    df_detail = pd.DataFrame(detail_rows)
    df_summary = pd.DataFrame(summary_rows)
    
    df_detail.to_csv('./fade_impulse_detail.csv', index=False)
    df_summary.to_csv('./fade_impulse_summary.csv', index=False)
    
    print("\nSaved:")
    print("  ./fade_impulse_detail.csv")
    print("  ./fade_impulse_summary.csv")

if __name__ == '__main__':
    main()







