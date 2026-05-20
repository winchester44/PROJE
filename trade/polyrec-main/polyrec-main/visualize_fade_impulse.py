#!/usr/bin/env python3
"""
Visualize Fade Impulse Strategy on a specific market
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def detect_impulse(df, idx, config):
    """Detect if there's an impulse at this moment"""
    if idx < 5:
        return None, None, {}
    
    row = df.iloc[idx]
    prev = df.iloc[idx-1]
    
    up_ask_change = abs(row['up_ask_1_price'] - prev['up_ask_1_price']) if pd.notnull(row['up_ask_1_price']) and pd.notnull(prev['up_ask_1_price']) else 0
    down_ask_change = abs(row['down_ask_1_price'] - prev['down_ask_1_price']) if pd.notnull(row['down_ask_1_price']) and pd.notnull(prev['down_ask_1_price']) else 0
    
    lat_dir = abs(row.get('lat_dir_norm_x1000', 0))
    vol_spike = row.get('binance_volume_spike', 0)
    
    impulse_reasons = []
    impulse_detected = False
    
    if up_ask_change >= config['impulse_price_thresh']:
        impulse_detected = True
        impulse_reasons.append(f'up_Δ{up_ask_change:.3f}')
    
    if down_ask_change >= config['impulse_price_thresh']:
        impulse_detected = True
        impulse_reasons.append(f'dn_Δ{down_ask_change:.3f}')
        
    if lat_dir >= config['impulse_latdir_thresh']:
        impulse_detected = True
        impulse_reasons.append(f'lat_dir{lat_dir:.0f}')
        
    if vol_spike >= config['impulse_volspike_thresh']:
        impulse_detected = True
        impulse_reasons.append(f'vol_spike{vol_spike:.1f}')
    
    if not impulse_detected:
        return None, None, {}
    
    if pd.isnull(row['up_ask_1_price']) or pd.isnull(row['down_ask_1_price']):
        return None, None, {}
    
    sum_ask = row['up_ask_1_price'] + row['down_ask_1_price']
    if sum_ask > config['max_spread_after_impulse']:
        return None, None, {}
    
    if row['up_ask_1_price'] < row['down_ask_1_price']:
        underdog = 'up'
        favorite = 'down'
    else:
        underdog = 'down'
        favorite = 'up'
    
    return underdog, favorite, {'reasons': impulse_reasons, 'sum_ask': sum_ask}

def simulate_and_visualize(log_path, config, output_path):
    """Simulate strategy and create visualization"""
    df = pd.read_csv(log_path)
    
    if len(df) < 10:
        print(f"Market too short: {len(df)} rows")
        return
    
    # Filter valid rows
    df = df[pd.notnull(df['up_ask_1_price']) & pd.notnull(df['down_ask_1_price'])].copy()
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Simulate
    up_contracts = 0.0
    down_contracts = 0.0
    total_invested = 0.0
    
    entries = []
    impulses = []
    pnl_history = []
    contracts_history = []
    
    pending_limits = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        up_ask = row['up_ask_1_price']
        down_ask = row['down_ask_1_price']
        timestamp = row['timestamp_ms']
        dt = row['datetime']
        
        # Check pending limits
        executed = []
        for i, (side, limit_price, size_usd, ts_placed) in enumerate(pending_limits):
            if timestamp - ts_placed > 2000:
                executed.append(i)
                continue
            
            current_ask = up_ask if side == 'up' else down_ask
            if current_ask <= limit_price:
                contracts = size_usd / limit_price
                if side == 'up':
                    up_contracts += contracts
                else:
                    down_contracts += contracts
                total_invested += size_usd
                entries.append({
                    'datetime': dt,
                    'side': side,
                    'price': limit_price,
                    'size_usd': size_usd,
                    'type': 'limit',
                    'up_ask': up_ask,
                    'down_ask': down_ask,
                })
                executed.append(i)
        
        for i in reversed(executed):
            pending_limits.pop(i)
        
        # Update PnL
        current_value = (up_contracts * up_ask) + (down_contracts * down_ask)
        unrealized_pnl = current_value - total_invested
        pnl_history.append({'datetime': dt, 'pnl': unrealized_pnl})
        contracts_history.append({'datetime': dt, 'up': up_contracts, 'down': down_contracts})
        
        # Check stop
        if unrealized_pnl < config['stop_loss']:
            break
        
        # Detect impulse
        underdog, favorite, info = detect_impulse(df, idx, config)
        
        if underdog is not None:
            impulses.append({
                'datetime': dt,
                'underdog': underdog,
                'favorite': favorite,
                'reasons': ', '.join(info['reasons']),
                'sum_ask': info['sum_ask'],
                'up_ask': up_ask,
                'down_ask': down_ask,
            })
        
        if underdog is None or total_invested >= config['max_budget']:
            continue
        
        # Place limit on underdog
        underdog_ask = up_ask if underdog == 'up' else down_ask
        limit_price = underdog_ask - config['limit_offset']
        
        if limit_price > 0.01:
            order_size = config['order_size']
            pending_limits.append((underdog, limit_price, order_size, timestamp))
        
        # Aggressive on favorite
        fav_ask = up_ask if favorite == 'up' else down_ask
        fav_size = config['order_size'] * config['favorite_multiplier']
        
        if total_invested + fav_size <= config['max_budget']:
            contracts = fav_size / fav_ask
            if favorite == 'up':
                up_contracts += contracts
            else:
                down_contracts += contracts
            total_invested += fav_size
            entries.append({
                'datetime': dt,
                'side': favorite,
                'price': fav_ask,
                'size_usd': fav_size,
                'type': 'aggressive',
                'up_ask': up_ask,
                'down_ask': down_ask,
            })
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Prices
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['up_ask_1_price'], 'b-', linewidth=1, label='UP ask', alpha=0.7)
    ax1.plot(df['datetime'], df['down_ask_1_price'], 'r-', linewidth=1, label='DOWN ask', alpha=0.7)
    
    # Mark impulses
    for imp in impulses:
        ax1.axvline(imp['datetime'], color='orange', alpha=0.3, linewidth=0.5)
    
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_title(f"Fade Impulse Strategy: {df['market_slug'].iloc[0] if 'market_slug' in df.columns else 'Market'}", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Entries
    ax2 = axes[1]
    ax2.plot(df['datetime'], df['up_ask_1_price'], 'b-', linewidth=0.5, alpha=0.3)
    ax2.plot(df['datetime'], df['down_ask_1_price'], 'r-', linewidth=0.5, alpha=0.3)
    
    # Plot entries
    for entry in entries:
        color = 'green' if entry['type'] == 'limit' else 'purple'
        marker = 'o' if entry['side'] == 'up' else '^'
        size = entry['size_usd']
        ax2.scatter(entry['datetime'], entry['price'], c=color, marker=marker, s=size*2, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Limit (UP)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Limit (DOWN)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Aggressive (UP)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=8, label='Aggressive (DOWN)', markeredgecolor='black'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
    ax2.set_ylabel('Entry Price', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Panel 3: Contracts
    ax3 = axes[2]
    contracts_df = pd.DataFrame(contracts_history)
    ax3.plot(contracts_df['datetime'], contracts_df['up'], 'b-', label='UP contracts', linewidth=1.5)
    ax3.plot(contracts_df['datetime'], contracts_df['down'], 'r-', label='DOWN contracts', linewidth=1.5)
    ax3.fill_between(contracts_df['datetime'], contracts_df['up'], alpha=0.2, color='blue')
    ax3.fill_between(contracts_df['datetime'], contracts_df['down'], alpha=0.2, color='red')
    ax3.set_ylabel('Contracts', fontsize=10)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Panel 4: PnL
    ax4 = axes[3]
    pnl_df = pd.DataFrame(pnl_history)
    ax4.plot(pnl_df['datetime'], pnl_df['pnl'], 'g-', linewidth=2, label='Unrealized PnL')
    ax4.fill_between(pnl_df['datetime'], 0, pnl_df['pnl'], where=(pnl_df['pnl']>=0), alpha=0.3, color='green', label='Profit')
    ax4.fill_between(pnl_df['datetime'], 0, pnl_df['pnl'], where=(pnl_df['pnl']<0), alpha=0.3, color='red', label='Loss')
    ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax4.axhline(config['stop_loss'], color='red', linewidth=1, linestyle='--', label=f"Stop Loss ${config['stop_loss']}")
    ax4.set_ylabel('PnL ($)', fontsize=10)
    ax4.set_xlabel('Time', fontsize=10)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(alpha=0.3)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Stats text
    final_pnl = pnl_df['pnl'].iloc[-1] if len(pnl_df) > 0 else 0
    max_dd = pnl_df['pnl'].min() if len(pnl_df) > 0 else 0
    stats_text = f"Impulses: {len(impulses)} | Entries: {len(entries)} | Invested: ${total_invested:.0f} | Final PnL: ${final_pnl:.2f} | Max DD: ${max_dd:.2f}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    print(f"Stats: {stats_text}")

def main():
    # Config for tight_small
    config = {
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
    }
    
    # Example market - change to your target market file
    log_path = './logs/btc-updown-15m-1963380.csv'
    output_path = './fade_impulse_visualization.png'
    
    print("Visualizing Fade Impulse Strategy on btc-updown-15m-1963380...")
    simulate_and_visualize(log_path, config, output_path)

if __name__ == '__main__':
    main()





