#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature Cities Figure 2 - Revised V9 (6 panels, 2×3)
Core theme: Development trajectories reveal diversified strategic pathways

Panel layout (2×3):
  Row 1: (a) Strategy distribution across 3 periods  (b) Station growth vs S1 change  (c) Maturation S1-S2 change quadrant
  Row 2: (d) Strategy × city tier heatmap             (e) P2→P3 transition matrix      (f) Representative city trajectories

V9 changes:
  - Panel (b) redesigned: scatter colored by STRATEGY TYPE (not city tier),
    removed undefined "Synergistic growth" / "Exp.-dilution" quadrant labels,
    added clear P2/P3 period differentiation with connecting arrows,
    highlighted the 61% S1-decline finding more directly.

V9 changes:
  - Panel (b) x-axis changed to symlog scale for better data distribution visualization.

Author: Urban Transportation Research Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== Nature style ====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.linewidth': 0.5, 'font.size': 6,
    'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'legend.fontsize': 5, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

# ==================== City classification ====================
TIER_1_CITIES = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']
NEW_TIER_1_CITIES = [
    'Chengdu', 'Chongqing', 'Hangzhou', 'Wuhan', 'Xian', 'Zhengzhou',
    'Nanjing', 'Tianjin', 'Suzhou', 'Changsha', 'Dongguan', 'Shenyang',
    'Qingdao', 'Ningbo', 'Foshan']
TIER_2_CITIES = [
    'Kunming', 'Hefei', 'Dalian', 'Fuzhou', 'Xiamen', 'Harbin',
    'Changchun', 'Nanchang', 'Wuxi', 'Guiyang', 'Shijiazhuang',
    'Nanning', 'Taiyuan', 'Lanzhou', 'Urumqi', 'Hohhot', 'Jinan']
TIER_3_CITIES = [
    'Changzhou', 'Nantong', 'Wenzhou', 'Xuzhou', 'Shaoxing',
    'Luoyang', 'Wuhu', 'Chuzhou']
ALL_CITIES = TIER_1_CITIES + NEW_TIER_1_CITIES + TIER_2_CITIES + TIER_3_CITIES

TIER_COLORS = {
    'Tier-1': '#C1292E', 'New Tier-1': '#235789',
    'Tier-2': '#F1D302', 'Tier-3': '#6B9080',
}
TIER_ORDER = ['Tier-1', 'New Tier-1', 'Tier-2', 'Tier-3']

# ==================== Strategy configuration ====================
STRATEGY_ORDER = ['Density-ori.', 'Expansion-ori.', 'Balanced', 'Exp.-dilution', 'Stable']
STRATEGY_COLORS = {
    'Density-ori.': '#E63946',
    'Expansion-ori.': '#457B9D',
    'Balanced': '#2A9D8F',
    'Exp.-dilution': '#9B2335',
    'Stable': '#A8A8A8',
}

PERIOD_LABELS = {'P1': 'Early\n(2000–09)', 'P2': 'Rapid exp.\n(2010–19)', 'P3': 'Maturation\n(2020–25)'}
PERIOD_COLORS = {'P1': '#E63946', 'P2': '#457B9D', 'P3': '#2A9D8F'}
PERIOD_MARKERS = {'P1': 'o', 'P2': 's', 'P3': '^'}

# Quadrant background colors (for panel c)
QUAD_BG = {
    'S1↑S2↑': '#E8F5E9', 'S1↑S2↓': '#FFF8E1',
    'S1↓S2↑': '#E3F2FD', 'S1↓S2↓': '#FFEBEE',
}
QUAD_FG = {
    'S1↑S2↑': '#2E7D32', 'S1↑S2↓': '#E65100',
    'S1↓S2↑': '#1565C0', 'S1↓S2↓': '#C62828',
}

# Representative city styles
REP_COLORS = {'Shenzhen': '#2A9D8F', 'Wuhan': '#E63946', 'Dalian': '#457B9D'}
REP_LS = {'Shenzhen': '-', 'Wuhan': '--', 'Dalian': ':'}
REP_MK = {'Shenzhen': 'o', 'Wuhan': 's', 'Dalian': '^'}
REP_LABELS = {
    'Shenzhen': 'Shenzhen (Dens.→Expan.→Dilut.)',
    'Wuhan': 'Wuhan (Dens.→Dilut.→Dilut.)',
    'Dalian': 'Dalian (Dens.→Dens.→Dilut.)',
}


def get_city_tier(city):
    if city in TIER_1_CITIES: return 'Tier-1'
    elif city in NEW_TIER_1_CITIES: return 'New Tier-1'
    elif city in TIER_2_CITIES: return 'Tier-2'
    elif city in TIER_3_CITIES: return 'Tier-3'
    return 'Other'


def classify_strategy(l1_rate, l2_rate, threshold=0.005, balance_tol=0.003):
    """Classify development strategy based on annualized S1/S2 change rates.
    
    Criteria (matching Method text exactly):
      Exp.-dilution:           r1 < -threshold
      Stable:             |r1| < threshold AND |r2| < threshold
      Balanced:           r1 > threshold AND r2 > threshold AND |r1-r2| < balance_tol
      Density-oriented:   r1 > r2 + balance_tol AND r1 > threshold
      Expansion-oriented: r2 > r1 + balance_tol AND r2 > threshold
      Otherwise:          Stable (catch-all)
    """
    # 1. Exp.-dilution: S1 significantly declining
    if l1_rate < -threshold:
        return 'Exp.-dilution'
    # 2. Stable: both metrics below significance threshold
    if abs(l1_rate) < threshold and abs(l2_rate) < threshold:
        return 'Stable'
    # 3. Balanced: both growing significantly and at similar rates
    if l1_rate > threshold and l2_rate > threshold:
        if abs(l1_rate - l2_rate) < balance_tol:
            return 'Balanced'
    # 4. Density-oriented: S1 dominates, both conditions required
    if l1_rate > l2_rate + balance_tol and l1_rate > threshold:
        return 'Density-ori.'
    # 5. Expansion-oriented: S2 dominates, both conditions required
    if l2_rate > l1_rate + balance_tol and l2_rate > threshold:
        return 'Expansion-ori.'
    # 6. Catch-all: remaining ambiguous cases
    return 'Stable'


# ==================== Simulated data ====================
def generate_simulated_data(path='./simulated_data.csv'):
    print("Generating simulated data...")
    np.random.seed(42)
    data, sc = [], 0
    for city in ALL_CITIES:
        tier = get_city_tier(city)
        cfg = {'Tier-1': (50,1.12,50000,900000,2000), 'New Tier-1': (20,1.15,38000,600000,None),
               'Tier-2': (10,1.18,30000,400000,None), 'Tier-3': (5,1.20,25000,300000,None)}[tier]
        bs, gr, l1b, l2b = cfg[0], cfg[1], cfg[2], cfg[3]
        sy = cfg[4] or np.random.choice({'New Tier-1':[2005,2008,2010],
            'Tier-2':[2012,2015,2017], 'Tier-3':[2017,2019,2021]}[tier])
        if city == 'Lanzhou': l1b, bs = 78000, 8
        elif city == 'Suzhou': l1b, bs = 22000, 40
        elif city == 'Shenzhen': l1b = 65000
        for year in range(2000, 2026):
            if year < sy: continue
            ns = min(int(bs * (gr ** ((year - sy) / 3))), 520)
            l1t = (1 - 0.008*(year-sy)) if city in ['Suzhou','Hangzhou','Nanjing'] else (1 + 0.003*(year-sy))
            for _ in range(ns):
                sc += 1
                data.append({'station_id': f'S{sc:06d}', 'city': city, 'year': year,
                    'served_population': max(5000, l1b*l1t*np.random.lognormal(0,0.4)),
                    'cumulative_opportunities': max(50000, l2b*np.random.lognormal(0,0.3))})
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"  {len(df):,} records -> {path}")
    return path


# ==================== Figure 2 Generator ====================
class Figure2Generator:
    def __init__(self, csv_path, output_dir='./figures'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.city_yearly = None
        self.city_strategies = None
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        print("=" * 60)
        print("Loading data for Figure 2...")
        self.df = pd.read_csv(self.csv_path)
        self.df['city_tier'] = self.df['city'].apply(get_city_tier)
        print(f"  Records: {len(self.df):,}, Cities: {self.df['city'].nunique()}")

        # City yearly averages
        cy = self.df.groupby(['city', 'year']).agg({
            'served_population': 'mean', 'cumulative_opportunities': 'mean',
            'city_tier': 'first', 'station_id': 'count'
        }).reset_index()
        cy.columns = ['city', 'year', 'S1', 'S2', 'city_tier', 'n_stations']

        # Global normalization
        for col in ['S1', 'S2']:
            mn, mx = cy[col].min(), cy[col].max()
            cy[f'{col}_norm'] = (cy[col] - mn) / (mx - mn)
        self.city_yearly = cy

        self._analyze_strategies()
        return self

    def _analyze_strategies(self):
        """Classify each city's strategy in three periods"""
        P = {'P1': (2000, 2009), 'P2': (2010, 2019), 'P3': (2020, 2025)}
        results = []
        for city in self.city_yearly['city'].unique():
            cd = self.city_yearly[self.city_yearly['city'] == city].sort_values('year')
            tier = cd['city_tier'].iloc[0]
            r = {'city': city, 'tier': tier, 'start_year': cd['year'].min()}
            for pn, (y0, y1) in P.items():
                pd_ = cd[(cd['year'] >= y0) & (cd['year'] <= y1)]
                if len(pd_) >= 2:
                    pd_ = pd_.sort_values('year')
                    dl1 = pd_['S1_norm'].iloc[-1] - pd_['S1_norm'].iloc[0]
                    dl2 = pd_['S2_norm'].iloc[-1] - pd_['S2_norm'].iloc[0]
                    yrs = pd_['year'].max() - pd_['year'].min()
                    l1r = dl1 / yrs if yrs > 0 else 0
                    l2r = dl2 / yrs if yrs > 0 else 0
                    r[f'{pn}_strategy'] = classify_strategy(l1r, l2r)
                    r[f'{pn}_dS1'] = dl1
                    r[f'{pn}_dS2'] = dl2
                    # Station growth
                    r[f'{pn}_st_start'] = pd_['n_stations'].iloc[0]
                    r[f'{pn}_st_end'] = pd_['n_stations'].iloc[-1]
                else:
                    for k in ['strategy', 'dS1', 'dS2', 'st_start', 'st_end']:
                        r[f'{pn}_{k}'] = None
            results.append(r)
        self.city_strategies = pd.DataFrame(results)
        # Print summary
        for pn in P:
            col = f'{pn}_strategy'
            valid = self.city_strategies[self.city_strategies[col].notna()]
            print(f"  {pn}: {len(valid)} cities classified")

    # ==================== PANEL METHODS ====================

    def _panel_a(self, ax):
        """(a) Strategy distribution across 3 periods - stacked bar"""
        period_names = ['P1', 'P2', 'P3']
        x = np.arange(len(period_names))
        width = 0.6

        pct_data = {}
        for pn in period_names:
            col = f'{pn}_strategy'
            valid = self.city_strategies[self.city_strategies[col].notna()]
            counts = valid[col].value_counts()
            total = counts.sum()
            pct_data[pn] = {s: counts.get(s, 0) / total * 100 if total > 0 else 0
                            for s in STRATEGY_ORDER}

        bottom = np.zeros(len(period_names))
        for strat in STRATEGY_ORDER:
            vals = [pct_data[pn].get(strat, 0) for pn in period_names]
            bars = ax.bar(x, vals, width, bottom=bottom,
                          color=STRATEGY_COLORS[strat], edgecolor='white',
                          linewidth=0.3, label=strat)
            for i, v in enumerate(vals):
                if v > 10:
                    ax.text(x[i], bottom[i] + v/2, f'{v:.1f}%',
                            ha='center', va='center', fontsize=4.5,
                            fontweight='bold', color='white')
            bottom += vals

        for i, pn in enumerate(period_names):
            col = f'{pn}_strategy'
            n = self.city_strategies[self.city_strategies[col].notna()].shape[0]
            ax.text(x[i], bottom[i] + 1.5, f'n={n}', ha='center', fontsize=5, color='#666')

        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_LABELS[p] for p in period_names], fontsize=5.5)
        ax.set_ylabel('Percentage of cities (%)', fontsize=7)
        ax.set_ylim(0, 110)
        ax.legend(
            loc='upper center', fontsize=4.5, ncol=3,
            bbox_to_anchor=(0.51, 1.1), frameon=False,
            fancybox=True, shadow=False, borderaxespad=0.5
        )
        ax.grid(True, axis='y', alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_b(self, ax):
        """(b) REDESIGNED: Station growth vs ΔS1 for P2 and P3, colored by strategy type.
        
        V9: x-axis uses symlog scale to better distribute data points.
        symlog keeps linear behavior near zero and switches to log for large values,
        which handles both negative and highly skewed positive growth rates well.
        """
        # --- Collect P2 data ---
        p2_valid = self.city_strategies[self.city_strategies['P2_strategy'].notna()].copy()
        p2_points = []
        for _, row in p2_valid.iterrows():
            st0 = float(row['P2_st_start'])
            st1 = float(row['P2_st_end'])
            g = (st1 - st0) / max(st0, 1) * 100
            d = float(row['P2_dS1']) * 100
            p2_points.append({'city': row['city'], 'tier': row['tier'],
                              'growth': g, 'dS1': d, 'strategy': row['P2_strategy']})
        p2_df = pd.DataFrame(p2_points)

        # --- Collect P3 data ---
        p3_valid = self.city_strategies[self.city_strategies['P3_strategy'].notna()].copy()
        p3_points = {}
        for _, row in p3_valid.iterrows():
            st0 = float(row['P3_st_start'])
            st1 = float(row['P3_st_end'])
            g = (st1 - st0) / max(st0, 1) * 100
            d = float(row['P3_dS1']) * 100
            p3_points[row['city']] = {'growth': g, 'dS1': d, 'strategy': row['P3_strategy']}

        # --- Determine symlog linthresh ---
        # linthresh defines the range around zero that stays linear.
        # Choose ~10% so small growth values are still distinguishable.
        all_g = p2_df['growth'].tolist() + [v['growth'] for v in p3_points.values()]
        all_d = p2_df['dS1'].tolist() + [v['dS1'] for v in p3_points.values()]
        linthresh = 10  # linear within ±10%

        # Set symlog scale on x-axis
        ax.set_xscale('symlog', linthresh=linthresh)

        # --- Y-axis limits (linear) ---
        ylo = min(min(all_d) - 2, -5)
        yhi = max(max(all_d) + 2, 5)
        ax.set_ylim(ylo, yhi)

        # --- X-axis limits ---
        xlo = min(min(all_g) - 5, -15)
        xhi = max(max(all_g) * 1.1, 20)
        ax.set_xlim(xlo, xhi)

        # --- Background shading (use actual data limits for fill_between) ---
        # For symlog, fill_between still works; we use wide numeric ranges
        ax.fill_between([xlo, xhi], ylo, 0, color='#FFEBEE', alpha=0.30, zorder=0)
        ax.fill_between([xlo, xhi], 0, yhi, color='#E8F5E9', alpha=0.15, zorder=0)

        # Zero line for ΔS1
        ax.axhline(0, color='#333', ls='-', lw=0.8, alpha=0.6, zorder=2)

        # --- Connecting lines from P2 to P3 ---
        for _, row in p2_df.iterrows():
            city = row['city']
            if city in p3_points:
                p3 = p3_points[city]
                ax.plot([row['growth'], p3['growth']], [row['dS1'], p3['dS1']],
                        color='#BDBDBD', lw=0.4, alpha=0.35, zorder=1)

        # --- Plot P2 scatter (circles) ---
        for strat in STRATEGY_ORDER:
            sub = p2_df[p2_df['strategy'] == strat]
            if len(sub) == 0:
                continue
            ax.scatter(sub['growth'], sub['dS1'],
                       c=STRATEGY_COLORS[strat], s=28, alpha=0.85,
                       edgecolors='white', linewidths=0.4, zorder=6,
                       marker='o')

        # --- Plot P3 scatter (triangles) ---
        for city, p3 in p3_points.items():
            strat = p3['strategy']
            color = STRATEGY_COLORS.get(strat, '#888')
            ax.scatter(p3['growth'], p3['dS1'],
                       c=color, s=16, alpha=0.55,
                       edgecolors='white', linewidths=0.3, zorder=5,
                       marker='^')

        # --- City name annotations ---
        label_cities = {
            'Beijing':  (6, 5),
            'Shenzhen': (6, -7),
            'Suzhou':   (6, 4),
            'Lanzhou':  (-14, 5),
            'Wuhan':    (-14, -6),
        }
        for city, offset in label_cities.items():
            p2_row = p2_df[p2_df['city'] == city]
            if len(p2_row) == 0:
                continue
            g = float(p2_row['growth'].values[0])
            d = float(p2_row['dS1'].values[0])
            ax.annotate(city, (g, d), xytext=offset, textcoords='offset points',
                        fontsize=4.5, style='italic', color='#333',
                        arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.3),
                        zorder=10)

        # --- Key statistics annotation ---
        n_decline_p2 = (p2_df['dS1'] < 0).sum()
        n_total_p2 = len(p2_df)
        pct_decline_p2 = n_decline_p2 / n_total_p2 * 100 if n_total_p2 > 0 else 0

        p3_dS1_vals = [v['dS1'] for v in p3_points.values()]
        n_decline_p3 = sum(1 for v in p3_dS1_vals if v < 0)
        n_total_p3 = len(p3_dS1_vals)
        pct_decline_p3 = n_decline_p3 / n_total_p3 * 100 if n_total_p3 > 0 else 0

        stat_text = (f'S1 decline:\n'
                     f'  P2: {n_decline_p2}/{n_total_p2} ({pct_decline_p2:.0f}%)\n'
                     f'  P3: {n_decline_p3}/{n_total_p3} ({pct_decline_p3:.0f}%)')
        ax.text(0.07, 0.05, stat_text, transform=ax.transAxes,
                fontsize=4.8, color='#B71C1C', fontweight='bold',
                ha='left', va='bottom', linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE',
                          ec='#EF9A9A', lw=0.5, alpha=0.9))

        # Zone labels
        ax.text(0.05, 0.65, 'S1 increase ↑', transform=ax.transAxes,
                fontsize=5, ha='left', color='#2E7D32', style='italic', alpha=0.65)
        ax.text(0.05, 0.35, '↓ S1 decline', transform=ax.transAxes,
                fontsize=5, ha='left', color='#C62828', style='italic', alpha=0.65)

        # --- Legend ---
        strat_handles = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=STRATEGY_COLORS[s],
                                markersize=4, markeredgecolor='white',
                                markeredgewidth=0.3, label=s)
                         for s in STRATEGY_ORDER if s in p2_df['strategy'].values or
                         s in [v['strategy'] for v in p3_points.values()]]

        period_handles = [
            Line2D([0], [0], marker='o', color='#666', ls='None', ms=4.5,
                   markerfacecolor='#999', markeredgecolor='white',
                   markeredgewidth=0.3, label='P2 (2010–19)'),
            Line2D([0], [0], marker='^', color='#666', ls='None', ms=3.5,
                   markerfacecolor='#999', markeredgecolor='white',
                   markeredgewidth=0.3, label='P3 (2020–25)'),
        ]
        leg = ax.legend(handles=strat_handles + period_handles,
                  loc='upper left', fontsize=4.2, markerscale=0.9,
                  ncol=2, handletextpad=0.3, labelspacing=0.3, columnspacing=0.6,
                  bbox_to_anchor=(0.0, 1.0))

        # --- Axes ---
        ax.set_xlabel('Station count change (%, symlog)', fontsize=7)
        ax.set_ylabel('ΔS1 (normalised, ×100)', fontsize=7)
        ax.grid(True, alpha=0.12, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'b', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    def _panel_c(self, ax):
        """(c) Maturation period S1-S2 change quadrant with strategy labels"""
        valid = self.city_strategies[self.city_strategies['P3_strategy'].notna()].copy()
        if len(valid) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            return

        ds1 = valid['P3_dS1'].astype(float)
        ds2 = valid['P3_dS2'].astype(float)

        margin = 0.05
        xmn, xmx = ds1.min() - margin, ds1.max() + margin
        ymn, ymx = ds2.min() - margin, ds2.max() + margin
        xmn, xmx = min(xmn, -0.05), max(xmx, 0.05)
        ymn, ymx = min(ymn, -0.05), max(ymx, 0.05)

        ax.fill_between([0, xmx], 0, ymx, color=QUAD_BG['S1↑S2↑'], alpha=0.25, zorder=0)
        ax.fill_between([0, xmx], ymn, 0, color=QUAD_BG['S1↑S2↓'], alpha=0.25, zorder=0)
        ax.fill_between([xmn, 0], 0, ymx, color=QUAD_BG['S1↓S2↑'], alpha=0.25, zorder=0)
        ax.fill_between([xmn, 0], ymn, 0, color=QUAD_BG['S1↓S2↓'], alpha=0.25, zorder=0)
        ax.axhline(0, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)
        ax.axvline(0, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)

        for strat in STRATEGY_ORDER:
            mask = valid['P3_strategy'] == strat
            if mask.sum() > 0:
                ax.scatter(ds1[mask], ds2[mask], c=STRATEGY_COLORS[strat],
                           s=25, alpha=0.8, label=strat, edgecolors='white',
                           linewidths=0.3, zorder=5)

        for pos, lbl, col in [((0.85,0.58),'S1↑S2↑','S1↑S2↑'), ((0.85,0.05),'S1↑S2↓','S1↑S2↓'),
                               ((0.12,0.58),'S1↓S2↑','S1↓S2↑'), ((0.12,0.05),'S1↓S2↓','S1↓S2↓')]:
            ax.text(*pos, lbl, transform=ax.transAxes, fontsize=5, ha='center',
                    color=QUAD_FG[col], style='italic', alpha=0.8)

        lz = valid[valid['city'] == 'Lanzhou']
        if len(lz) > 0:
            ax.annotate('Lanzhou\n(Balanced)', (float(lz['P3_dS1'].values[0]), float(lz['P3_dS2'].values[0])),
                        xytext=(5, 3), textcoords='offset points', fontsize=5,
                        fontweight='bold', color=STRATEGY_COLORS['Balanced'],
                        arrowprops=dict(arrowstyle='->', color=STRATEGY_COLORS['Balanced'], lw=0.8))

        ax.set_xlabel('ΔS1 (2020–2025)', fontsize=7)
        ax.set_ylabel('ΔS2 (2020–2025)', fontsize=7)
        ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx)
        ax.legend(loc='upper right', fontsize=4.5, markerscale=0.8)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_d(self, ax):
        """(d) Strategy × city tier heatmap (P3 period)"""
        valid = self.city_strategies[self.city_strategies['P3_strategy'].notna()]
        ct = pd.crosstab(valid['tier'], valid['P3_strategy'])
        for t in TIER_ORDER:
            if t not in ct.index: ct.loc[t] = 0
        for s in STRATEGY_ORDER:
            if s not in ct.columns: ct[s] = 0
        ct = ct.reindex(index=TIER_ORDER, columns=STRATEGY_ORDER, fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct = pct.fillna(0)
        mx = pct.values / 100

        im = ax.imshow(mx, aspect='auto', cmap='Blues',
                       norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))
        for i in range(len(TIER_ORDER)):
            for j in range(len(STRATEGY_ORDER)):
                v = pct.iloc[i, j]
                c = int(ct.iloc[i, j])
                if c > 0:
                    tc = 'white' if mx[i, j] > 0.55 else 'black'
                    ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                            fontsize=5, color=tc, fontweight='bold')

        short = ['Dens.', 'Expan.', 'Bal.', 'Dilut.', 'Stable']
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=5, rotation=35, ha='right')
        ax.set_yticks(range(len(TIER_ORDER)))
        ax.set_yticklabels(TIER_ORDER, fontsize=5.5)
        ax.set_xlabel('Strategy (2020–2025)', fontsize=7)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=5)
        ax.text(-0.18, 1.08, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_e(self, ax):
        """(e) Strategy transition matrix P2→P3 (row-normalised transition probabilities)"""
        valid = self.city_strategies[
            self.city_strategies['P2_strategy'].notna() &
            self.city_strategies['P3_strategy'].notna()]
        ct = pd.crosstab(valid['P2_strategy'], valid['P3_strategy'])
        for s in STRATEGY_ORDER:
            if s not in ct.index: ct.loc[s] = 0
            if s not in ct.columns: ct[s] = 0
        ct = ct.reindex(index=STRATEGY_ORDER, columns=STRATEGY_ORDER, fill_value=0)

        vals = ct.values.astype(float)

        # Row-normalize to get transition probabilities (%)
        row_sums = vals.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        prob = vals / row_sums * 100  # percentage 0-100

        im = ax.imshow(prob, aspect='auto', cmap='Blues', vmin=0, vmax=100)
        for i in range(len(STRATEGY_ORDER)):
            for j in range(len(STRATEGY_ORDER)):
                cnt = int(vals[i, j])
                pct = prob[i, j]
                if cnt > 0:
                    tc = 'white' if pct > 45 else 'black'
                    # Show count with percentage below
                    ax.text(j, i - 0.12, str(cnt), ha='center', va='center',
                            fontsize=5.5, color=tc, fontweight='bold')
                    ax.text(j, i + 0.18, f'{pct:.0f}%', ha='center', va='center',
                            fontsize=4, color=tc, alpha=0.85)

        short = ['Dens.', 'Expan.', 'Bal.', 'Dilut.', 'Stable']
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=5, rotation=35, ha='right')
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=5)
        ax.set_xlabel('P3 strategy (2020–25)', fontsize=7)
        ax.set_ylabel('P2 strategy (2010–19)', fontsize=7)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Transition prob. (%)', fontsize=5, labelpad=2)
        cbar.ax.tick_params(labelsize=5)

        diag_sum = np.trace(vals)
        total = vals.sum()
        cons = diag_sum / total * 100 if total > 0 else 0
        ax.text(0.02, 0.02, f'Consistency: {cons:.1f}%', transform=ax.transAxes,
                fontsize=5.5, style='italic', color='#666',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#CCC', lw=0.5))
        ax.text(-0.18, 1.08, 'e', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_f(self, ax):
        """(f) Representative city trajectories: Shenzhen, Wuhan, Dalian"""
        for city in ['Shenzhen', 'Wuhan', 'Dalian']:
            cd = self.city_yearly[self.city_yearly['city'] == city].sort_values('year')
            if len(cd) == 0:
                continue
            s1n, s2n = cd['S1_norm'].values, cd['S2_norm'].values
            years = cd['year'].values

            ax.plot(s1n, s2n, ls=REP_LS[city], color=REP_COLORS[city],
                    lw=1.8, alpha=0.85)
            step = max(1, len(s1n) // 4)
            ax.scatter(s1n[1:-1:step], s2n[1:-1:step], c=REP_COLORS[city], s=12,
                       marker=REP_MK[city], edgecolors='white', linewidths=0.3, alpha=0.7)

            for yr in [2010, 2020]:
                if yr in years:
                    idx = np.where(years == yr)[0][0]
                    ax.scatter(s1n[idx], s2n[idx], c=REP_COLORS[city], s=18,
                               marker=REP_MK[city], edgecolors='white', linewidths=0.5, zorder=5)
                    ax.annotate(f"'{str(yr)[2:]}", (s1n[idx], s2n[idx]),
                                xytext=(3, 3), textcoords='offset points',
                                fontsize=4.5, color=REP_COLORS[city])

            ax.scatter(s1n[0], s2n[0], c='white', s=35, marker='o',
                       edgecolors=REP_COLORS[city], linewidths=1.5, zorder=10)
            ax.scatter(s1n[-1], s2n[-1], c=REP_COLORS[city], s=60, marker='*',
                       edgecolors='white', linewidths=0.5, zorder=10)

        ax.axhline(0.5, color='#888', ls='--', alpha=0.4, lw=0.5)
        ax.axvline(0.5, color='#888', ls='--', alpha=0.4, lw=0.5)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('S1 (Service intensity, norm.)', fontsize=7)
        ax.set_ylabel('S2 (Network access., norm.)', fontsize=7)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)

        handles = [Line2D([0], [0], color=REP_COLORS[c], ls=REP_LS[c], lw=1.8,
                          marker=REP_MK[c], markersize=4, markeredgecolor='white',
                          markeredgewidth=0.3, label=REP_LABELS[c])
                   for c in ['Shenzhen', 'Wuhan', 'Dalian']]
        handles += [
            Line2D([0], [0], marker='o', color='#666', ls='None', ms=4,
                   markerfacecolor='white', markeredgewidth=1, label='Start'),
            Line2D([0], [0], marker='*', color='#666', ls='None', ms=6,
                   markeredgecolor='white', markeredgewidth=0.3, label='End'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=4.5)
        ax.text(-0.15, 1.08, 'f', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    # ==================== MAIN PLOT ====================
    def plot_figure2(self):
        print("\n" + "=" * 60)
        print("Generating Figure 2 (V9, 2×3, symlog x-axis on panel b)")
        print("=" * 60)
        fig = plt.figure(figsize=(180/25.4, 120/25.4))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                               hspace=0.45, wspace=0.42,
                               top=0.94, bottom=0.08, left=0.08, right=0.96)
        panels = [('a', self._panel_a), ('b', self._panel_b), ('c', self._panel_c),
                  ('d', self._panel_d), ('e', self._panel_e), ('f', self._panel_f)]
        for idx, (label, func) in enumerate(panels):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            func(ax)
            print(f"  ✓ Panel {label}")

        path = os.path.join(self.output_dir, 'Figure2_Trajectories_v9.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {path}")
        return path


def main():
    import sys
    out = "./figures"; os.makedirs(out, exist_ok=True)
    csv = "/Users/liangwang/output/metro_accessibility_yearly/accessibility_all_years.csv"
    if not os.path.exists(csv):
        csv = generate_simulated_data('./simulated_data.csv')
    print(f"Using: {csv}")
    gen = Figure2Generator(csv, out)
    gen.load_data()
    gen.plot_figure2()

if __name__ == "__main__":
    main()