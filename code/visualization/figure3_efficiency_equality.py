#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature Cities Figure 3 - Revised V7 (6 panels, 2×3)
Core theme: Efficiency-equality tradeoffs — why win-win is rare

Panel layout (2×3):
  Row 1: (a) Efficiency-Equality quadrant scatter  (b) Tier-level eff/eq paired bar  (c) Quadrant by tier stacked bar
  Row 2: (d) Tier-1 city trajectories            (e) ΔS1 vs ΔEquality quadrant       (f) Radar chart (tier profiles)

Definitions:
  Efficiency = normalised mean S2 (network accessibility)
  Equality = 1 - Gini(S1) (evenness of service intensity)

Author: Urban Transportation Research Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import warnings, os

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.linewidth': 0.5, 'font.size': 6,
    'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'legend.fontsize': 5, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
})

T1 = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']
NT1 = ['Chengdu','Chongqing','Hangzhou','Wuhan','Xian','Zhengzhou','Nanjing','Tianjin',
       'Suzhou','Changsha','Dongguan','Shenyang','Qingdao','Ningbo','Foshan']
T2 = ['Kunming','Hefei','Dalian','Fuzhou','Xiamen','Harbin','Changchun','Nanchang','Wuxi',
      'Guiyang','Shijiazhuang','Nanning','Taiyuan','Lanzhou','Urumqi','Hohhot','Jinan']
T3 = ['Changzhou','Nantong','Wenzhou','Xuzhou','Shaoxing','Luoyang','Wuhu','Chuzhou']
ALL = T1 + NT1 + T2 + T3

TC = {'Tier-1': '#C1292E', 'New Tier-1': '#235789', 'Tier-2': '#F1D302', 'Tier-3': '#6B9080'}
TO = ['Tier-1', 'New Tier-1', 'Tier-2', 'Tier-3']

T1_COLORS = {'Beijing': '#E63946', 'Shanghai': '#457B9D', 'Guangzhou': '#2A9D8F', 'Shenzhen': '#F4A261'}
T1_LS = {'Beijing': '-', 'Shanghai': '--', 'Guangzhou': '-.', 'Shenzhen': ':'}
T1_MK = {'Beijing': 'o', 'Shanghai': 's', 'Guangzhou': '^', 'Shenzhen': 'D'}

QUAD_COLORS = {'Win-Win': '#2E7D32', 'Eff-Priority': '#1565C0',
               'Eq-Priority': '#E65100', 'Dual-Challenge': '#C62828'}
QUAD_BG = {'Win-Win': '#E8F5E9', 'Eff-Priority': '#E3F2FD',
           'Eq-Priority': '#FFF8E1', 'Dual-Challenge': '#FFEBEE'}


def tier(c):
    if c in T1: return 'Tier-1'
    elif c in NT1: return 'New Tier-1'
    elif c in T2: return 'Tier-2'
    elif c in T3: return 'Tier-3'
    return 'Other'


def gini(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 2: return 0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


class Figure3Generator:
    def __init__(self, csv_path, output_dir='./figures'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.cy = None  # city-year
        self.c25 = None  # 2025 snapshot
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        print("=" * 60)
        print("Loading data for Figure 3...")
        self.df = pd.read_csv(self.csv_path)
        self.df['tier'] = self.df['city'].apply(tier)
        print(f"  Records: {len(self.df):,}, Cities: {self.df['city'].nunique()}")

        rows = []
        for (city, year), grp in self.df.groupby(['city', 'year']):
            s1 = grp['served_population'].values
            s2 = grp['cumulative_opportunities'].values
            rows.append({
                'city': city, 'year': year, 'tier': grp['tier'].iloc[0],
                'n': len(grp), 'S1_mean': s1.mean(), 'S2_mean': s2.mean(),
                'gini': gini(s1), 'equality': 1 - gini(s1),
            })
        self.cy = pd.DataFrame(rows)
        self.cy['efficiency'] = (self.cy['S2_mean'] - self.cy['S2_mean'].min()) / \
                                (self.cy['S2_mean'].max() - self.cy['S2_mean'].min())

        self.c25 = self.cy[self.cy['year'] == self.cy['year'].max()].copy()
        eff_med = self.c25['efficiency'].median()
        eq_med = self.c25['equality'].median()
        self.c25['quadrant'] = ''
        self.c25.loc[(self.c25['efficiency'] >= eff_med) & (self.c25['equality'] >= eq_med), 'quadrant'] = 'Win-Win'
        self.c25.loc[(self.c25['efficiency'] >= eff_med) & (self.c25['equality'] < eq_med), 'quadrant'] = 'Eff-Priority'
        self.c25.loc[(self.c25['efficiency'] < eff_med) & (self.c25['equality'] >= eq_med), 'quadrant'] = 'Eq-Priority'
        self.c25.loc[(self.c25['efficiency'] < eff_med) & (self.c25['equality'] < eq_med), 'quadrant'] = 'Dual-Challenge'

        # S1/equality change from earliest to 2025
        for city in self.c25['city'].unique():
            cd = self.cy[self.cy['city'] == city].sort_values('year')
            if len(cd) >= 2:
                self.c25.loc[self.c25['city'] == city, 'ds1'] = cd['S1_mean'].iloc[-1] - cd['S1_mean'].iloc[0]
                self.c25.loc[self.c25['city'] == city, 'deq'] = cd['equality'].iloc[-1] - cd['equality'].iloc[0]

        print(f"  City-year records: {len(self.cy)}")
        for q in ['Win-Win', 'Eff-Priority', 'Eq-Priority', 'Dual-Challenge']:
            print(f"  {q}: {(self.c25['quadrant'] == q).sum()} cities")
        return self

    # ==================== PANELS ====================

    def _panel_a(self, ax):
        """(a) Efficiency-Equality quadrant scatter"""
        c = self.c25
        eff_med = c['efficiency'].median()
        eq_med = c['equality'].median()

        # Background shading
        xmn, xmx = -0.02, c['efficiency'].max() * 1.1
        ymn, ymx = c['equality'].min() * 0.95, c['equality'].max() * 1.05
        ax.fill_between([eff_med, xmx], eq_med, ymx, color=QUAD_BG['Win-Win'], alpha=0.3, zorder=0)
        ax.fill_between([eff_med, xmx], ymn, eq_med, color=QUAD_BG['Eff-Priority'], alpha=0.3, zorder=0)
        ax.fill_between([xmn, eff_med], eq_med, ymx, color=QUAD_BG['Eq-Priority'], alpha=0.3, zorder=0)
        ax.fill_between([xmn, eff_med], ymn, eq_med, color=QUAD_BG['Dual-Challenge'], alpha=0.3, zorder=0)
        ax.axhline(eq_med, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)
        ax.axvline(eff_med, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)

        for t in TO:
            m = c['tier'] == t
            ax.scatter(c.loc[m, 'efficiency'], c.loc[m, 'equality'],
                       c=TC[t], s=30, alpha=0.8, label=t, edgecolors='white', linewidths=0.3, zorder=5)

        # Quadrant labels
        ax.text(0.52, 0.95, 'Win-Win', transform=ax.transAxes, fontsize=5.5,
                ha='center', color=QUAD_COLORS['Win-Win'], fontweight='bold', alpha=0.8)
        ax.text(0.52, 0.1, 'Eff-Priority', transform=ax.transAxes, fontsize=5.5,
                ha='center', color=QUAD_COLORS['Eff-Priority'], fontweight='bold', alpha=0.8)
        ax.text(0.15, 0.95, 'Eq-Priority', transform=ax.transAxes, fontsize=5.5,
                ha='center', color=QUAD_COLORS['Eq-Priority'], fontweight='bold', alpha=0.8)
        ax.text(0.15, 0.1, 'Dual-Challenge', transform=ax.transAxes, fontsize=5.5,
                ha='center', color=QUAD_COLORS['Dual-Challenge'], fontweight='bold', alpha=0.8)

        # Annotate key cities
        for city in ['Beijing', 'Shenzhen', 'Lanzhou', 'Suzhou', 'Guangzhou']:
            r = c[c['city'] == city]
            if len(r) > 0:
                off = {'Beijing': (5, 3), 'Shenzhen': (5, -6), 'Lanzhou': (-5, -6),
                       'Suzhou': (5, 3), 'Guangzhou': (5, 5)}.get(city, (5, 3))
                ax.annotate(city, (r['efficiency'].values[0], r['equality'].values[0]),
                            xytext=off, textcoords='offset points', fontsize=5,
                            style='italic', arrowprops=dict(arrowstyle='-', color='#888', lw=0.4))

        # Count box
        ww = (c['quadrant'] == 'Win-Win').sum()
        ax.text(0.98, 0.78, f'Win-Win:\n{ww}/{len(c)} ({ww/len(c)*100:.1f}%)',
                transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#2E7D32', lw=0.5))

        ax.set_xlabel('Efficiency (normalised mean S2)', fontsize=7)
        ax.set_ylabel('Equality (1 − Gini of S1)', fontsize=7)
        ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx)
        ax.legend(loc='lower right', fontsize=5, markerscale=0.8, title='City tier', title_fontsize=5.5)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_b(self, ax):
        """(b) Tier-level efficiency vs equality (grouped bar)"""
        c = self.c25
        tiers = TO
        eff_means = [c[c['tier'] == t]['efficiency'].mean() for t in tiers]
        eq_means = [c[c['tier'] == t]['equality'].mean() for t in tiers]
        eff_stds = [c[c['tier'] == t]['efficiency'].std() for t in tiers]
        eq_stds = [c[c['tier'] == t]['equality'].std() for t in tiers]

        x = np.arange(len(tiers))
        w = 0.35
        ax.bar(x - w/2, eff_means, w, yerr=eff_stds, color='#457B9D', alpha=0.8,
               edgecolor='white', lw=0.3, label='Efficiency', capsize=2, error_kw=dict(lw=0.5))
        ax.bar(x + w/2, eq_means, w, yerr=eq_stds, color='#E63946', alpha=0.8,
               edgecolor='white', lw=0.3, label='Equality', capsize=2, error_kw=dict(lw=0.5))

        # Value labels
        for i in range(len(tiers)):
            ax.text(x[i] - w/2, eff_means[i] + eff_stds[i] + 0.02, f'{eff_means[i]:.2f}',
                    ha='center', fontsize=4.5, color='#457B9D')
            ax.text(x[i] + w/2, eq_means[i] + eq_stds[i] + 0.02, f'{eq_means[i]:.2f}',
                    ha='center', fontsize=4.5, color='#E63946')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}\n(n={len(c[c["tier"]==t])})' for t in tiers], fontsize=5.5)
        ax.set_ylabel('Score', fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=5)
        ax.grid(True, axis='y', alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_c(self, ax):
        """(c) Quadrant distribution by tier (stacked bar)"""
        c = self.c25
        quads = ['Win-Win', 'Eff-Priority', 'Eq-Priority', 'Dual-Challenge']
        qc = {q: QUAD_COLORS[q] for q in quads}

        y_pos = np.arange(len(TO))
        left = np.zeros(len(TO))
        for q in quads:
            vals = []
            for t in TO:
                td = c[c['tier'] == t]
                vals.append((td['quadrant'] == q).sum() / len(td) * 100 if len(td) > 0 else 0)
            ax.barh(y_pos, vals, left=left, color=qc[q], edgecolor='white',
                    lw=0.3, label=q, height=0.65)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v > 12:
                    ax.text(l + v/2, i, f'{v:.1f}%', ha='center', va='center',
                            fontsize=4.5, fontweight='bold', color='white')
            left += vals

        np_ = {t: len(c[c['tier'] == t]) for t in TO}
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{t}\n(n={np_[t]})' for t in TO], fontsize=5.5)
        ax.set_xlabel('Percentage of cities (%)', fontsize=7)
        ax.set_xlim(0, 100)
        ax.legend(loc='upper right', fontsize=4.5, ncol=2, bbox_to_anchor=(1.02, 1.08),
                  columnspacing=0.5, handletextpad=0.3)
        ax.text(-0.18, 1.08, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_d(self, ax):
        """(d) Tier-1 city trajectories in efficiency-equality space"""
        for city in T1:
            cd = self.cy[self.cy['city'] == city].sort_values('year')
            if len(cd) < 2: continue
            eff, eq = cd['efficiency'].values, cd['equality'].values
            years = cd['year'].values

            ax.plot(eff, eq, ls=T1_LS[city], color=T1_COLORS[city], lw=1.5, alpha=0.85)
            step = max(1, len(eff) // 5)
            ax.scatter(eff[1:-1:step], eq[1:-1:step], c=T1_COLORS[city], s=10,
                       marker=T1_MK[city], edgecolors='white', linewidths=0.3, alpha=0.7)

            # Start
            ax.scatter(eff[0], eq[0], c='#2E7D32', s=30, marker='o',
                       zorder=10, edgecolors='white', linewidths=0.6)
            # Key years
            for yr, mk, clr in [(2010, 'X', '#FF9800'), (2020, 'D', '#9C27B0')]:
                if yr in years:
                    idx = np.where(years == yr)[0][0]
                    ax.scatter(eff[idx], eq[idx], c=clr, s=30, marker=mk,
                               zorder=10, edgecolors='white', linewidths=0.5)
            # End
            ax.scatter(eff[-1], eq[-1], c='#C62828', s=45, marker='*',
                       zorder=10, edgecolors='white', linewidths=0.4)

        ax.axhline(0.5, color='#888', ls='--', alpha=0.3, lw=0.5)
        ax.axvline(0.5, color='#888', ls='--', alpha=0.3, lw=0.5)
        ax.set_xlabel('Efficiency (norm. mean S2)', fontsize=7)
        ax.set_ylabel('Equality (1 − Gini)', fontsize=7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 0.9)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)

        # City legend
        handles = [Line2D([0], [0], color=T1_COLORS[c], ls=T1_LS[c], lw=1.5,
                          marker=T1_MK[c], ms=3.5, markeredgecolor='white',
                          markeredgewidth=0.3, label=c) for c in T1]
        handles += [
            Line2D([0], [0], marker='o', color='#2E7D32', ls='None', ms=4, label='Start',
                   markeredgecolor='white', markeredgewidth=0.4),
            Line2D([0], [0], marker='X', color='#FF9800', ls='None', ms=4, label='2010',
                   markeredgecolor='white', markeredgewidth=0.4),
            Line2D([0], [0], marker='D', color='#9C27B0', ls='None', ms=3.5, label='2020',
                   markeredgecolor='white', markeredgewidth=0.4),
            Line2D([0], [0], marker='*', color='#C62828', ls='None', ms=5.5, label='2025',
                   markeredgecolor='white', markeredgewidth=0.3),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=4.5, ncol=2)
        ax.text(-0.15, 1.08, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_e(self, ax):
        """(e) ΔS1 vs ΔEquality quadrant scatter"""
        c = self.c25.dropna(subset=['ds1', 'deq']).copy()
        ds1 = c['ds1'].values / 1000  # thousands
        deq = c['deq'].values

        xmn, xmx = min(ds1.min() * 1.1, -5), max(ds1.max() * 1.1, 5)
        ymn, ymx = min(deq.min() * 1.1, -0.05), max(deq.max() * 1.1, 0.05)

        # Background
        ax.fill_between([0, xmx], 0, ymx, color='#E8F5E9', alpha=0.25, zorder=0)
        ax.fill_between([0, xmx], ymn, 0, color='#FFF8E1', alpha=0.25, zorder=0)
        ax.fill_between([xmn, 0], 0, ymx, color='#E3F2FD', alpha=0.25, zorder=0)
        ax.fill_between([xmn, 0], ymn, 0, color='#FFEBEE', alpha=0.25, zorder=0)
        ax.axhline(0, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)
        ax.axvline(0, color='#888', ls='--', lw=0.5, alpha=0.6, zorder=1)

        for t in TO:
            m = c['tier'] == t
            ax.scatter(ds1[m.values], deq[m.values], c=TC[t], s=25, alpha=0.8,
                       label=t, edgecolors='white', linewidths=0.3, zorder=5)

        # Quadrant labels
        ax.text(0.85, 0.62, 'S1↑ Eq↑', transform=ax.transAxes, fontsize=5,
                color='#2E7D32', style='italic', alpha=0.8, ha='center')
        ax.text(0.85, 0.35, 'S1↑ Eq↓', transform=ax.transAxes, fontsize=5,
                color='#E65100', style='italic', alpha=0.8, ha='center')
        ax.text(0.12, 0.62, 'S1↓ Eq↑', transform=ax.transAxes, fontsize=5,
                color='#1565C0', style='italic', alpha=0.8, ha='center')
        ax.text(0.12, 0.35, 'S1↓ Eq↓', transform=ax.transAxes, fontsize=5,
                color='#C62828', style='italic', alpha=0.8, ha='center')

        # Count percentages
        n = len(c)
        s1up_eqUp = ((ds1 > 0) & (deq > 0)).sum()
        s1dn_eqUp = ((ds1 <= 0) & (deq > 0)).sum()
        s1up_eqDn = ((ds1 > 0) & (deq <= 0)).sum()
        s1dn_eqDn = ((ds1 <= 0) & (deq <= 0)).sum()
        ax.text(0.70, 0.20,
                f'S1↑Eq↑: {s1up_eqUp/n*100:.1f}%\n'
                f'S1↓Eq↑: {s1dn_eqUp/n*100:.1f}%\n'
                f'S1↑Eq↓: {s1up_eqDn/n*100:.1f}%\n'
                f'S1↓Eq↓: {s1dn_eqDn/n*100:.1f}%',
                transform=ax.transAxes, fontsize=5, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', lw=0.5))

        ax.set_xlabel('ΔS1 (×1,000 persons)', fontsize=7)
        ax.set_ylabel('ΔEquality', fontsize=7)
        ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx)
        ax.legend(loc='upper left', fontsize=4.5, markerscale=0.8)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'e', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_f(self, ax):
        """(f) Radar chart — multi-dimensional tier profiles"""
        c = self.c25
        # Dimensions: efficiency, equality, S1 mean (norm), network size (norm), S2 mean (norm)
        dims = ['Efficiency', 'Equality', 'S1 intensity', 'Network\nscale', 'S2 access.']
        n_max = c['n'].max()
        s1_max = c['S1_mean'].max()

        tier_vals = {}
        for t in TO:
            td = c[c['tier'] == t]
            tier_vals[t] = [
                td['efficiency'].mean(),
                td['equality'].mean(),
                td['S1_mean'].mean() / s1_max,
                td['n'].mean() / n_max,
                td['S2_mean'].mean() / c['S2_mean'].max(),
            ]

        n_dims = len(dims)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        for t in TO:
            vals = tier_vals[t] + tier_vals[t][:1]
            ax.plot(angles, vals, color=TC[t], lw=1.5, alpha=0.8, label=t)
            ax.fill(angles, vals, color=TC[t], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=4.5, color='#888')
        ax.legend(loc='lower left', fontsize=4.5, bbox_to_anchor=(-0.18, -0.05))
        ax.text(-0.15, 1.12, 'f', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    # ==================== MAIN ====================
    def plot_figure3(self):
        print("\n" + "=" * 60)
        print("Generating Figure 3 (V7, 2×3)")
        print("=" * 60)
        fig = plt.figure(figsize=(180/25.4, 125/25.4))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                               hspace=0.48, wspace=0.42,
                               top=0.94, bottom=0.08, left=0.08, right=0.96)

        for idx, (label, func) in enumerate([
            ('a', self._panel_a), ('b', self._panel_b), ('c', self._panel_c),
            ('d', self._panel_d), ('e', self._panel_e)]):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            func(ax)
            print(f"  ✓ Panel {label}")

        # Panel f: radar (polar)
        ax = fig.add_subplot(gs[1, 2], polar=True)
        self._panel_f(ax)
        print("  ✓ Panel f")

        path = os.path.join(self.output_dir, 'Figure3_EfficiencyEquality_v7.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {path}")
        return path


def main():
    import sys
    out = "./figures"; os.makedirs(out, exist_ok=True)
    csv = "/Users/liangwang/output/metro_accessibility_yearly/accessibility_all_years.csv"
    if len(sys.argv) > 1: csv = sys.argv[1]
    if len(sys.argv) > 2: out = sys.argv[2]
    print(f"Using: {csv}")
    gen = Figure3Generator(csv, out)
    gen.load_data()
    gen.plot_figure3()

if __name__ == "__main__":
    main()