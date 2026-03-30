#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature Cities Figure 4 - V7 (6 panels, 2×3)
Core theme: Path dependence constrains station development trajectories

Panel layout (2×3):
  Row 1: (a) Type distribution over time (area)  (b) Transition matrix (first→2025)  (c) Sankey 2010→2020→2025
  Row 2: (d) LL→HH upgrade paths (bar)           (e) Representative station cases     (f) City HH% comparison

Author: Urban Transportation Research Team
Date: 2025
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors
from collections import defaultdict
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'axes.unicode_minus':False,'figure.dpi':300,'savefig.dpi':300,
    'axes.linewidth':0.5,'font.size':6,'axes.labelsize':7,'axes.titlesize':7,
    'xtick.labelsize':6,'ytick.labelsize':6,'xtick.major.size':2,'ytick.major.size':2,
    'xtick.major.width':0.5,'ytick.major.width':0.5,
    'legend.fontsize':5,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial',
})

T1=['Beijing','Shanghai','Guangzhou','Shenzhen']
NT1=['Chengdu','Chongqing','Hangzhou','Wuhan','Xian','Zhengzhou','Nanjing','Tianjin',
     'Suzhou','Changsha','Dongguan','Shenyang','Qingdao','Ningbo','Foshan']
T2=['Kunming','Hefei','Dalian','Fuzhou','Xiamen','Harbin','Changchun','Nanchang','Wuxi',
    'Guiyang','Shijiazhuang','Nanning','Taiyuan','Lanzhou','Urumqi','Hohhot','Jinan']
T3=['Changzhou','Nantong','Wenzhou','Xuzhou','Shaoxing','Luoyang','Wuhu','Chuzhou']
FC={'HH':'#E63946','HL':'#F4A261','LH':'#A8DADC','LL':'#457B9D'}
TC={'Tier-1':'#C1292E','New Tier-1':'#235789','Tier-2':'#F1D302','Tier-3':'#6B9080'}
TO=['Tier-1','New Tier-1','Tier-2','Tier-3']
TYPE_ORDER=['HH','HL','LH','LL']

def tier(c):
    if c in T1: return 'Tier-1'
    elif c in NT1: return 'New Tier-1'
    elif c in T2: return 'Tier-2'
    elif c in T3: return 'Tier-3'
    return 'Other'


class Figure4Generator:
    def __init__(self, csv_path, output_dir='./figures'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.merged = None  # station tracking
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        print("="*60)
        print("Loading data for Figure 4...")
        self.df = pd.read_csv(self.csv_path)
        self.df['tier'] = self.df['city'].apply(tier)

        # Compute 2D types per year
        parts = []
        for yr in sorted(self.df['year'].unique()):
            yd = self.df[self.df['year']==yr].copy()
            s1m, s2m = yd['served_population'].median(), yd['cumulative_opportunities'].median()
            l1 = np.where(yd['served_population'].values > s1m, 'H', 'L')
            l2 = np.where(yd['cumulative_opportunities'].values > s2m, 'H', 'L')
            yd['type_2d'] = pd.array([a + b for a, b in zip(l1, l2)])
            parts.append(yd)
        self.df = pd.concat(parts, ignore_index=True)

        # Station tracking: first type → 2025 type
        d25 = self.df[self.df['year']==self.df['year'].max()][['station_id','city','tier','type_2d']].copy()
        d25.rename(columns={'type_2d':'type_2025'}, inplace=True)
        first = self.df.sort_values('year').groupby('station_id').first().reset_index()
        first = first[['station_id','year','type_2d']].rename(columns={'year':'first_year','type_2d':'type_first'})
        self.merged = d25.merge(first, on='station_id', how='left')

        n_same = (self.merged['type_first']==self.merged['type_2025']).sum()
        print(f"  Stations: {len(self.merged)}, Unchanged: {n_same} ({n_same/len(self.merged)*100:.1f}%)")
        return self

    def _panel_a(self, ax):
        """(a) Type distribution over time — stacked area"""
        years = sorted(self.df['year'].unique())
        pcts = {t: [] for t in TYPE_ORDER}
        for yr in years:
            yd = self.df[self.df['year']==yr]
            total = len(yd)
            for t in TYPE_ORDER:
                pcts[t].append((yd['type_2d']==t).sum()/total*100 if total>0 else 0)

        ax.stackplot(years, [pcts[t] for t in TYPE_ORDER],
                     labels=TYPE_ORDER, colors=[FC[t] for t in TYPE_ORDER], alpha=0.8)
        ax.set_xlabel('Year', fontsize=7); ax.set_ylabel('Percentage of stations (%)', fontsize=7)
        ax.set_xlim(years[0], years[-1]); ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=5, ncol=2)
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)

        # Station count annotation
        for yr in [2000, 2010, 2020, 2025]:
            n = len(self.df[self.df['year']==yr])
            ax.annotate(f'n={n:,}', (yr, 2), fontsize=4.5, ha='center', color='white', fontweight='bold')

        ax.text(-0.15, 1.08, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_b(self, ax):
        """(b) Transition matrix: first type → 2025 type"""
        ct = pd.crosstab(self.merged['type_first'], self.merged['type_2025'])
        for t in TYPE_ORDER:
            if t not in ct.index: ct.loc[t] = 0
            if t not in ct.columns: ct[t] = 0
        ct = ct.reindex(index=TYPE_ORDER, columns=TYPE_ORDER, fill_value=0)

        # Normalize rows to percentages
        row_sums = ct.sum(axis=1)
        pct = ct.div(row_sums, axis=0) * 100

        im = ax.imshow(pct.values, cmap='Blues', aspect='auto',
                       norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
        for i in range(4):
            for j in range(4):
                v = pct.iloc[i, j]
                n = ct.iloc[i, j]
                if n > 0:
                    tc = 'white' if v > 50 else 'black'
                    ax.text(j, i, f'{v:.1f}%\n({n})', ha='center', va='center',
                            fontsize=5, color=tc, fontweight='bold' if i==j else 'normal')

        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(TYPE_ORDER, fontsize=6); ax.set_yticklabels(TYPE_ORDER, fontsize=6)
        ax.set_xlabel('Type in 2025', fontsize=7); ax.set_ylabel('Initial type', fontsize=7)

        # Persistence annotation
        n_same = (self.merged['type_first']==self.merged['type_2025']).sum()
        ax.text(0.02, 0.02, f'Persistence: {n_same/len(self.merged)*100:.1f}%',
                transform=ax.transAxes, fontsize=5.5, style='italic', color='#333',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#CCC', lw=0.5))

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=5)
        ax.text(-0.18, 1.08, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_c(self, ax):
        """(c) Simplified Sankey: 2010→2020→2025 transitions"""
        d10 = self.df[self.df['year']==2010][['station_id','type_2d']].rename(columns={'type_2d':'t1'})
        d20 = self.df[self.df['year']==2020][['station_id','type_2d']].rename(columns={'type_2d':'t2'})
        d25 = self.df[self.df['year']==2025][['station_id','type_2d']].rename(columns={'type_2d':'t3'})
        m = d10.merge(d20, on='station_id').merge(d25, on='station_id')

        # Draw as alluvial/parallel coordinates
        periods = ['2010', '2020', '2025']
        x_pos = [0, 1, 2]
        types = TYPE_ORDER

        # Count at each period
        for xi, col in zip(x_pos, ['t1', 't2', 't3']):
            bottom = 0
            total = len(m)
            for t in types:
                h = (m[col] == t).sum() / total
                rect = Rectangle((xi - 0.15, bottom), 0.3, h, facecolor=FC[t],
                                  edgecolor='white', linewidth=0.5, alpha=0.9, zorder=5)
                ax.add_patch(rect)
                if h > 0.08:
                    ax.text(xi, bottom + h/2, f'{t}\n{h*100:.0f}%', ha='center', va='center',
                            fontsize=4.5, color='white', fontweight='bold', zorder=6)
                bottom += h

        # Draw flows between periods
        for (col1, col2), (x1, x2) in [(['t1','t2'], [0, 1]), (['t2','t3'], [1, 2])]:
            flow = m.groupby([col1, col2]).size().reset_index(name='count')
            total = len(m)
            # Compute cumulative positions
            src_cum = {t: 0 for t in types}
            dst_cum = {t: 0 for t in types}
            for t in types:
                src_cum[t] = sum((m[col1] == tt).sum()/total for tt in types[:types.index(t)])
                dst_cum[t] = sum((m[col2] == tt).sum()/total for tt in types[:types.index(t)])

            for _, row in flow.iterrows():
                s, d, cnt = row[col1], row[col2], row['count']
                h = cnt / total
                if h < 0.01: continue
                y_s = src_cum[s]; src_cum[s] += h
                y_d = dst_cum[d]; dst_cum[d] += h
                # Bezier curve
                verts = [(x1+0.15, y_s), (x1+0.15, y_s+h),
                         ((x1+x2)/2, (y_s+y_d)/2+h), (x2-0.15, y_d+h),
                         (x2-0.15, y_d), ((x1+x2)/2, (y_s+y_d)/2),
                         (x1+0.15, y_s)]
                # Simple polygon approximation
                n_pts = 20
                top = np.column_stack([np.linspace(x1+0.15, x2-0.15, n_pts),
                    [y_s+h + (y_d+h-y_s-h)*t**1.5 for t in np.linspace(0,1,n_pts)]])
                bot = np.column_stack([np.linspace(x2-0.15, x1+0.15, n_pts),
                    [y_d + (y_s-y_d)*t**1.5 for t in np.linspace(0,1,n_pts)]])
                poly = np.vstack([top, bot])
                color = FC[s] if s == d else '#CCCCCC'
                alpha = 0.5 if s == d else 0.2
                ax.fill(poly[:,0], poly[:,1], color=color, alpha=alpha, zorder=2)

        ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(x_pos); ax.set_xticklabels(periods, fontsize=7)
        ax.set_ylabel('Proportion', fontsize=7)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.text(-0.15, 1.08, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_d(self, ax):
        """(d) LL→HH upgrade paths"""
        ll_hh = self.merged[(self.merged['type_first']=='LL') & (self.merged['type_2025']=='HH')]
        ll_hh_ids = ll_hh['station_id'].tolist()

        paths = {'LL→HL→HH': 0, 'LL→LH→HH': 0, 'Direct\n(LL→HH)': 0, 'Other': 0}
        for sid in ll_hh_ids:
            hist = self.df[self.df['station_id']==sid].sort_values('year')['type_2d'].tolist()
            seq = [hist[0]]
            for h in hist[1:]:
                if h != seq[-1]: seq.append(h)
            if 'HL' in seq and 'HH' in seq and seq.index('HL') < seq.index('HH'):
                paths['LL→HL→HH'] += 1
            elif 'LH' in seq and 'HH' in seq and seq.index('LH') < seq.index('HH'):
                paths['LL→LH→HH'] += 1
            elif seq == ['LL', 'HH']:
                paths['Direct\n(LL→HH)'] += 1
            else:
                paths['Other'] += 1

        labels = list(paths.keys())
        values = list(paths.values())
        total = sum(values)
        colors = ['#F4A261', '#A8DADC', '#E63946', '#999999']

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white',
                       lw=0.3, height=0.6, alpha=0.85)
        for i, (v, bar) in enumerate(zip(values, bars)):
            ax.text(v + 2, i, f'{v} ({v/total*100:.1f}%)', va='center', fontsize=5.5)

        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('Number of stations', fontsize=7)

        # Subtitle
        ax.text(0.98, 0.95, f'Total LL→HH: {total}\n({total/len(self.merged)*100:.1f}% of all)',
                transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCC', lw=0.5))

        ax.text(-0.18, 1.08, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_e(self, ax):
        """(e) Representative station type evolution over time"""
        # Pick a few representative stations with interesting trajectories
        # Find one LL→HL→HH, one LL→LH→HH, one stable HH, one stable LL
        cases = {}
        for sid in self.merged['station_id'].unique():
            hist = self.df[self.df['station_id']==sid].sort_values('year')
            if len(hist) < 5: continue
            types = hist['type_2d'].tolist()
            seq = [types[0]]
            for t in types[1:]:
                if t != seq[-1]: seq.append(t)
            city = hist['city'].iloc[0]
            name = hist['name_en'].iloc[0] if 'name_en' in hist.columns else sid

            if 'LL→HL→HH' not in cases and seq == ['LL','HL','HH']:
                cases['LL→HL→HH'] = (sid, name, city)
            elif 'LL→LH→HH' not in cases and seq == ['LL','LH','HH']:
                cases['LL→LH→HH'] = (sid, name, city)
            elif 'HH stable' not in cases and all(t=='HH' for t in types) and len(types)>10:
                cases['HH stable'] = (sid, name, city)
            elif 'LL stable' not in cases and all(t=='LL' for t in types) and len(types)>10:
                cases['LL stable'] = (sid, name, city)

            if len(cases) >= 4: break

        type_to_y = {'HH': 4, 'HL': 3, 'LH': 2, 'LL': 1}
        colors_case = ['#E63946', '#2A9D8F', '#457B9D', '#F4A261']
        ls_case = ['-', '--', '-.', ':']

        for i, (path_name, (sid, name, city)) in enumerate(cases.items()):
            hist = self.df[self.df['station_id']==sid].sort_values('year')
            years = hist['year'].values
            y_vals = [type_to_y[t] for t in hist['type_2d'].values]
            short_name = name[:15] + '..' if len(str(name)) > 15 else name
            label = f'{short_name} ({path_name})'
            ax.plot(years, y_vals, color=colors_case[i], ls=ls_case[i], lw=1.5, alpha=0.8,
                    marker='o', markersize=2, markeredgecolor='white', markeredgewidth=0.3, label=label)

        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['LL', 'LH', 'HL', 'HH'], fontsize=6)
        ax.set_xlabel('Year', fontsize=7); ax.set_ylabel('Station type', fontsize=7)
        ax.set_ylim(0.5, 4.5)

        # Background bands
        for y, t in [(1,'LL'), (2,'LH'), (3,'HL'), (4,'HH')]:
            ax.axhspan(y-0.4, y+0.4, color=FC[t], alpha=0.08, zorder=0)

        ax.legend(loc='upper left', fontsize=4.5, ncol=1)
        ax.grid(True, axis='x', alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'e', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def _panel_f(self, ax):
        """(f) City-level HH proportion comparison at key years"""
        cities = ['Shanghai', 'Beijing', 'Shenzhen', 'Guangzhou', 'Wuhan', 'Suzhou']
        years_show = [2010, 2020, 2025]
        yr_colors = ['#E63946', '#457B9D', '#2A9D8F']

        x = np.arange(len(cities))
        w = 0.25

        for i, (yr, clr) in enumerate(zip(years_show, yr_colors)):
            vals = []
            for city in cities:
                cd = self.df[(self.df['city']==city) & (self.df['year']==yr)]
                if len(cd) > 0:
                    vals.append((cd['type_2d']=='HH').sum()/len(cd)*100)
                else:
                    vals.append(0)
            bars = ax.bar(x + (i - 1) * w, vals, w, color=clr, alpha=0.8,
                          edgecolor='white', lw=0.3, label=str(yr))
            for xi, v in zip(x + (i-1)*w, vals):
                if v > 0:
                    ax.text(xi, v + 1, f'{v:.0f}', ha='center', fontsize=4.5, color=clr)

        ax.set_xticks(x)
        ax.set_xticklabels(cities, fontsize=5.5, rotation=15, ha='right')
        ax.set_ylabel('HH station proportion (%)', fontsize=7)
        ax.set_ylim(0, 80)
        ax.legend(loc='upper right', fontsize=5)
        ax.grid(True, axis='y', alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.15, 1.08, 'f', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    def plot_figure4(self):
        print("\n" + "="*60)
        print("Generating Figure 4 (V7, 2×3)")
        print("="*60)
        fig = plt.figure(figsize=(180/25.4, 125/25.4))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                               hspace=0.48, wspace=0.42,
                               top=0.94, bottom=0.08, left=0.08, right=0.96)
        for idx, (label, func) in enumerate([
            ('a', self._panel_a), ('b', self._panel_b), ('c', self._panel_c),
            ('d', self._panel_d), ('e', self._panel_e), ('f', self._panel_f)]):
            ax = fig.add_subplot(gs[idx//3, idx%3])
            func(ax)
            print(f"  ✓ Panel {label}")

        path = os.path.join(self.output_dir, 'Figure4_PathDependence_v7.png')
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
    gen = Figure4Generator(csv, out)
    gen.load_data()
    gen.plot_figure4()

if __name__ == "__main__":
    main()