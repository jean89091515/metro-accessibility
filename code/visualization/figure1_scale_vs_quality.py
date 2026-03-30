#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature Cities Figure 1 - V4 (5 panels, 2+3 layout)
Core theme: Scale does not equal quality

Panel layout:
  Row 1 (2 wide):  (a) Scale vs S1 scatter [enhanced]   (b) Station S1 CCDF
  Row 2 (3 narrow): (c) Xierqi station (HH)  (d) Huangcun station (HL)  (e) HH/HL/LH/LL stacked bar

Change log v4:
  - Removed panel b (heatmap). Its content absorbed into panel a annotations.
  - Panel a enhanced: key cities annotated with (n stations, S1=xx k);
    double-headed arrow highlights "same-scale, different-S1" pair
    (Shenzhen vs Suzhou).
  - Layout changed from 2x3 to 2-row (2 wide + 3 narrow).
  - Panels relabelled: a->a, c->b, d->c, e->d, f->e.

Usage:
    python nature_cities_figure1_v4.py [csv_path] [output_dir]
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import sys

warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ===================================================================
# Global style - Nature-compatible
# ===================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': 5,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
})

# ===================================================================
# City tier definitions & colour palettes
# ===================================================================
TIER_1_CITIES = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']
NEW_TIER_1_CITIES = [
    'Chengdu', 'Chongqing', 'Hangzhou', 'Wuhan', 'Xian', 'Zhengzhou',
    'Nanjing', 'Tianjin', 'Suzhou', 'Changsha', 'Dongguan', 'Shenyang',
    'Qingdao', 'Ningbo', 'Foshan',
]
TIER_2_CITIES = [
    'Kunming', 'Hefei', 'Dalian', 'Fuzhou', 'Xiamen', 'Harbin',
    'Changchun', 'Nanchang', 'Wuxi', 'Guiyang', 'Shijiazhuang',
    'Nanning', 'Taiyuan', 'Lanzhou', 'Urumqi', 'Hohhot', 'Jinan',
]
TIER_3_CITIES = [
    'Changzhou', 'Nantong', 'Wenzhou', 'Xuzhou', 'Shaoxing',
    'Luoyang', 'Wuhu', 'Chuzhou',
]
ALL_CITIES = TIER_1_CITIES + NEW_TIER_1_CITIES + TIER_2_CITIES + TIER_3_CITIES

TIER_COLORS = {
    'Tier-1': '#C1292E',
    'New Tier-1': '#235789',
    'Tier-2': '#F1D302',
    'Tier-3': '#6B9080',
}
FOUR_TYPE_COLORS = {
    'HH': '#E63946',
    'HL': '#F4A261',
    'LH': '#A8DADC',
    'LL': '#457B9D',
}
TIER_ORDER = ['Tier-1', 'New Tier-1', 'Tier-2', 'Tier-3']

MAP_COLORS = {
    'building': '#4A4A4A',
    'building_edge': '#2D2D2D',
    'metro_line': '#3C5488',
    'station_main': '#E64B35',
    'station_other': '#00A087',
    'buffer_edge': '#7E6148',
    'text': '#2D2D2D',
    'station_name': '#E64B35',
}

STATION_PATH = "/Users/liangwang/0-成果转化/01-论文/37-地铁时序数据集/2-数据/metro_stops_with_opening_years.shp"
ROUTE_PATH = "/Users/liangwang/0-成果转化/01-论文/37-地铁时序数据集/2-数据/metro_routes.shp"
BUILDING_PATH = "/Users/liangwang/12-数据/34-建筑轮廓数据/02-公众号分享百度2019/北京/Beijing_Buildings_DWG-Polygon.shp"
BUFFER_RADIUS = 1000
STATION_MERGE_THRESHOLD = 50
DISPLAY_MARGIN = 1.2
BUILDING_CLIP_MARGIN = 3.5

STATIONS_CONFIG = {
    'Xierqi': {
        'name_cn': '西二旗',
        'name_en': 'Xierqi',
        'type': 'HH',
        'S1': 43675,
        'S2': 7743733,
    },
    'Huangcun': {
        'name_cn': '黄村西大街',
        'name_en': 'Huangcun W. St.',
        'type': 'HL',
        'S1': 53873,
        'S2': 1351612,
    },
}


# ===================================================================
# Helper functions
# ===================================================================
def get_city_tier(city):
    if city in TIER_1_CITIES:
        return 'Tier-1'
    if city in NEW_TIER_1_CITIES:
        return 'New Tier-1'
    if city in TIER_2_CITIES:
        return 'Tier-2'
    if city in TIER_3_CITIES:
        return 'Tier-3'
    return 'Other'


def load_map_data():
    if not HAS_GEOPANDAS:
        return None, None, None
    try:
        return (
            gpd.read_file(STATION_PATH),
            gpd.read_file(ROUTE_PATH),
            gpd.read_file(BUILDING_PATH),
        )
    except Exception:
        return None, None, None


def get_station_point(stations, name_cn):
    s = stations[stations['name_cn'] == name_cn]
    if len(s) == 0:
        s = stations[stations['name_cn'].str.contains(name_cn, na=False)]
    return s.iloc[0].geometry if len(s) > 0 else None


def create_buffer(point, radius, crs):
    gdf = gpd.GeoDataFrame(geometry=[point], crs=crs).to_crs(epsg=32650)
    buf = gpd.GeoDataFrame(geometry=gdf.geometry.buffer(radius), crs='EPSG:32650')
    return buf.to_crs(crs).geometry.iloc[0]


def deduplicate_stations_gdf(sgdf, center_point, threshold_m=50):
    if len(sgdf) == 0:
        return gpd.GeoDataFrame()
    sp = sgdf.to_crs(epsg=32650)
    cp = (
        gpd.GeoDataFrame(geometry=[center_point], crs=sgdf.crs)
        .to_crs(epsg=32650)
        .geometry.iloc[0]
    )
    pts = [(g.x, g.y) for g in sp.geometry]
    uniq, used = [], set()
    for i, (x1, y1) in enumerate(pts):
        if i in used:
            continue
        cx, cy = [x1], [y1]
        used.add(i)
        for j, (x2, y2) in enumerate(pts):
            if j in used:
                continue
            if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < threshold_m:
                cx.append(x2)
                cy.append(y2)
                used.add(j)
        uniq.append(Point(np.mean(cx), np.mean(cy)))
    res = gpd.GeoDataFrame(geometry=uniq, crs='EPSG:32650').to_crs(sgdf.crs)
    res['is_main'] = False
    if len(res) > 0:
        rp = res.to_crs(epsg=32650)
        ds = [p.distance(cp) for p in rp.geometry]
        res.loc[np.argmin(ds), 'is_main'] = True
    return res


# -- Simulated data generator -----------------------------------------
def generate_simulated_data(path='./simulated_data.csv'):
    np.random.seed(42)
    data = []
    sc = 0
    cfg_map = {
        'Tier-1':     (50, 1.12, 50000, 900000, 2000),
        'New Tier-1': (20, 1.15, 38000, 600000, None),
        'Tier-2':     (10, 1.18, 30000, 400000, None),
        'Tier-3':     (5,  1.20, 25000, 300000, None),
    }
    for city in ALL_CITIES:
        t = get_city_tier(city)
        bs, gr, l1b, l2b, sy = cfg_map.get(t, cfg_map['Tier-3'])
        if sy is None:
            if t == 'New Tier-1':
                sy = np.random.choice([2005, 2008, 2010])
            elif t == 'Tier-2':
                sy = np.random.choice([2012, 2015, 2017])
            else:
                sy = np.random.choice([2017, 2019, 2021])
        if city == 'Lanzhou':
            l1b, bs = 78000, 8
        elif city == 'Suzhou':
            l1b, bs = 22000, 40
        elif city == 'Shenzhen':
            l1b = 65000
        for year in range(2000, 2026):
            if year < sy:
                continue
            ns = min(int(bs * (gr ** ((year - sy) / 3))), 520)
            l1t = 1 + (
                -0.008 if city in ['Suzhou', 'Hangzhou', 'Nanjing'] else 0.003
            ) * (year - sy)
            for _ in range(ns):
                sc += 1
                data.append({
                    'station_id': f'S{sc:06d}',
                    'city': city,
                    'year': year,
                    'served_population': max(
                        5000, l1b * l1t * np.random.lognormal(0, 0.4)
                    ),
                    'cumulative_opportunities': max(
                        50000, l2b * np.random.lognormal(0, 0.3)
                    ),
                })
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"  Simulated: {len(df):,} records")
    return path


# ===================================================================
# Main figure generator
# ===================================================================
class Figure1Generator:
    """Generate the 5-panel (2+3) Nature Cities Figure 1."""

    def __init__(self, csv_path, output_dir='./figures'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.d_latest = None
        self.city_stats = None
        self.map_stations = None
        self.map_routes = None
        self.map_buildings = None
        self.has_map_data = False
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print(
            f"  Records: {len(self.df):,}  "
            f"Years: {self.df['year'].min()}-{self.df['year'].max()}  "
            f"Cities: {self.df['city'].nunique()}"
        )
        self.df['tier'] = self.df['city'].apply(get_city_tier)
        self.d_latest = self.df[self.df['year'] == self.df['year'].max()].copy()

        s1_med = self.d_latest['served_population'].median()
        s2_med = self.d_latest['cumulative_opportunities'].median()
        self.d_latest['type_2d'] = [
            ('H' if s1 > s1_med else 'L') + ('H' if s2 > s2_med else 'L')
            for s1, s2 in zip(
                self.d_latest['served_population'],
                self.d_latest['cumulative_opportunities'],
            )
        ]

        cs = (
            self.d_latest.groupby('city')
            .agg(
                n=('station_id', 'count'),
                S1_mean=('served_population', 'mean'),
                S1_std=('served_population', 'std'),
                S2_mean=('cumulative_opportunities', 'mean'),
                S2_std=('cumulative_opportunities', 'std'),
                tier=('tier', 'first'),
            )
            .reset_index()
        )
        scaler = MinMaxScaler()
        cs['S1_norm'] = scaler.fit_transform(cs[['S1_mean']])
        cs['S2_norm'] = scaler.fit_transform(cs[['S2_mean']])
        cs['n_norm'] = scaler.fit_transform(cs[['n']])
        self.city_stats = cs

        self._load_maps()
        return self

    def _load_maps(self):
        if not HAS_GEOPANDAS:
            return
        self.map_stations, self.map_routes, self.map_buildings = load_map_data()
        if self.map_stations is not None:
            self.has_map_data = True
            tgt = self.map_stations.crs
            if self.map_routes is not None and self.map_routes.crs != tgt:
                self.map_routes = self.map_routes.to_crs(tgt)
            if self.map_buildings is not None and self.map_buildings.crs != tgt:
                self.map_buildings = self.map_buildings.to_crs(tgt)

    # ==============================================================
    # Panel a - Scale vs S1 scatter (ENHANCED, absorbs former panel b)
    # ==============================================================
    def _plot_panel_a(self, ax):
        cs = self.city_stats

        # --- scatter by tier ---
        for t in TIER_ORDER:
            d = cs[cs['tier'] == t]
            ax.scatter(
                d['n'], d['S1_mean'] / 1e3,
                c=TIER_COLORS[t], s=35, alpha=0.8, label=t,
                edgecolors='white', linewidths=0.3, zorder=5,
            )

        # --- regression line + R2 ---
        x = cs['n'].values
        y = cs['S1_mean'].values / 1e3
        r, p = stats.pearsonr(x, y)
        r2 = r ** 2
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, np.poly1d(np.polyfit(x, y, 1))(xl),
                '--', color='#888', lw=1, alpha=0.7)
        ax.text(
            0.95, 0.95, f'$R^2$ = {r2:.2f}',
            transform=ax.transAxes, fontsize=6, fontweight='bold',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCC', lw=0.5),
        )

        # --- Enhanced city annotations (absorbing former panel b) ---
        # Each entry: city -> (offset_x, offset_y)
        # Labels auto-generated with station count and S1 value
        annotate_cities = {
            'Lanzhou':   (15, 2),
            'Suzhou':    (-10, -14),
            'Shenzhen':  (12, 5),
            'Shanghai':  (-15, 8),
            'Beijing':   (-15, -10),
            'Guangzhou': (12, -8),
        }
        for city, (ox, oy) in annotate_cities.items():
            row = cs[cs['city'] == city]
            if len(row) == 0:
                continue
            cx_val = row['n'].values[0]
            cy_val = row['S1_mean'].values[0] / 1e3

            label = f'{city}\n({int(cx_val)} stn, S1={cy_val:.0f}k)'

            ax.annotate(
                label,
                (cx_val, cy_val),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=5, style='italic',
                arrowprops=dict(arrowstyle='-', color='#888', lw=0.5),
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#DDD',
                          lw=0.3, alpha=0.85),
            )

        
        ax.set_xlabel('Number of stations')
        ax.set_ylabel('Mean S1 (x1,000 persons)')
        ax.set_xlim(0, max(x) * 1.18)
        ax.legend(
            loc='lower right', fontsize=5, markerscale=0.8,
            title='City tier', title_fontsize=5.5,
        )
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)
        ax.text(-0.10, 1.08, 'a', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
        print(f"    R2 = {r2:.4f}")

    # ==============================================================
    # Panel b - Station-level S1 CCDF (formerly panel c)
    # ==============================================================
    def _plot_panel_b(self, ax):
        dl = self.d_latest

        for t in TIER_ORDER:
            td = dl[dl['tier'] == t]['served_population'].values
            td = td[td > 0]
            if len(td) == 0:
                continue
            td_sorted = np.sort(td)[::-1]
            rank = np.arange(1, len(td_sorted) + 1)
            ccdf = rank / len(td_sorted)
            ax.plot(
                td_sorted / 1e3, ccdf,
                color=TIER_COLORS[t], lw=1.2, alpha=0.85,
                label=f'{t} (n={len(td):,})',
            )

        ax.set_xlabel('S1 (x1,000 persons per station)', fontsize=7)
        ax.set_ylabel('P(X >= x)', fontsize=7)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(
            loc='upper right', fontsize=4.5,
            title='City tier', title_fontsize=5.5,
        )
        ax.grid(True, alpha=0.15, ls='-', lw=0.3)

        n_total = len(dl)
        ax.text(
            0.97, 0.55,
            f'N = {n_total:,}',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5',
                      ec='#CCC', lw=0.5),
        )
        ax.text(-0.10, 1.08, 'b', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # ==============================================================
    # Panel c/d - Station map (GIS) with card fallback
    # ==============================================================
    def _plot_station_panel(self, ax, station_key, panel_label):
        cfg = STATIONS_CONFIG[station_key]
        if self.has_map_data:
            sp = get_station_point(self.map_stations, cfg['name_cn'])
            if sp is not None:
                self._plot_station_map(ax, cfg, panel_label, sp)
                return
        self._plot_station_card(ax, cfg, panel_label)

    def _plot_station_map(self, ax, cfg, label, station_point):
        tgt = self.map_stations.crs
        buf = create_buffer(station_point, BUFFER_RADIUS, tgt)
        bds = buf.bounds
        cx, cy = (bds[0] + bds[2]) / 2, (bds[1] + bds[3]) / 2
        rad = (bds[2] - bds[0]) / 2
        dr = rad * DISPLAY_MARGIN
        cb = create_buffer(
            station_point, BUFFER_RADIUS * BUILDING_CLIP_MARGIN * 1.05, tgt
        )
        bld = gpd.clip(self.map_buildings, cb)
        rts = gpd.clip(self.map_routes, cb)
        sts = deduplicate_stations_gdf(
            gpd.clip(self.map_stations, buf),
            station_point, STATION_MERGE_THRESHOLD,
        )

        ax.set_xlim(cx - dr, cx + dr)
        ax.set_ylim(cy - dr, cy + dr)

        ax.add_patch(Circle(
            (cx, cy), rad,
            fc='none', ec=MAP_COLORS['buffer_edge'],
            lw=1.5, ls='--', alpha=0.9, zorder=2,
        ))
        if len(bld) > 0:
            bld.plot(
                ax=ax, fc=MAP_COLORS['building'],
                ec=MAP_COLORS['building_edge'],
                lw=0.1, alpha=0.85, zorder=3,
            )
        if len(rts) > 0:
            rts.plot(ax=ax, color=MAP_COLORS['metro_line'],
                     lw=2.5, alpha=0.9, zorder=4)
        for _, r in sts.iterrows():
            if r.geometry is None:
                continue
            im = r.get('is_main', False)
            ax.add_patch(Circle(
                (r.geometry.x, r.geometry.y),
                rad * (0.05 if im else 0.03),
                fc=MAP_COLORS['station_main'] if im else MAP_COLORS['station_other'],
                ec='white', lw=1.5 if im else 1, zorder=7,
            ))
        ax.text(
            cx, cy + rad * 0.08, cfg['name_en'],
            fontsize=6, ha='center', va='bottom',
            color=MAP_COLORS['station_name'], fontweight='bold', zorder=10,
        )
        tc = FOUR_TYPE_COLORS.get(cfg['type'], '#888')
        ax.text(
            0.02, 0.98, cfg['type'],
            transform=ax.transAxes, fontsize=6, fontweight='bold',
            va='top', ha='left', color='white', zorder=10,
            bbox=dict(boxstyle='round,pad=0.2', fc=tc, ec='none'),
        )
        # Scale bar
        sx, sy, sl = cx - dr * 0.85, cy - dr * 0.82, rad * 0.5
        ax.add_patch(Rectangle(
            (sx - rad * 0.03, sy - rad * 0.06),
            sl + rad * 0.06, rad * 0.18,
            fc='white', ec='#CCC', lw=0.5, alpha=0.95, zorder=9,
        ))
        ax.plot([sx, sx + sl], [sy, sy],
                color=MAP_COLORS['text'], lw=1.5, zorder=10)
        for s in [sx, sx + sl]:
            ax.plot([s, s], [sy - rad * 0.015, sy + rad * 0.015],
                    color=MAP_COLORS['text'], lw=1, zorder=10)
        ax.text(sx + sl / 2, sy + rad * 0.04, '500 m',
                ha='center', va='bottom', fontsize=5,
                color=MAP_COLORS['text'], zorder=10)
        # North arrow
        nx, ny, al = cx + dr * 0.75, cy + dr * 0.75, rad * 0.08
        ax.add_patch(Circle(
            (nx, ny), rad * 0.12,
            fc='white', ec='#CCC', lw=0.5, alpha=0.9, zorder=9,
        ))
        ax.annotate(
            '', xy=(nx, ny + al), xytext=(nx, ny - al * 0.3),
            arrowprops=dict(arrowstyle='->', color=MAP_COLORS['text'], lw=1),
            zorder=10,
        )
        ax.text(nx, ny + al * 1.2, 'N', ha='center', va='bottom',
                fontsize=5, fontweight='bold',
                color=MAP_COLORS['text'], zorder=10)
        # Metric inset
        ax.text(
            nx, ny - rad * 0.22,
            f"S1: {cfg['S1']:,}\nS2: {cfg['S2']:,}",
            ha='center', va='top', fontsize=5.5, style='italic',
            color=MAP_COLORS['text'], zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCC', lw=0.5),
        )
        # Legend
        elems = [
            mpatches.Patch(
                fc=MAP_COLORS['building'], ec=MAP_COLORS['building_edge'],
                lw=0.5, label='Buildings',
            ),
            Line2D([0], [0], color=MAP_COLORS['metro_line'], lw=2,
                   label='Metro line'),
            Line2D([0], [0], marker='o', color='w',
                   mfc=MAP_COLORS['station_main'], mec='white',
                   ms=4.5, mew=1, label='Target stn.', ls='None'),
            Line2D([0], [0], marker='o', color='w',
                   mfc=MAP_COLORS['station_other'], mec='white',
                   ms=3.5, mew=1, label='Other stn.', ls='None'),
            Line2D([0], [0], color=MAP_COLORS['buffer_edge'], lw=1.5,
                   ls='--', label=f'Buffer ({BUFFER_RADIUS / 1000:.0f} km)'),
        ]
        leg = ax.legend(
            handles=elems, loc='lower right', fontsize=4.5,
            frameon=True, facecolor='white', edgecolor='#CCC',
            framealpha=0.95, handlelength=1.5, handletextpad=0.4,
            borderpad=0.5, labelspacing=0.6,
        )
        leg.set_zorder(15)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(-0.05, 1.08, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    def _plot_station_card(self, ax, cfg, label):
        """Fallback info card when GIS data is unavailable."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.text(-0.05, 1.08, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

        stype = cfg['type']
        color = FOUR_TYPE_COLORS.get(stype, '#888')

        ax.add_patch(FancyBboxPatch(
            (0.3, 0.8), 9.4, 8.5,
            boxstyle="round,pad=0.1", fc='#FAFAFA', ec='#CCC', lw=0.5,
        ))
        ax.add_patch(FancyBboxPatch(
            (8.2, 8.4), 1.2, 0.55,
            boxstyle="round,pad=0.02", fc=color, ec='none',
        ))
        ax.text(8.8, 8.67, stype, fontsize=7, fontweight='bold',
                color='white', ha='center', va='center')
        ax.text(0.8, 8.5, cfg['name_en'], fontsize=9, fontweight='bold',
                ha='left', va='top', style='italic')

        # Quadrant mini-diagram
        qx, qy, qs = 0.8, 3.2, 3.2
        cell = qs / 2
        for dx, dy, ql in [(0, 0, 'LL'), (1, 0, 'HL'),
                            (0, 1, 'LH'), (1, 1, 'HH')]:
            a = 0.7 if ql == stype else 0.2
            ax.add_patch(Rectangle(
                (qx + dx * cell, qy + dy * cell), cell, cell,
                fc=FOUR_TYPE_COLORS[ql], alpha=a, ec='white',
            ))
            ax.text(
                qx + dx * cell + cell / 2,
                qy + dy * cell + cell / 2,
                ql, fontsize=6, ha='center', va='center',
                color='white' if ql == stype else '#999',
                fontweight='bold' if ql == stype else 'normal',
            )
        ax.text(qx + qs / 2, qy - 0.3, 'S1 ->',
                fontsize=5, ha='center', color='#666')
        ax.text(qx - 0.3, qy + qs / 2, 'S2 ->',
                fontsize=5, ha='center', va='center', rotation=90, color='#666')

        mx = 4.8
        ax.text(mx, 7.2, 'S1 (Service intensity):',
                fontsize=6, color='#555')
        ax.text(mx, 6.3, f'{cfg["S1"]:,}',
                fontsize=10, fontweight='bold', color=color)
        ax.text(mx, 5.3, 'S2 (Network access.):',
                fontsize=6, color='#555')
        ax.text(mx, 4.4, f'{cfg["S2"]:,}',
                fontsize=10, fontweight='bold', color=color)

        other_key = 'Huangcun' if cfg['name_en'] == 'Xierqi' else 'Xierqi'
        other = STATIONS_CONFIG[other_key]
        ratio = cfg['S2'] / other['S2']
        if ratio > 1:
            ax.text(mx, 3.2,
                    f'S2 = {ratio:.1f}x {other["name_en"]}',
                    fontsize=5.5, color='#888', style='italic')

    # ==============================================================
    # Panel e - HH/HL/LH/LL stacked bar (formerly panel f)
    # ==============================================================
    def _plot_panel_e(self, ax):
        dl = self.d_latest
        tc = dl.groupby(['tier', 'type_2d']).size().unstack(fill_value=0)
        tp = tc.div(tc.sum(axis=1), axis=0) * 100
        type_order = ['HH', 'HL', 'LH', 'LL']
        tp = tp.reindex(TIER_ORDER)
        for t in type_order:
            if t not in tp.columns:
                tp[t] = 0
        tp = tp[type_order]

        yp = np.arange(len(TIER_ORDER))
        left = np.zeros(len(TIER_ORDER))
        for typ in type_order:
            vals = tp[typ].values
            ax.barh(
                yp, vals, left=left,
                color=FOUR_TYPE_COLORS[typ], edgecolor='white',
                lw=0.3, label=typ, height=0.65,
            )
            for i, (v, l) in enumerate(zip(vals, left)):
                if v > 12:
                    ax.text(
                        l + v / 2, i, f'{v:.1f}%',
                        ha='center', va='center',
                        fontsize=4.5, fontweight='bold', color='white',
                    )
            left += vals

        npt = {t: len(dl[dl['tier'] == t]) for t in TIER_ORDER}
        ax.set_yticks(yp)
        ax.set_yticklabels(
            [f'{t}\n(n={npt[t]})' for t in TIER_ORDER], fontsize=5.5,
        )
        ax.set_xlabel('Percentage of stations (%)', fontsize=7)
        ax.set_xlim(0, 100)
        ax.legend(
            loc='upper right', fontsize=5, ncol=4,
            bbox_to_anchor=(1.0, 1.05),
            columnspacing=0.5, handletextpad=0.3,
        )
        ax.text(-0.15, 1.08, 'e', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # ==============================================================
    # Assemble figure - NEW 2+3 layout
    # ==============================================================
    def plot(self):
        print("\n=============================================")
        print("  Generating Figure 1 (V4, 2+3 layout)")
        print("=============================================")

        fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))

        # Nested GridSpec: row 1 = 2 cols (wide), row 2 = 3 cols
        gs_outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[1.1, 1.0],
            hspace=0.45,
            top=0.94, bottom=0.06, left=0.08, right=0.96,
        )

        # Row 1: 2 wide panels
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_outer[0],
            wspace=0.35,
        )

        # Row 2: 3 panels
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_outer[1],
            wspace=0.35,
        )

        # Panel a - Scale vs S1 (enhanced)
        ax_a = fig.add_subplot(gs_top[0, 0])
        self._plot_panel_a(ax_a)
        print("  Panel a - Scale vs S1 scatter (enhanced)")

        # Panel b - CCDF (formerly c)
        ax_b = fig.add_subplot(gs_top[0, 1])
        self._plot_panel_b(ax_b)
        print("  Panel b - Station S1 CCDF")

        # Panel c - Xierqi (formerly d)
        ax_c = fig.add_subplot(gs_bot[0, 0])
        self._plot_station_panel(ax_c, 'Xierqi', 'c')
        print("  Panel c - Xierqi (HH)")

        # Panel d - Huangcun (formerly e)
        ax_d = fig.add_subplot(gs_bot[0, 1])
        self._plot_station_panel(ax_d, 'Huangcun', 'd')
        print("  Panel d - Huangcun (HL)")

        # Panel e - Stacked bar (formerly f)
        ax_e = fig.add_subplot(gs_bot[0, 2])
        self._plot_panel_e(ax_e)
        print("  Panel e - Stacked bar")

        # Save
        fp_png = os.path.join(self.output_dir, 'Figure1_v4.png')
        fp_pdf = fp_png.replace('.png', '.pdf')
        plt.savefig(fp_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(fp_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {fp_png}")
        print(f"  Saved: {fp_pdf}")
        return fp_png


# ===================================================================
# CLI entry point
# ===================================================================
def main():
    output_dir = './figures'
    csv_path = (
        "/Users/liangwang/output/metro_accessibility_yearly/"
        "accessibility_all_years.csv"
    )
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"  CSV not found at {csv_path}, generating simulated data ...")
        csv_path = generate_simulated_data('./simulated_data.csv')

    gen = Figure1Generator(csv_path, output_dir)
    gen.load_data()
    gen.plot()


if __name__ == "__main__":
    main()