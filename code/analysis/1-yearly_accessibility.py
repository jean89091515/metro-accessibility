#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地铁可达性逐年计算程序（内存优化版 - 修复版）

功能：
1. 逐年计算地铁站点的服务人口（基于WorldPop TIF）
2. 计算累计机会（30分钟可达范围内的总人口）
3. 计算交互可能（服务人口 × 累计机会）
4. 不保存中间数据，直接从TIF读取并处理
5. 支持分城市、分年份处理

修复内容：
- 删除重复的WorldPop裁剪调用
- 修正坐标系转换逻辑

作者：Urban Transportation Research Team
日期：2024-12-18
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')


class YearlyMetroAccessibility:
    """
    逐年地铁可达性计算器（内存优化版）
    """
    
    def __init__(self, metro_timeline_path, worldpop_dir, county_population_path, output_path):
        """
        初始化
        
        Parameters:
        -----------
        metro_timeline_path : str
            地铁站点时序数据路径
        worldpop_dir : str
            WorldPop TIF文件目录
        county_population_path : str
            区县人口数据路径
        output_path : str
            输出路径
        """
        self.metro_timeline_path = Path(metro_timeline_path)
        self.worldpop_dir = Path(worldpop_dir)
        self.county_population_path = Path(county_population_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 参数设置
        self.buffer_distance = 1000  # 服务半径（米）
        self.time_threshold = 30      # 时间阈值（分钟）
        self.avg_speed = 35           # 平均速度（km/h）
        
        # 城市映射
        self.target_cities = {
            'Beijing': '北京市', 'Shanghai': '上海市', 'Guangzhou': '广州市',
            'Shenzhen': '深圳市', 'Chengdu': '成都市', 'Chongqing': '重庆市',
            'Tianjin': '天津市', 'Wuhan': '武汉市', 'Hangzhou': '杭州市',
            'Nanjing': '南京市', 'Suzhou': '苏州市', 'Shenyang': '沈阳市',
            'Dalian': '大连市', 'Changchun': '长春市', 'Harbin': '哈尔滨市',
            'Xiamen': '厦门市', 'Qingdao': '青岛市', 'Ningbo': '宁波市',
            'Wuxi': '无锡市', 'Changsha': '长沙市', 'Zhengzhou': '郑州市',
            'Dongguan': '东莞市', 'Foshan': '佛山市', 'Kunming': '昆明市',
            'Nanchang': '南昌市', 'Fuzhou': '福州市', 'Changzhou': '常州市',
            'Hefei': '合肥市', 'Guiyang': '贵阳市', 'Shijiazhuang': '石家庄市',
            'Taiyuan': '太原市', 'Nanning': '南宁市', 'Lanzhou': '兰州市',
            'Urumqi': '乌鲁木齐市', 'Hohhot': '呼和浩特市', 'Nantong': '南通市',
            'Wenzhou': '温州市', 'Xuzhou': '徐州市', 'Shaoxing': '绍兴市',
            'Luoyang': '洛阳市', 'Wuhu': '芜湖市', 'Chuzhou': '滁州市'
        }
        
        # 加载基础数据
        self.metro_stations = None
        self.county_population = None
        
        # 统计信息
        self.stats = {
            'processed_years': [],
            'failed_years': [],
            'results_by_year': {}
        }
        
        # 累积所有年份的结果
        self.all_results = []
    
    def load_base_data(self):
        """加载基础数据"""
        print("=" * 70)
        print("【1】加载基础数据...")
        
        # 加载地铁站点时序数据
        print(f"  加载地铁站点: {self.metro_timeline_path}")
        self.metro_stations = gpd.read_file(self.metro_timeline_path)
        print(f"    ✓ 站点总数: {len(self.metro_stations)}")
        print(f"    ✓ 字段: {list(self.metro_stations.columns)}")
        
        # 检查开通年份字段
        if 'opening_yr' not in self.metro_stations.columns:
            raise ValueError("地铁数据缺少 opening_yr 字段")
        
        # 过滤掉未知年份的站点
        self.metro_stations = self.metro_stations[
            self.metro_stations['opening_yr'] != 9999
        ].copy()
        print(f"    ✓ 有效站点数（已知开通年份）: {len(self.metro_stations)}")
        
        # 统计各城市站点分布
        if 'city_en' in self.metro_stations.columns:
            city_counts = self.metro_stations['city_en'].value_counts()
            print(f"    ✓ 城市分布（前10）:")
            for city, count in city_counts.head(10).items():
                print(f"      {city}: {count}个站点")
        elif 'city_cn' in self.metro_stations.columns:
            city_counts = self.metro_stations['city_cn'].value_counts()
            print(f"    ✓ 城市分布（前10）:")
            for city, count in city_counts.head(10).items():
                print(f"      {city}: {count}个站点")
        
        # 加载区县人口数据
        print(f"  加载区县人口: {self.county_population_path}")
        self.county_population = gpd.read_file(self.county_population_path)
        print(f"    ✓ 区县数: {len(self.county_population)}")
        print(f"    ✓ 字段: {list(self.county_population.columns)}")
        
        # 识别关键字段
        print(f"    ✓ 字段识别:")
        for key_field in ['省', '市', '县', '区划码', '县代码', '名称']:
            if key_field in self.county_population.columns:
                unique_count = self.county_population[key_field].nunique()
                print(f"      {key_field}: {unique_count}个唯一值")
        
        print("  ✓ 基础数据加载完成\n")
    
    def get_worldpop_path(self, year):
        """获取指定年份的WorldPop文件路径"""
        tif_path = self.worldpop_dir / f"chn_ppp_{year}.tif"
        if not tif_path.exists():
            print(f"    ⚠ WorldPop文件不存在: {tif_path}")
            return None
        return tif_path
    
    def get_stations_by_year(self, year):
        """获取指定年份已开通的站点"""
        stations = self.metro_stations[
            self.metro_stations['opening_yr'] <= year
        ].copy()
        return stations
    
    def clip_worldpop_for_city(self, tif_path, city_boundary):
        """
        裁剪城市的WorldPop数据（内存优化）
        注意：WorldPop数据通常是WGS84坐标系(EPSG:4326)
        
        Returns:
        --------
        tuple: (array, transform, valid_mask, crs) 或 (None, None, None, None)
        """
        try:
            # 打开TIF文件获取其坐标系
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs
                print(f"      栅格坐标系: {raster_crs}")
                
                # 将边界转换到栅格坐标系
                if city_boundary.crs != raster_crs:
                    print(f"      转换边界坐标系: {city_boundary.crs} -> {raster_crs}")
                    city_boundary = city_boundary.to_crs(raster_crs)
                
                # 创建几何体列表
                geom_list = [geom.__geo_interface__ for geom in city_boundary.geometry]
                
                # 裁剪
                out_image, out_transform = mask(
                    src, geom_list, crop=True, nodata=-99999, filled=True
                )
                out_array = out_image[0]
                
                print(f"      裁剪后数组形状: {out_array.shape}")
                print(f"      数值范围: [{out_array.min():.2f}, {out_array.max():.2f}]")
                
                # 创建有效值掩膜（WorldPop中负值和NoData都是无效值）
                valid_mask = (out_array > 0) & (out_array < 1e10)
                valid_count = np.sum(valid_mask)
                
                print(f"      有效像素数: {valid_count:,}")
                
                if valid_count > 0:
                    valid_sum = out_array[valid_mask].sum()
                    print(f"      有效人口总数: {valid_sum:,.0f}")
                
                return out_array, out_transform, valid_mask, raster_crs
                
        except Exception as e:
            print(f"      ✗ 裁剪失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def reproject_raster_to_utm(self, array, transform, src_crs, center_lon, center_lat):
        """
        将栅格重投影到UTM坐标系
        
        Returns:
        --------
        tuple: (reprojected_array, new_transform, utm_crs)
        """
        # 确定UTM区
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f'EPSG:{32600 + utm_zone}' if center_lat >= 0 else f'EPSG:{32700 + utm_zone}'
        print(f"      使用投影: {utm_crs}")
        
        # 计算栅格边界
        rows, cols = array.shape
        left = transform.c
        top = transform.f
        right = left + cols * transform.a
        bottom = top + rows * transform.e
        src_bounds = (left, bottom, right, top)
        
        # 计算目标变换
        dst_transform, width, height = calculate_default_transform(
            src_crs, utm_crs, cols, rows, *src_bounds
        )
        
        # 执行重投影
        dst_array = np.zeros((height, width), dtype=array.dtype)
        reproject(
            source=array,
            destination=dst_array,
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=utm_crs,
            resampling=Resampling.bilinear
        )
        
        return dst_array, dst_transform, utm_crs
    
    def calculate_served_population_from_raster(self, stations, array, transform, valid_mask):
        """
        直接从栅格数据计算服务人口（不保存网格点）
        
        Parameters:
        -----------
        stations : GeoDataFrame
            站点数据（已转换为与栅格相同的投影坐标系）
        array : numpy.ndarray
            人口栅格数组
        transform : Affine
            仿射变换矩阵
        valid_mask : numpy.ndarray
            有效值掩膜
            
        Returns:
        --------
        Series : 每个站点的服务人口
        """
        served_pop = {}
        
        # 获取有效像素的坐标和人口值
        rows, cols = np.where(valid_mask)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs = np.array(xs)
        ys = np.array(ys)
        values = array[rows, cols]
        
        # 调试信息：检查坐标范围
        print(f"      栅格像素坐标范围: X=[{xs.min():.0f}, {xs.max():.0f}], Y=[{ys.min():.0f}, {ys.max():.0f}]")
        
        station_bounds = stations.total_bounds
        print(f"      站点坐标范围: X=[{station_bounds[0]:.0f}, {station_bounds[2]:.0f}], Y=[{station_bounds[1]:.0f}, {station_bounds[3]:.0f}]")
        
        # 创建点的空间索引（使用简单的网格索引提高效率）
        grid_size = self.buffer_distance * 2
        grid_dict = {}
        
        print(f"      构建人口网格索引（{len(xs):,} 个有效像素）...")
        for i in range(len(xs)):
            x, y, val = xs[i], ys[i], values[i]
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            key = (grid_x, grid_y)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append((x, y, val))
        
        # 为每个站点计算服务人口
        print(f"      计算每个站点的服务人口...")
        for idx, station in tqdm(stations.iterrows(), 
                                 total=len(stations), 
                                 desc="        站点处理",
                                 leave=False):
            station_point = station.geometry
            station_x, station_y = station_point.x, station_point.y
            
            # 确定需要检查的网格
            grid_x = int(station_x // grid_size)
            grid_y = int(station_y // grid_size)
            
            total_pop = 0
            
            # 检查周围9个网格
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (grid_x + dx, grid_y + dy)
                    if key in grid_dict:
                        for x, y, val in grid_dict[key]:
                            dist = np.sqrt((x - station_x)**2 + (y - station_y)**2)
                            if dist <= self.buffer_distance:
                                total_pop += val
            
            served_pop[idx] = total_pop
        
        return pd.Series(served_pop)
    
    def calculate_od_time_matrix(self, stations):
        """计算站点间的OD时间矩阵（基于距离估算）"""
        n_stations = len(stations)
        time_matrix = np.zeros((n_stations, n_stations))
        
        coords = np.array([[pt.x, pt.y] for pt in stations.geometry])
        
        for i in range(n_stations):
            # 计算到所有其他站点的距离
            distances = np.sqrt(
                (coords[:, 0] - coords[i, 0])**2 + 
                (coords[:, 1] - coords[i, 1])**2
            )
            
            # 距离转时间（考虑网络系数和换乘时间）
            network_distances = distances * 1.3
            travel_times = (network_distances / 1000) / self.avg_speed * 60
            transfer_times = (network_distances / 5000).astype(int) * 3
            
            time_matrix[i] = travel_times + transfer_times
        
        return pd.DataFrame(
            time_matrix,
            index=stations.index,
            columns=stations.index
        )
    
    def calculate_cumulative_opportunities(self, time_matrix, served_population):
        """计算累计机会"""
        cumulative_opp = {}
        
        for station_id in time_matrix.index:
            # 找出时间阈值内可达的站点
            accessible_stations = time_matrix.loc[station_id][
                time_matrix.loc[station_id] <= self.time_threshold
            ]
            
            # 累加这些站点的服务人口
            total_opp = 0
            for accessible_id in accessible_stations.index:
                if accessible_id in served_population.index:
                    total_opp += served_population.loc[accessible_id]
            
            cumulative_opp[station_id] = total_opp
        
        return pd.Series(cumulative_opp)
    
    def calculate_interaction_potential(self, served_population, cumulative_opportunities):
        """计算交互可能"""
        interaction = {}
        
        for station_id in served_population.index:
            if station_id in cumulative_opportunities.index:
                interaction[station_id] = (
                    served_population.loc[station_id] * 
                    cumulative_opportunities.loc[station_id]
                )
            else:
                interaction[station_id] = 0
        
        return pd.Series(interaction)
    
    def process_single_year(self, year):
        """
        处理单个年份的数据
        
        Parameters:
        -----------
        year : int
            年份
            
        Returns:
        --------
        DataFrame : 年度汇总结果
        """
        print(f"\n{'='*70}")
        print(f"【处理 {year} 年】")
        print(f"{'='*70}")
        
        # 1. 获取WorldPop文件
        worldpop_path = self.get_worldpop_path(year)
        if worldpop_path is None:
            print(f"  ✗ 跳过 {year} 年（无WorldPop数据）")
            self.stats['failed_years'].append(year)
            return None
        
        print(f"  ✓ WorldPop: {worldpop_path.name}")
        
        # 2. 获取该年份已开通的站点
        stations = self.get_stations_by_year(year)
        print(f"  ✓ 已开通站点数: {len(stations)}")
        
        if len(stations) == 0:
            print(f"  ✗ 跳过 {year} 年（无已开通站点）")
            self.stats['failed_years'].append(year)
            return None
        
        # 3. 按城市分组处理
        city_results = []
        
        # 按城市分组处理（使用city_en字段）
        if 'city_en' in stations.columns:
            cities_to_process = [
                (city, group) for city, group in stations.groupby('city_en')
            ]
        elif 'city_cn' in stations.columns:
            cities_to_process = [
                (city, group) for city, group in stations.groupby('city_cn')
            ]
        else:
            print("  ⚠ 站点数据缺少城市字段，将所有站点作为一个整体处理")
            cities_to_process = [('All', stations)]
        
        print(f"  处理 {len(cities_to_process)} 个城市/区域...")
        
        for city_name, city_stations in cities_to_process:
            print(f"\n  【{city_name}】({len(city_stations)} 个站点)")
            
            try:
                # 获取城市边界（从区县数据中提取）
                chinese_name = self.target_cities.get(city_name, city_name)
                
                # 检查字段名称
                city_field = None
                for field in ['市', '地级', 'city', 'CITY']:
                    if field in self.county_population.columns:
                        city_field = field
                        break
                
                if city_field is None:
                    print(f"    ⚠ 区县数据中未找到城市字段，尝试使用空间连接")
                    city_counties = None
                else:
                    city_counties = self.county_population[
                        self.county_population[city_field] == chinese_name
                    ]
                
                if city_counties is None or len(city_counties) == 0:
                    print(f"    ⚠ 未找到 {chinese_name} 的区县边界，使用空间连接...")
                    # 通过空间连接找到站点所在的区县
                    city_stations_temp = city_stations.to_crs(self.county_population.crs)
                    joined = gpd.sjoin(
                        city_stations_temp, 
                        self.county_population,
                        how='left',
                        predicate='within'
                    )
                    
                    county_code_field = None
                    for field in ['县代码', '区划码']:
                        if field in joined.columns:
                            county_code_field = field
                            break
                    
                    if county_code_field is None:
                        print(f"    ✗ 无法确定区县代码字段，跳过")
                        continue
                    
                    county_codes = joined[county_code_field].dropna().unique()
                    
                    if len(county_codes) == 0:
                        print(f"    ✗ 无法确定站点所属区县，跳过")
                        continue
                    
                    city_counties = self.county_population[
                        self.county_population[county_code_field].isin(county_codes)
                    ]
                    print(f"    ✓ 找到 {len(city_counties)} 个相关区县")
                
                # ========== 关键修复：只裁剪一次，然后重投影 ==========
                
                # 4. 裁剪WorldPop数据（只调用一次！）
                print(f"    裁剪WorldPop数据...")
                array, transform, valid_mask, raster_crs = self.clip_worldpop_for_city(
                    worldpop_path, city_counties
                )
                
                if array is None:
                    print(f"    ✗ 裁剪失败，跳过")
                    continue
                
                valid_pixels = np.sum(valid_mask)
                print(f"      ✓ 有效像素: {valid_pixels:,}")
                
                if valid_pixels == 0:
                    print(f"    ✗ 无有效人口数据，跳过")
                    continue
                
                # 5. 统一坐标系到米制（UTM）
                print(f"    统一坐标系...")
                
                # 获取站点中心点用于确定UTM区
                city_stations_wgs84 = city_stations.to_crs('EPSG:4326')
                bounds = city_stations_wgs84.total_bounds
                center_lon = (bounds[0] + bounds[2]) / 2
                center_lat = (bounds[1] + bounds[3]) / 2
                
                # 如果栅格是WGS84，需要重投影到米制坐标系
                raster_crs_str = str(raster_crs).upper()
                if 'EPSG:4326' in raster_crs_str or 'WGS' in raster_crs_str:
                    # 重投影栅格到UTM
                    array, transform, utm_crs = self.reproject_raster_to_utm(
                        array, transform, raster_crs, center_lon, center_lat
                    )
                    
                    # 更新有效掩膜
                    valid_mask = (array > 0) & (array < 1e10)
                    print(f"      重投影后有效像素: {np.sum(valid_mask):,}")
                    
                    if np.sum(valid_mask) > 0:
                        print(f"      重投影后人口总数: {array[valid_mask].sum():,.0f}")
                    
                    # 将站点转换到同一坐标系
                    city_stations_proj = city_stations.to_crs(utm_crs)
                else:
                    # 栅格已经是投影坐标系，直接使用
                    city_stations_proj = city_stations.to_crs(raster_crs)
                
                # ========== 修复结束：不再重复裁剪 ==========
                
                # 6. 计算服务人口（直接从栅格）
                print(f"    计算服务人口...")
                served_pop = self.calculate_served_population_from_raster(
                    city_stations_proj, array, transform, valid_mask
                )
                
                # 释放栅格数据内存
                del array, valid_mask
                gc.collect()
                
                # 7. 计算OD时间矩阵
                print(f"    计算OD时间矩阵...")
                time_matrix = self.calculate_od_time_matrix(city_stations_proj)
                
                # 8. 计算累计机会
                print(f"    计算累计机会...")
                cumulative_opp = self.calculate_cumulative_opportunities(
                    time_matrix, served_pop
                )
                
                # 9. 计算交互可能
                print(f"    计算交互可能...")
                interaction_pot = self.calculate_interaction_potential(
                    served_pop, cumulative_opp
                )
                
                # 10. 组合结果（保留原始站点信息）
                city_result_data = []
                for idx in city_stations.index:
                    station_info = city_stations.loc[idx]
                    result_row = {
                        'year': year,
                        'city': city_name,
                        'station_id': idx,
                        'stop_id': station_info.get('stop_id', idx),
                        'name_cn': station_info.get('name_cn', ''),
                        'name_en': station_info.get('name_en', ''),
                        'route_cn': station_info.get('route_cn', ''),
                        'opening_yr': station_info.get('opening_yr', year),
                        'served_population': served_pop.get(idx, 0),
                        'cumulative_opportunities': cumulative_opp.get(idx, 0),
                        'interaction_potential': interaction_pot.get(idx, 0),
                        'geometry': station_info.geometry
                    }
                    city_result_data.append(result_row)
                
                city_result = gpd.GeoDataFrame(city_result_data, crs=city_stations.crs)
                city_results.append(city_result)
                
                # 输出城市统计
                print(f"    ✓ 完成:")
                print(f"      平均服务人口: {served_pop.mean():,.0f}")
                print(f"      平均累计机会: {cumulative_opp.mean():,.0f}")
                print(f"      平均交互可能: {interaction_pot.mean():.2e}")
                
                # 清理内存
                del time_matrix, served_pop, cumulative_opp, interaction_pot
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 11. 合并所有城市结果
        if city_results:
            year_result = pd.concat(city_results, ignore_index=True)
            
            # 添加到累积结果中
            self.all_results.append(year_result)
            
            # 也保存单年结果（用于调试）
            year_csv = self.output_path / f"debug_accessibility_{year}.csv"
            year_result_df = pd.DataFrame(year_result.drop(columns='geometry'))
            year_result_df.to_csv(year_csv, index=False, encoding='utf-8-sig')
            print(f"\n  ✓ {year}年调试CSV: {year_csv}")
            
            # 保存统计信息
            year_stats = {
                'year': year,
                'total_stations': len(year_result),
                'avg_served_population': year_result['served_population'].mean(),
                'avg_cumulative_opportunities': year_result['cumulative_opportunities'].mean(),
                'avg_interaction_potential': year_result['interaction_potential'].mean(),
                'total_served_population': year_result['served_population'].sum(),
                'stations_with_population': (year_result['served_population'] > 0).sum()
            }
            
            print(f"\n  📊 {year}年统计:")
            print(f"      总站点数: {year_stats['total_stations']}")
            print(f"      有服务人口的站点: {year_stats['stations_with_population']}")
            print(f"      平均服务人口: {year_stats['avg_served_population']:,.0f}")
            print(f"      平均累计机会: {year_stats['avg_cumulative_opportunities']:,.0f}")
            
            self.stats['processed_years'].append(year)
            self.stats['results_by_year'][year] = year_stats
            
            return year_result
        else:
            print(f"  ✗ {year}年无有效结果")
            self.stats['failed_years'].append(year)
            return None
    
    def process_all_years(self, start_year=2000, end_year=2025):
        """
        处理所有年份
        
        Parameters:
        -----------
        start_year : int
            起始年份
        end_year : int
            结束年份
        """
        print("=" * 70)
        print("地铁可达性逐年计算程序")
        print("=" * 70)
        print(f"年份范围: {start_year} - {end_year}")
        print(f"缓冲半径: {self.buffer_distance}米")
        print(f"时间阈值: {self.time_threshold}分钟")
        print("=" * 70)
        
        # 加载基础数据
        self.load_base_data()
        
        # 逐年处理
        for year in range(start_year, end_year + 1):
            result = self.process_single_year(year)
        
        # 生成总结报告（使用累积的所有结果）
        self.create_summary_report(self.all_results)
    
    def create_summary_report(self, all_results):
        """生成总结报告"""
        print("\n" + "=" * 70)
        print("【生成总结报告】")
        
        if not all_results:
            print("  ✗ 无有效结果")
            return
        
        # 合并所有年份数据
        print("  合并所有年份数据...")
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True))
        
        print(f"  ✓ 总记录数: {len(combined_gdf):,}")
        print(f"  ✓ 年份范围: {combined_gdf['year'].min()} - {combined_gdf['year'].max()}")
        
        # 保存完整SHP数据
        full_shp_output = self.output_path / "accessibility_all_years.shp"
        try:
            combined_gdf.to_file(full_shp_output, encoding='utf-8')
            print(f"  ✓ 完整SHP数据: {full_shp_output}")
        except Exception as e:
            print(f"  ⚠ SHP保存失败: {e}")
            print(f"    尝试使用GBK编码...")
            try:
                combined_gdf.to_file(full_shp_output, encoding='gbk')
                print(f"  ✓ 完整SHP数据(GBK): {full_shp_output}")
            except Exception as e2:
                print(f"  ✗ SHP保存失败: {e2}")
        
        # 保存完整CSV数据
        full_csv_output = self.output_path / "accessibility_all_years.csv"
        combined_df = pd.DataFrame(combined_gdf.drop(columns='geometry'))
        combined_df.to_csv(full_csv_output, index=False, encoding='utf-8-sig')
        print(f"  ✓ 完整CSV数据: {full_csv_output}")
        
        # 生成年度统计
        yearly_stats = pd.DataFrame(self.stats['results_by_year']).T
        yearly_stats_file = self.output_path / "yearly_statistics.csv"
        yearly_stats.to_csv(yearly_stats_file, encoding='utf-8-sig')
        print(f"  ✓ 年度统计: {yearly_stats_file}")
        
        # 生成详细文本报告
        report_file = self.output_path / "summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("地铁可达性逐年分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("参数设置:\n")
            f.write(f"  服务半径: {self.buffer_distance}米\n")
            f.write(f"  时间阈值: {self.time_threshold}分钟\n")
            f.write(f"  平均速度: {self.avg_speed}公里/小时\n\n")
            
            f.write("处理统计:\n")
            f.write(f"  成功处理年份数: {len(self.stats['processed_years'])}\n")
            f.write(f"  失败年份数: {len(self.stats['failed_years'])}\n")
            f.write(f"  总站点记录数: {len(combined_gdf):,}\n\n")
            
            f.write("年度详细统计:\n")
            f.write("-" * 60 + "\n")
            for year in sorted(self.stats['processed_years']):
                stats = self.stats['results_by_year'][year]
                f.write(f"\n{year}年:\n")
                f.write(f"  站点总数: {stats['total_stations']}\n")
                f.write(f"  有服务人口站点数: {stats.get('stations_with_population', 'N/A')}\n")
                f.write(f"  平均服务人口: {stats['avg_served_population']:,.0f}\n")
                f.write(f"  平均累计机会: {stats['avg_cumulative_opportunities']:,.0f}\n")
                f.write(f"  平均交互可能: {stats['avg_interaction_potential']:.2e}\n")
                f.write(f"  总服务人口: {stats['total_served_population']:,.0f}\n")
            
            if self.stats['failed_years']:
                f.write(f"\n失败的年份: {self.stats['failed_years']}\n")
            
            # 添加城市统计
            f.write("\n\n城市统计:\n")
            f.write("-" * 60 + "\n")
            city_stats = combined_df.groupby('city').agg({
                'station_id': 'count',
                'served_population': 'mean',
                'cumulative_opportunities': 'mean',
                'interaction_potential': 'mean'
            }).round(0)
            city_stats.columns = ['站点数', '平均服务人口', '平均累计机会', '平均交互可能']
            f.write(city_stats.to_string())
        
        print(f"  ✓ 文本报告: {report_file}")
        
        # 数据质量检查
        print("\n  📊 数据质量检查:")
        zero_pop_count = (combined_df['served_population'] == 0).sum()
        zero_pop_pct = zero_pop_count / len(combined_df) * 100
        print(f"      服务人口为0的记录: {zero_pop_count:,} ({zero_pop_pct:.1f}%)")
        
        if zero_pop_pct > 50:
            print(f"      ⚠️ 警告: 超过50%的站点服务人口为0，请检查:")
            print(f"         1. WorldPop数据是否正确裁剪")
            print(f"         2. 坐标系转换是否正确")
            print(f"         3. 缓冲区距离设置是否合理")
        
        print("\n" + "=" * 70)
        print("✅ 处理完成！")
        print(f"  ✓ 成功处理: {len(self.stats['processed_years'])} 年")
        print(f"  ✓ 总站点-年份记录: {len(combined_gdf):,}")
        print(f"  ✓ 输出目录: {self.output_path}")
        print("=" * 70)


def main():
    """主函数"""
    
    # 配置路径
    metro_timeline_path = "/Users/liangwang/0-成果转化/01-论文/37-地铁时序数据集/4-提交数据集/metro_stations_timeline.shp"
    worldpop_dir = "/Users/liangwang/Downloads/worldpop"
    county_population_path = "/Users/liangwang/12-数据/46-一直七普空间人口数据/区县级一普至七普人口数据_立方数据学社/一普到七普人口数据/区县人口_2000_2025_插值.shp"
    output_path = "/Users/liangwang/output/metro_accessibility_yearly"
    
    # 创建计算器
    calculator = YearlyMetroAccessibility(
        metro_timeline_path=metro_timeline_path,
        worldpop_dir=worldpop_dir,
        county_population_path=county_population_path,
        output_path=output_path
    )
    
    # 执行计算（可指定年份范围）
    calculator.process_all_years(start_year=2000, end_year=2025)


if __name__ == "__main__":
    main()