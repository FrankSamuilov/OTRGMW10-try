# market_auction_analyzer.py
# 市场拍卖理论分析器 - POC、VAH/VAL、TPO等

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored


class MarketAuctionAnalyzer:
    """
    市场拍卖理论分析器
    实现Market Profile、Volume Profile、TPO等概念
    """

    def __init__(self, logger=None):
        self.logger = logger

        # 配置参数
        self.config = {
            'value_area_percentage': 0.70,  # 价值区域占总成交量的70%
            'poc_magnetic_range': 0.002,  # POC磁力范围0.2%
            'tpo_period': 30,  # TPO周期（分钟）
            'profile_bins': 100,  # 价格档位数量
        }

        print_colored("✅ 市场拍卖理论分析器初始化完成", Colors.GREEN)

    def analyze_market_structure(self, df: pd.DataFrame, lookback_hours: int = 24) -> Dict:
        """分析市场结构"""
        lookback_periods = lookback_hours * 60 // 5  # 转换为5分钟K线数量
        analysis_df = df.tail(lookback_periods).copy()

        # 1. 计算Volume Profile
        volume_profile = self._calculate_volume_profile(analysis_df)

        # 2. 计算Market Profile (TPO)
        market_profile = self._calculate_market_profile(analysis_df)

        # 3. 识别关键价位
        key_levels = self._identify_key_levels(volume_profile, market_profile)

        # 4. 分析市场状态
        market_state = self._analyze_market_state(analysis_df, key_levels)

        # 5. 生成交易信号
        signals = self._generate_auction_signals(analysis_df, key_levels, market_state)

        return {
            'volume_profile': volume_profile,
            'market_profile': market_profile,
            'key_levels': key_levels,
            'market_state': market_state,
            'signals': signals,
            'timestamp': df.index[-1]
        }

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """计算成交量分布"""
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min

        # 创建价格档位
        bins = np.linspace(price_min, price_max, self.config['profile_bins'])
        volume_distribution = np.zeros(len(bins) - 1)

        # 分配成交量到各个价格档位
        for idx, row in df.iterrows():
            # 计算K线跨越的价格档位
            low_bin = np.searchsorted(bins, row['low']) - 1
            high_bin = np.searchsorted(bins, row['high'])

            # 平均分配成交量
            if high_bin > low_bin:
                vol_per_bin = row['volume'] / (high_bin - low_bin)
                for i in range(max(0, low_bin), min(len(volume_distribution), high_bin)):
                    volume_distribution[i] += vol_per_bin

        # 找出POC（控制点）
        poc_idx = np.argmax(volume_distribution)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # 计算价值区域（VAH/VAL）
        total_volume = volume_distribution.sum()
        target_volume = total_volume * self.config['value_area_percentage']

        # 从POC开始向两边扩展，直到包含70%的成交量
        val_idx, vah_idx = self._calculate_value_area(volume_distribution, poc_idx, target_volume)

        val_price = (bins[val_idx] + bins[val_idx + 1]) / 2
        vah_price = (bins[vah_idx] + bins[vah_idx + 1]) / 2

        # 识别高低成交量节点（HVN/LVN）
        hvn_lvn = self._identify_volume_nodes(volume_distribution, bins)

        return {
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'volume_distribution': volume_distribution,
            'price_bins': bins,
            'hvn': hvn_lvn['hvn'],  # 高成交量节点
            'lvn': hvn_lvn['lvn'],  # 低成交量节点
            'value_area_volume_pct': target_volume / total_volume,
            'profile_skew': self._calculate_profile_skew(volume_distribution, poc_idx)
        }

    def _calculate_market_profile(self, df: pd.DataFrame) -> Dict:
        """计算市场概况（基于时间）"""
        # TPO (Time Price Opportunity) 分析
        tpo_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        tpo_profile = defaultdict(list)

        # 按时间段分组
        period_minutes = self.config['tpo_period']
        df['period'] = df.index.floor(f'{period_minutes}min')

        for period_idx, (period, group) in enumerate(df.groupby('period')):
            if period_idx >= len(tpo_letters):
                break

            letter = tpo_letters[period_idx]

            # 记录该时间段触及的价格
            period_high = group['high'].max()
            period_low = group['low'].min()

            # 将价格范围离散化
            price_levels = np.linspace(period_low, period_high, 20)
            for price in price_levels:
                tpo_profile[round(price, 2)].append(letter)

        # 计算IB (Initial Balance) - 前两个时间段
        ib_high = df[df.index < df.index[0] + pd.Timedelta(minutes=period_minutes * 2)]['high'].max()
        ib_low = df[df.index < df.index[0] + pd.Timedelta(minutes=period_minutes * 2)]['low'].min()
        ib_range = ib_high - ib_low

        # 判断市场类型
        market_type = self._classify_market_type(df, ib_high, ib_low, ib_range)

        return {
            'tpo_profile': dict(tpo_profile),
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'market_type': market_type,
            'period_count': len(df.groupby('period')),
            'range_extension': self._calculate_range_extension(df, ib_high, ib_low)
        }

    def _calculate_value_area(self, volume_distribution: np.ndarray, poc_idx: int,
                              target_volume: float) -> Tuple[int, int]:
        """计算价值区域的上下边界"""
        cumulative_volume = volume_distribution[poc_idx]
        val_idx = poc_idx
        vah_idx = poc_idx

        # 交替向上下扩展
        while cumulative_volume < target_volume:
            # 检查上方
            if vah_idx < len(volume_distribution) - 1:
                upper_vol = volume_distribution[vah_idx + 1]
            else:
                upper_vol = 0

            # 检查下方
            if val_idx > 0:
                lower_vol = volume_distribution[val_idx - 1]
            else:
                lower_vol = 0

            # 选择成交量更大的方向扩展
            if upper_vol >= lower_vol and vah_idx < len(volume_distribution) - 1:
                vah_idx += 1
                cumulative_volume += upper_vol
            elif val_idx > 0:
                val_idx -= 1
                cumulative_volume += lower_vol
            else:
                break

        return val_idx, vah_idx

    def _identify_volume_nodes(self, volume_distribution: np.ndarray,
                               price_bins: np.ndarray) -> Dict[str, List]:
        """识别高低成交量节点"""
        # 使用移动平均识别突出的成交量节点
        window = 5
        vol_ma = pd.Series(volume_distribution).rolling(window, center=True).mean()

        hvn = []  # 高成交量节点
        lvn = []  # 低成交量节点

        threshold_high = vol_ma.mean() + vol_ma.std()
        threshold_low = vol_ma.mean() - vol_ma.std() * 0.5

        for i in range(len(volume_distribution)):
            if pd.notna(vol_ma.iloc[i]):
                price = (price_bins[i] + price_bins[i + 1]) / 2

                if volume_distribution[i] > threshold_high:
                    hvn.append({
                        'price': price,
                        'volume': volume_distribution[i],
                        'strength': volume_distribution[i] / vol_ma.mean()
                    })
                elif volume_distribution[i] < threshold_low:
                    lvn.append({
                        'price': price,
                        'volume': volume_distribution[i],
                        'strength': 1 - (volume_distribution[i] / vol_ma.mean())
                    })

        return {'hvn': hvn, 'lvn': lvn}

    def _calculate_profile_skew(self, volume_distribution: np.ndarray, poc_idx: int) -> float:
        """计算成交量分布的偏斜度"""
        # 正值表示上方成交量更多，负值表示下方成交量更多
        upper_volume = volume_distribution[poc_idx:].sum()
        lower_volume = volume_distribution[:poc_idx].sum()

        total = upper_volume + lower_volume
        if total > 0:
            return (upper_volume - lower_volume) / total
        return 0

    def _classify_market_type(self, df: pd.DataFrame, ib_high: float,
                              ib_low: float, ib_range: float) -> str:
        """分类市场类型"""
        # 计算全天波动范围
        day_range = df['high'].max() - df['low'].min()

        # 范围扩展比例
        range_extension = day_range / ib_range if ib_range > 0 else 1

        # 收盘位置
        close_position = (df['close'].iloc[-1] - df['low'].min()) / day_range if day_range > 0 else 0.5

        if range_extension < 1.5:
            if 0.3 < close_position < 0.7:
                return "BALANCE"  # 平衡市场
            else:
                return "BALANCE_BREAKOUT"  # 平衡后突破
        elif range_extension > 2.0:
            if close_position > 0.8:
                return "TREND_UP"  # 上升趋势
            elif close_position < 0.2:
                return "TREND_DOWN"  # 下降趋势
            else:
                return "VOLATILE"  # 高波动
        else:
            return "NORMAL"  # 正常波动

    def _calculate_range_extension(self, df: pd.DataFrame, ib_high: float, ib_low: float) -> Dict:
        """计算范围扩展情况"""
        extensions = {
            'above_ib': 0,
            'below_ib': 0,
            'total': 0
        }

        # 计算突破IB的幅度
        max_high = df['high'].max()
        min_low = df['low'].min()

        if max_high > ib_high:
            extensions['above_ib'] = (max_high - ib_high) / (ib_high - ib_low)

        if min_low < ib_low:
            extensions['below_ib'] = (ib_low - min_low) / (ib_high - ib_low)

        extensions['total'] = extensions['above_ib'] + extensions['below_ib']

        return extensions

    def _identify_key_levels(self, volume_profile: Dict, market_profile: Dict) -> Dict:
        """识别关键价位"""
        current_price = volume_profile['price_bins'][-1]  # 使用最新价格

        key_levels = {
            'poc': volume_profile['poc'],
            'vah': volume_profile['vah'],
            'val': volume_profile['val'],
            'ib_high': market_profile['ib_high'],
            'ib_low': market_profile['ib_low'],
            'hvn_levels': [node['price'] for node in volume_profile['hvn']],
            'lvn_levels': [node['price'] for node in volume_profile['lvn']],
            'current_price': current_price,
            'poc_distance': abs(current_price - volume_profile['poc']) / volume_profile['poc'],
            'value_area_position': self._get_value_area_position(current_price, volume_profile)
        }

        # 添加关键支撑阻力
        key_levels['resistance'] = []
        key_levels['support'] = []

        # HVN作为支撑/阻力
        for hvn in volume_profile['hvn']:
            if hvn['price'] > current_price:
                key_levels['resistance'].append(hvn['price'])
            else:
                key_levels['support'].append(hvn['price'])

        # 排序
        key_levels['resistance'].sort()
        key_levels['support'].sort(reverse=True)

        return key_levels

    def _get_value_area_position(self, price: float, volume_profile: Dict) -> str:
        """判断价格在价值区域的位置"""
        if price > volume_profile['vah']:
            return "ABOVE_VALUE"
        elif price < volume_profile['val']:
            return "BELOW_VALUE"
        else:
            return "IN_VALUE"

    def _analyze_market_state(self, df: pd.DataFrame, key_levels: Dict) -> Dict:
        """分析市场状态"""
        current_price = df['close'].iloc[-1]

        # 计算各种市场指标
        volatility = df['close'].pct_change().std()
        trend = self._calculate_trend_strength(df)
        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean()

        # 判断市场阶段
        if key_levels['value_area_position'] == "IN_VALUE":
            if volatility < df['close'].pct_change().rolling(50).std().mean():
                phase = "BALANCE"  # 平衡阶段
            else:
                phase = "ROTATION"  # 轮动阶段
        else:
            if trend > 0.7:
                phase = "TRENDING"  # 趋势阶段
            else:
                phase = "DISCOVERY"  # 价格发现阶段

        # 判断拍卖状态
        if current_price > key_levels['vah'] and volume_trend > 1.2:
            auction_state = "INITIATIVE_BUYING"  # 主动买入
        elif current_price < key_levels['val'] and volume_trend > 1.2:
            auction_state = "INITIATIVE_SELLING"  # 主动卖出
        elif abs(current_price - key_levels['poc']) / key_levels['poc'] < 0.002:
            auction_state = "ACCEPTANCE"  # 接受当前价格
        else:
            auction_state = "REJECTION"  # 拒绝当前价格

        return {
            'phase': phase,
            'auction_state': auction_state,
            'trend_strength': trend,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'market_confidence': self._calculate_market_confidence(df, key_levels)
        }

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度"""
        # 使用线性回归斜率
        periods = len(df)
        if periods < 20:
            return 0

        x = np.arange(periods)
        y = df['close'].values

        # 归一化
        y_norm = (y - y.mean()) / y.std() if y.std() > 0 else y

        # 计算斜率
        slope = np.polyfit(x, y_norm, 1)[0]

        # 计算R²
        y_pred = np.polyval([slope, np.mean(y_norm) - slope * np.mean(x)], x)
        r_squared = 1 - (np.sum((y_norm - y_pred) ** 2) / np.sum((y_norm - y_norm.mean()) ** 2))

        return abs(slope) * r_squared

    def _calculate_market_confidence(self, df: pd.DataFrame, key_levels: Dict) -> float:
        """计算市场信心指数"""
        confidence = 0.5

        # 1. 价格在价值区域内加分
        if key_levels['value_area_position'] == "IN_VALUE":
            confidence += 0.2

        # 2. 接近POC加分
        if key_levels['poc_distance'] < 0.005:  # 0.5%以内
            confidence += 0.15

        # 3. 成交量递增加分
        if df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean():
            confidence += 0.1

        # 4. 波动率降低加分
        recent_volatility = df['close'].iloc[-10:].pct_change().std()
        avg_volatility = df['close'].pct_change().std()
        if recent_volatility < avg_volatility * 0.8:
            confidence += 0.05

        return min(1.0, confidence)

    def _generate_auction_signals(self, df: pd.DataFrame, key_levels: Dict,
                                  market_state: Dict) -> List[Dict]:
        """基于拍卖理论生成交易信号"""
        signals = []
        current_price = df['close'].iloc[-1]

        # 1. POC回归交易
        if abs(key_levels['poc_distance']) > 0.01:  # 偏离POC超过1%
            if current_price > key_levels['poc']:
                signals.append({
                    'type': 'POC_REVERSION',
                    'direction': 'SHORT',
                    'confidence': min(0.7 + key_levels['poc_distance'] * 10, 0.9),
                    'entry': current_price,
                    'target': key_levels['poc'],
                    'stop': current_price * 1.005,
                    'reason': 'Price above POC, expecting reversion'
                })
            else:
                signals.append({
                    'type': 'POC_REVERSION',
                    'direction': 'LONG',
                    'confidence': min(0.7 + key_levels['poc_distance'] * 10, 0.9),
                    'entry': current_price,
                    'target': key_levels['poc'],
                    'stop': current_price * 0.995,
                    'reason': 'Price below POC, expecting reversion'
                })

        # 2. 价值区域边界交易
        if key_levels['value_area_position'] == "ABOVE_VALUE":
            if market_state['auction_state'] == "REJECTION":
                signals.append({
                    'type': 'VALUE_AREA_REJECTION',
                    'direction': 'SHORT',
                    'confidence': 0.75,
                    'entry': current_price,
                    'target': key_levels['vah'],
                    'stop': current_price * 1.003,
                    'reason': 'Rejection above value area'
                })
        elif key_levels['value_area_position'] == "BELOW_VALUE":
            if market_state['auction_state'] == "REJECTION":
                signals.append({
                    'type': 'VALUE_AREA_REJECTION',
                    'direction': 'LONG',
                    'confidence': 0.75,
                    'entry': current_price,
                    'target': key_levels['val'],
                    'stop': current_price * 0.997,
                    'reason': 'Rejection below value area'
                })

        # 3. 低成交量节点突破
        for lvn in key_levels['lvn_levels']:
            if abs(current_price - lvn) / lvn < 0.002:  # 接近LVN
                # LVN通常会快速穿越
                if df['close'].iloc[-1] > df['close'].iloc[-2]:  # 上涨中
                    signals.append({
                        'type': 'LVN_BREAKOUT',
                        'direction': 'LONG',
                        'confidence': 0.65,
                        'entry': current_price,
                        'target': min(
                            [h for h in key_levels['hvn_levels'] if h > current_price] or [current_price * 1.01]),
                        'stop': lvn * 0.995,
                        'reason': 'Approaching LVN from below'
                    })

        # 4. 趋势日交易
        if market_state['phase'] == "TRENDING":
            if market_state['trend_strength'] > 0.7:
                # 强趋势跟随
                if df['close'].iloc[-1] > df['close'].iloc[-5]:
                    signals.append({
                        'type': 'TREND_CONTINUATION',
                        'direction': 'LONG',
                        'confidence': 0.8,
                        'entry': current_price,
                        'target': current_price * 1.02,
                        'stop': key_levels['val'],
                        'reason': 'Strong uptrend continuation'
                    })

        # 5. IB突破交易
        if current_price > key_levels['ib_high'] * 1.001:
            if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
                signals.append({
                    'type': 'IB_BREAKOUT',
                    'direction': 'LONG',
                    'confidence': 0.7,
                    'entry': current_price,
                    'target': key_levels['ib_high'] + (key_levels['ib_high'] - key_levels['ib_low']),
                    'stop': key_levels['ib_high'] * 0.995,
                    'reason': 'Initial Balance breakout with volume'
                })

        return signals