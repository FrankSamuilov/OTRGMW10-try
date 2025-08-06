# 文件: fvg_module.py
"""
FVG (Fair Value Gap) 检测模块
从原SMC模块中提取的纯FVG功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from logger_utils import Colors, print_colored


def detect_fair_value_gap(df: pd.DataFrame, timeframe: str = "15m", lookback: int = 100,
                          sensitivity: float = 0.5, use_volume: bool = True) -> List[Dict[str, Any]]:
    """
    检测Fair Value Gaps (FVGs)

    参数:
        df: 包含OHLC数据的DataFrame
        timeframe: 时间框架
        lookback: 回溯检查的K线数量
        sensitivity: 灵敏度参数 (0-1)
        use_volume: 是否使用成交量分析

    返回:
        FVG列表，每个FVG包含详细信息
    """
    if df is None or len(df) < 4:
        print_colored("⚠️ 数据不足，无法检测FVG", Colors.WARNING)
        return []

    fvg_list = []
    min_gap_threshold = 0.001  # 最小缺口阈值

    # 计算ATR用于过滤小缺口
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
        min_gap_threshold = atr * (0.1 + (1 - sensitivity) * 0.3)
        print_colored(f"FVG最小缺口阈值: {min_gap_threshold:.6f}", Colors.INFO)

    # 限制回溯范围
    check_range = min(lookback, len(df) - 3)

    # 从最新的K线开始回溯检查
    for i in range(len(df) - 3, len(df) - 3 - check_range, -1):
        if i < 0:
            break

        # 获取三根K线数据
        candle1_low = df['low'].iloc[i]
        candle1_high = df['high'].iloc[i]
        candle2_high = df['high'].iloc[i + 1]
        candle2_low = df['low'].iloc[i + 1]
        candle2_volume = df['volume'].iloc[i + 1] if 'volume' in df.columns else 0
        candle3_low = df['low'].iloc[i + 2]
        candle3_high = df['high'].iloc[i + 2]

        # 检测看涨FVG (Bullish FVG)
        if candle1_low > candle3_high:
            gap_size = candle1_low - candle3_high

            if gap_size >= min_gap_threshold:
                # 检查是否已被填补
                filled = check_fvg_filled(df, i + 3, candle3_high, 'UP')

                fvg_data = {
                    'type': 'bullish',
                    'direction': 'UP',
                    'start_idx': i,
                    'end_idx': i + 2,
                    'upper_boundary': candle1_low,
                    'lower_boundary': candle3_high,
                    'gap_size': gap_size,
                    'gap_midpoint': candle3_high + gap_size / 2,
                    'is_filled': filled,
                    'age': len(df) - (i + 2),
                    'timeframe': timeframe
                }
                fvg_list.append(fvg_data)

        # 检测看跌FVG (Bearish FVG)
        elif candle3_low > candle1_high:
            gap_size = candle3_low - candle1_high

            if gap_size >= min_gap_threshold:
                # 检查是否已被填补
                filled = check_fvg_filled(df, i + 3, candle3_low, 'DOWN')

                fvg_data = {
                    'type': 'bearish',
                    'direction': 'DOWN',
                    'start_idx': i,
                    'end_idx': i + 2,
                    'upper_boundary': candle3_low,
                    'lower_boundary': candle1_high,
                    'gap_size': gap_size,
                    'gap_midpoint': candle1_high + gap_size / 2,
                    'is_filled': filled,
                    'age': len(df) - (i + 2),
                    'timeframe': timeframe
                }
                fvg_list.append(fvg_data)

    # 按照缺口大小排序
    fvg_list = sorted(fvg_list, key=lambda x: x['gap_size'], reverse=True)

    # 打印检测结果
    if fvg_list:
        print_colored(f"检测到 {len(fvg_list)} 个FVG", Colors.INFO)
        for i, fvg in enumerate(fvg_list[:3]):  # 只显示前3个
            status = "已填补" if fvg['is_filled'] else "未填补"
            fvg_color = Colors.GREEN if fvg['direction'] == 'UP' else Colors.RED
            print_colored(
                f"  {i + 1}. {fvg['type']} FVG: 缺口 {fvg['gap_size']:.6f} ({status})",
                fvg_color
            )

    return fvg_list


def check_fvg_filled(df: pd.DataFrame, start_idx: int, fill_level: float, direction: str) -> bool:
    """
    检查FVG是否已被填补

    参数:
        df: 价格数据
        start_idx: 开始检查的索引
        fill_level: 需要填补的价格水平
        direction: FVG方向 ('UP' 或 'DOWN')

    返回:
        是否已填补
    """
    for j in range(start_idx, len(df)):
        if direction == 'UP':
            if df['low'].iloc[j] <= fill_level:
                return True
        else:  # DOWN
            if df['high'].iloc[j] >= fill_level:
                return True
    return False


def analyze_fvg_strength(fvg: Dict[str, Any], current_price: float, atr: float) -> Dict[str, Any]:
    """
    分析FVG的强度和交易价值

    参数:
        fvg: FVG数据
        current_price: 当前价格
        atr: 平均真实波幅

    返回:
        FVG强度分析
    """
    analysis = {
        'strength': 0,
        'distance_ratio': 0,
        'age_factor': 0,
        'tradeable': False,
        'reason': ''
    }

    # 1. 计算到FVG的距离
    distance_to_fvg = abs(current_price - fvg['gap_midpoint'])
    distance_ratio = distance_to_fvg / current_price
    analysis['distance_ratio'] = distance_ratio

    # 2. 年龄因素（越新越好）
    age_factor = max(0, 1 - fvg['age'] / 50)  # 50根K线后价值降低
    analysis['age_factor'] = age_factor

    # 3. 缺口大小因素
    gap_ratio = fvg['gap_size'] / atr if atr > 0 else 0
    gap_strength = min(gap_ratio / 2, 1.0)  # ATR的2倍为满分

    # 4. 综合强度
    analysis['strength'] = (gap_strength * 0.5 + age_factor * 0.3 + (1 - distance_ratio) * 0.2)

    # 5. 判断是否可交易
    if not fvg['is_filled'] and distance_ratio < 0.02 and analysis['strength'] > 0.6:
        analysis['tradeable'] = True
        analysis['reason'] = 'FVG未填补且接近入场点'
    elif fvg['is_filled']:
        analysis['reason'] = 'FVG已填补'
    elif distance_ratio >= 0.02:
        analysis['reason'] = f'距离过远 ({distance_ratio:.1%})'
    else:
        analysis['reason'] = f'强度不足 ({analysis["strength"]:.2f})'

    return analysis


# 在 fvg_module.py 中添加

def analyze_fvg_with_auction_theory(fvg_list: List[Dict], auction_analysis: Dict) -> Dict[str, Any]:
    """
    结合拍卖理论分析FVG
    FVG在拍卖理论中代表价格快速移动留下的不平衡
    """
    enhanced_analysis = {
        'tradeable_fvgs': [],
        'fvg_confluence': False,
        'auction_alignment': False,
        'recommendation': None
    }

    try:
        # 检查FVG是否与拍卖理论的不平衡区域一致
        imbalance_zones = auction_analysis.get('imbalance_zones', [])

        for fvg in fvg_list:
            # 评估FVG的交易价值
            fvg_score = 0

            # 1. 未填补的FVG更有价值
            if not fvg['is_filled']:
                fvg_score += 0.3

            # 2. 与拍卖不平衡区域重合
            for imbalance in imbalance_zones:
                if (fvg['direction'] == imbalance['type'] and
                        abs(fvg['start_idx'] - imbalance['idx']) < 5):
                    fvg_score += 0.3
                    enhanced_analysis['auction_alignment'] = True

            # 3. 在价值区域附近
            value_areas = auction_analysis.get('value_areas', [])
            if value_areas:
                va = value_areas[0]
                if va['low'] <= fvg['gap_midpoint'] <= va['high']:
                    fvg_score += 0.2

            # 4. 年龄因素
            if fvg['age'] < 10:
                fvg_score += 0.2

            if fvg_score > 0.5:
                enhanced_fvg = fvg.copy()
                enhanced_fvg['trade_score'] = fvg_score
                enhanced_fvg['auction_aligned'] = enhanced_analysis['auction_alignment']
                enhanced_analysis['tradeable_fvgs'].append(enhanced_fvg)

        # 生成建议
        if enhanced_analysis['tradeable_fvgs']:
            best_fvg = max(enhanced_analysis['tradeable_fvgs'], key=lambda x: x['trade_score'])
            enhanced_analysis['recommendation'] = {
                'type': 'FVG_TRADE',
                'direction': best_fvg['direction'],
                'target': best_fvg['gap_midpoint'],
                'confidence': best_fvg['trade_score']
            }

    except Exception as e:
        print_colored(f"FVG拍卖理论分析错误: {e}", Colors.ERROR)

    return enhanced_analysis