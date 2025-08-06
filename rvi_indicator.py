"""
RVI (Relative Vigor Index) 相对波动指标
用于确认趋势强度和过滤假信号
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_rvi(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算 RVI 指标

    RVI = (Close - Open) / (High - Low) 的移动平均

    参数:
        df: 包含 OHLC 数据的 DataFrame
        period: RVI 周期（默认 10）

    返回:
        添加了 RVI 相关列的 DataFrame
    """

    # 计算分子：收盘价 - 开盘价（代表多空力量）
    numerator = df['close'] - df['open']

    # 计算分母：最高价 - 最低价（代表波动范围）
    denominator = df['high'] - df['low']

    # 避免除零错误
    denominator = denominator.replace(0, 0.0001)

    # 计算原始 RVI 值
    raw_rvi = numerator / denominator

    # 使用加权移动平均（给近期数据更高权重）
    # 权重：1, 2, 3, 4（对于4周期）
    weights = np.arange(1, period + 1)

    # 计算 RVI 主线
    rvi_values = []
    for i in range(len(df)):
        if i < period - 1:
            rvi_values.append(np.nan)
        else:
            # 加权平均
            weighted_sum = sum(raw_rvi.iloc[i - period + 1:i + 1] * weights) / sum(weights)
            rvi_values.append(weighted_sum)

    df['RVI'] = rvi_values

    # 计算信号线（RVI 的移动平均）
    df['RVI_Signal'] = df['RVI'].rolling(window=4).mean()

    # 计算 RVI 直方图（用于识别动量变化）
    df['RVI_Histogram'] = df['RVI'] - df['RVI_Signal']

    return df


def analyze_rvi_signals(df: pd.DataFrame) -> Dict[str, any]:
    """
    分析 RVI 信号

    返回:
        包含信号类型和强度的字典
    """

    if 'RVI' not in df.columns or 'RVI_Signal' not in df.columns:
        return {'signal': 'NEUTRAL', 'strength': 0}

    # 获取最新值
    current_rvi = df['RVI'].iloc[-1]
    current_signal = df['RVI_Signal'].iloc[-1]
    prev_rvi = df['RVI'].iloc[-2]
    prev_signal = df['RVI_Signal'].iloc[-2]

    # 检查交叉
    bullish_cross = prev_rvi <= prev_signal and current_rvi > current_signal
    bearish_cross = prev_rvi >= prev_signal and current_rvi < current_signal

    # 计算 RVI 强度
    rvi_strength = abs(current_rvi)

    # 检查背离
    price_trend = df['close'].iloc[-10:].pct_change().sum()
    rvi_trend = df['RVI'].iloc[-10:].diff().sum()

    bullish_divergence = price_trend < 0 and rvi_trend > 0
    bearish_divergence = price_trend > 0 and rvi_trend < 0

    # 生成信号
    signal = 'NEUTRAL'
    strength = 0

    if bullish_cross:
        signal = 'BULLISH'
        strength = min(rvi_strength * 2, 1.0)
    elif bearish_cross:
        signal = 'BEARISH'
        strength = min(rvi_strength * 2, 1.0)
    elif current_rvi > current_signal and current_rvi > 0:
        signal = 'BULLISH'
        strength = min(rvi_strength, 0.7)
    elif current_rvi < current_signal and current_rvi < 0:
        signal = 'BEARISH'
        strength = min(abs(rvi_strength), 0.7)

    # 背离增强信号
    if bullish_divergence and signal != 'BEARISH':
        signal = 'BULLISH'
        strength = min(strength + 0.3, 1.0)
    elif bearish_divergence and signal != 'BULLISH':
        signal = 'BEARISH'
        strength = min(strength + 0.3, 1.0)

    return {
        'signal': signal,
        'strength': strength,
        'rvi_value': current_rvi,
        'signal_value': current_signal,
        'has_divergence': bullish_divergence or bearish_divergence,
        'divergence_type': 'BULLISH' if bullish_divergence else 'BEARISH' if bearish_divergence else None
    }


def rvi_entry_filter(df: pd.DataFrame, position_side: str) -> Tuple[bool, str]:
    """
    使用 RVI 作为入场过滤器

    参数:
        df: 包含 RVI 数据的 DataFrame
        position_side: 'LONG' 或 'SHORT'

    返回:
        (是否允许入场, 原因)
    """

    rvi_analysis = analyze_rvi_signals(df)

    # 做多条件
    if position_side == 'LONG':
        if rvi_analysis['signal'] == 'BEARISH' and rvi_analysis['strength'] > 0.5:
            return False, "RVI 显示强烈看跌信号"
        elif rvi_analysis['signal'] == 'BULLISH' and rvi_analysis['strength'] > 0.7:
            return True, "RVI 确认强烈看涨信号"
        elif rvi_analysis['has_divergence'] and rvi_analysis['divergence_type'] == 'BEARISH':
            return False, "RVI 显示看跌背离"

    # 做空条件
    elif position_side == 'SHORT':
        if rvi_analysis['signal'] == 'BULLISH' and rvi_analysis['strength'] > 0.5:
            return False, "RVI 显示强烈看涨信号"
        elif rvi_analysis['signal'] == 'BEARISH' and rvi_analysis['strength'] > 0.7:
            return True, "RVI 确认强烈看跌信号"
        elif rvi_analysis['has_divergence'] and rvi_analysis['divergence_type'] == 'BULLISH':
            return False, "RVI 显示看涨背离"

    # 中性情况
    return True, "RVI 信号中性"


def rvi_exit_signal(df: pd.DataFrame, position_side: str, profit_pct: float) -> Tuple[bool, str]:
    """
    使用 RVI 生成出场信号

    返回:
        (是否应该出场, 原因)
    """

    rvi_analysis = analyze_rvi_signals(df)

    # 获取 RVI 趋势
    rvi_trend = df['RVI'].iloc[-5:].diff().mean()

    # 做多持仓
    if position_side == 'LONG':
        # 强烈反向信号
        if rvi_analysis['signal'] == 'BEARISH' and rvi_analysis['strength'] > 0.6:
            return True, f"RVI 转为看跌 (强度: {rvi_analysis['strength']:.2f})"

        # RVI 快速下降
        if rvi_trend < -0.1 and profit_pct > 1.0:
            return True, "RVI 快速下降，锁定利润"

        # 看跌背离
        if rvi_analysis['has_divergence'] and rvi_analysis['divergence_type'] == 'BEARISH':
            return True, "RVI 看跌背离，建议出场"

    # 做空持仓
    elif position_side == 'SHORT':
        # 强烈反向信号
        if rvi_analysis['signal'] == 'BULLISH' and rvi_analysis['strength'] > 0.6:
            return True, f"RVI 转为看涨 (强度: {rvi_analysis['strength']:.2f})"

        # RVI 快速上升
        if rvi_trend > 0.1 and profit_pct > 1.0:
            return True, "RVI 快速上升，锁定利润"

        # 看涨背离
        if rvi_analysis['has_divergence'] and rvi_analysis['divergence_type'] == 'BULLISH':
            return True, "RVI 看涨背离，建议出场"

    return False, "RVI 未显示出场信号"


# 将 RVI 集成到您的指标计算中
def add_rvi_to_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 RVI 添加到指标计算函数中
    """
    # 计算 RVI
    df = calculate_rvi(df, period=10)

    # 分析 RVI 信号
    rvi_signals = analyze_rvi_signals(df)

    # 添加信号强度列（用于后续分析）
    df['RVI_Strength'] = rvi_signals['strength']
    df['RVI_Signal_Type'] = rvi_signals['signal']

    return df