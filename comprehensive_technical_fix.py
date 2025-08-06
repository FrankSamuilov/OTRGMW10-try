# comprehensive_technical_fix.py
# 综合修复技术分析问题

import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored


def get_technical_indicator_safely(df, indicator_name, default_value=50):
    """安全获取技术指标值"""
    try:
        if indicator_name not in df.columns:
            print_colored(f"⚠️ {indicator_name} 列不存在，使用默认值 {default_value}", Colors.WARNING)
            return default_value

        # 获取最后一个有效值
        series = df[indicator_name].dropna()
        if len(series) == 0:
            print_colored(f"⚠️ {indicator_name} 没有有效值，使用默认值 {default_value}", Colors.WARNING)
            return default_value

        value = float(series.iloc[-1])
        print_colored(f"✅ {indicator_name}: {value:.2f}", Colors.SUCCESS)
        return value

    except Exception as e:
        print_colored(f"❌ 获取 {indicator_name} 失败: {e}，使用默认值 {default_value}", Colors.ERROR)
        return default_value


def analyze_technical_indicators(df):
    """分析技术指标并返回结果"""
    result = {
        'rsi': get_technical_indicator_safely(df, 'RSI', 50),
        'macd': get_technical_indicator_safely(df, 'MACD', 0),
        'macd_signal': get_technical_indicator_safely(df, 'MACD_signal', 0),
        'bb_position': 50,
        'volume_ratio': 1.0,
        'volume_surge': False
    }

    # 计算布林带位置
    if all(col in df.columns for col in ['close', 'BB_Upper', 'BB_Lower']):
        try:
            close = df['close'].iloc[-1]
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            if upper > lower:
                result['bb_position'] = ((close - lower) / (upper - lower)) * 100
        except:
            pass

    # 计算成交量比率
    if 'volume' in df.columns and len(df) >= 20:
        try:
            volume_mean = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            if volume_mean > 0:
                result['volume_ratio'] = current_volume / volume_mean
                result['volume_surge'] = result['volume_ratio'] > 1.5
        except:
            pass

    return result


# 在 _perform_technical_analysis 中使用：
# indicators = analyze_technical_indicators(df)
# rsi = indicators['rsi']
# volume_ratio = indicators['volume_ratio']
