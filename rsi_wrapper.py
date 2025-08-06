# rsi_wrapper.py
# RSI 计算和显示包装器

import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored


def get_rsi_value(df, default=50.0):
    """安全获取 RSI 值"""
    try:
        if df is None or df.empty:
            return default

        if 'RSI' not in df.columns:
            print_colored("⚠️ RSI 列不存在", Colors.WARNING)
            return default

        rsi_series = df['RSI']

        # 尝试获取最后一个有效值
        last_valid_idx = rsi_series.last_valid_index()
        if last_valid_idx is not None:
            rsi_value = float(rsi_series.loc[last_valid_idx])
            if not np.isnan(rsi_value):
                return rsi_value

        # 如果没有有效值，返回默认值
        print_colored(f"⚠️ RSI 无有效值，使用默认值 {default}", Colors.WARNING)
        return default

    except Exception as e:
        print_colored(f"❌ 获取 RSI 值失败: {e}", Colors.ERROR)
        return default


def display_technical_summary(df):
    """显示技术分析总结，确保 RSI 正确显示"""

    # 获取各项指标
    rsi = get_rsi_value(df)

    # 获取其他指标...
    macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
    bb_position = 50  # 默认值

    if all(col in df.columns for col in ['close', 'BB_Upper', 'BB_Lower']):
        try:
            close = df['close'].iloc[-1]
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            if upper > lower:
                bb_position = ((close - lower) / (upper - lower)) * 100
        except:
            pass

    # 显示总结
    print_colored("\n📊 技术分析总结:", Colors.CYAN)
    print_colored(f"    • RSI: {rsi:.2f}", Colors.INFO)
    print_colored(f"    • MACD: {'金叉' if macd > 0 else '死叉'}", Colors.INFO)
    print_colored(f"    • 布林带位置: {bb_position:.1f}%", Colors.INFO)

    return {
        'rsi': rsi,
        'macd': macd,
        'bb_position': bb_position
    }
