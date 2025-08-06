# volume_analysis_fix.py
# 修复成交量分析

def calculate_volume_surge(df, lookback=20):
    """计算成交量激增"""
    try:
        if 'volume' not in df.columns or len(df) < lookback:
            return 0.0, False

        # 计算平均成交量
        volume_mean = df['volume'].rolling(lookback).mean()
        current_volume = df['volume'].iloc[-1]
        mean_volume = volume_mean.iloc[-1]

        # 计算比率
        if mean_volume > 0:
            volume_ratio = current_volume / mean_volume
        else:
            volume_ratio = 1.0

        # 判断是否激增（超过平均值的1.5倍）
        is_surge = volume_ratio > 1.5

        return volume_ratio, is_surge

    except Exception as e:
        print(f"成交量分析错误: {e}")
        return 0.0, False


def get_volume_analysis_text(volume_ratio, is_surge):
    """获取成交量分析文本"""
    if is_surge:
        return f"⚡ 成交量激增 (比率: {volume_ratio:.1f}x)"
    else:
        return f"成交量正常 (比率: {volume_ratio:.1f}x)"
