# volume_spike_detector.py
"""
15分钟成交量突变检测系统
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from logger_utils import Colors, print_colored


class VolumeSpikDetector:
    """15分钟成交量突变检测器"""

    def __init__(self):
        self.spike_threshold = 1.5  # 1.5倍标准差为突变
        self.lookback_periods = 20  # 回看20个15分钟K线（5小时）

    def detect_volume_spike(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        检测15分钟K线的成交量突变

        返回:
            {
                'has_spike': bool,
                'spike_direction': 'UP'/'DOWN'/'NEUTRAL',
                'spike_strength': float (1-10),
                'momentum': float,
                'confidence': float (0-1)
            }
        """
        result = {
            'has_spike': False,
            'spike_direction': 'NEUTRAL',
            'spike_strength': 0.0,
            'momentum': 0.0,
            'confidence': 0.0
        }

        if len(df) < self.lookback_periods:
            return result

        # 获取最近的成交量数据
        volumes = df['volume'].tail(self.lookback_periods).values
        prices = df['close'].tail(self.lookback_periods).values

        # 计算成交量统计
        vol_mean = np.mean(volumes[:-1])  # 不包括最新的
        vol_std = np.std(volumes[:-1])
        current_vol = volumes[-1]

        # 计算Z分数
        if vol_std > 0:
            z_score = (current_vol - vol_mean) / vol_std
        else:
            z_score = 0

        # 检测突变
        if abs(z_score) > self.spike_threshold:
            result['has_spike'] = True

            # 计算价格动能（使用最近3根K线）
            if len(prices) >= 3:
                price_change = (prices[-1] - prices[-3]) / prices[-3] * 100
                result['momentum'] = price_change

                # 判断方向
                if z_score > 0 and price_change > 0:
                    result['spike_direction'] = 'UP'
                    result['confidence'] = min(z_score / 4, 1.0)
                elif z_score > 0 and price_change < 0:
                    result['spike_direction'] = 'DOWN'  # 放量下跌
                    result['confidence'] = min(z_score / 4, 1.0)
                elif z_score < 0:
                    result['spike_direction'] = 'NEUTRAL'  # 缩量
                    result['confidence'] = 0.3

            # 计算突变强度（1-10分）
            result['spike_strength'] = min(abs(z_score) * 2.5, 10.0)

        # 输出调试信息
        print_colored(f"\n      📊 15分钟成交量突变检测:", Colors.CYAN)
        print_colored(f"        当前成交量: {current_vol:,.0f}", Colors.INFO)
        print_colored(f"        平均成交量: {vol_mean:,.0f}", Colors.INFO)
        print_colored(f"        Z分数: {z_score:.2f}", Colors.INFO)

        if result['has_spike']:
            color = Colors.GREEN if result['spike_direction'] == 'UP' else Colors.RED
            print_colored(
                f"        ⚡ 检测到成交量突变! 方向: {result['spike_direction']}, 强度: {result['spike_strength']:.1f}",
                color)
            print_colored(f"        价格动能: {result['momentum']:.2f}%", color)

        return result