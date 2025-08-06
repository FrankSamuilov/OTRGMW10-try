# advanced_pattern_recognition.py
# 高级形态识别系统 - 包含经典技术形态和博弈论形态

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
from logger_utils import Colors, print_colored


class AdvancedPatternRecognition:
    """
    高级形态识别系统
    包含：经典技术形态、博弈论形态、市场微观结构形态
    """

    def __init__(self, logger=None):
        self.logger = logger

        # 形态权重配置
        self.pattern_weights = {
            # 经典反转形态
            'head_shoulders': 2.0,
            'inverse_head_shoulders': 2.5,
            'double_top': 1.8,
            'double_bottom': 2.0,
            'triple_top': 2.2,
            'triple_bottom': 2.5,

            # 持续形态
            'ascending_triangle': 1.5,
            'descending_triangle': 1.5,
            'symmetrical_triangle': 1.2,
            'bull_flag': 1.8,
            'bear_flag': 1.8,
            'pennant': 1.3,
            'wedge': 1.4,

            # 博弈论形态
            'stop_hunt': 2.0,
            'liquidity_grab': 2.2,
            'false_breakout': 1.8,
            'short_squeeze': 2.5,
            'long_squeeze': 2.5,
            'wyckoff_spring': 2.8,
            'wyckoff_upthrust': 2.5,

            # 市场微观结构
            'absorption': 1.6,
            'exhaustion': 1.8,
            'iceberg_accumulation': 2.0,
            'hidden_divergence': 1.7,
        }

        print_colored("✅ 高级形态识别系统初始化完成", Colors.GREEN)

    def detect_all_patterns(self, df: pd.DataFrame, current_price: float) -> Dict:
        """检测所有形态"""
        patterns_detected = {
            'classical': self._detect_classical_patterns(df),
            'game_theory': self._detect_game_theory_patterns(df),
            'microstructure': self._detect_microstructure_patterns(df),
            'combined_score': 0,
            'signals': []
        }

        # 计算综合得分
        total_score = 0
        for category in ['classical', 'game_theory', 'microstructure']:
            for pattern in patterns_detected[category]:
                score = pattern['confidence'] * self.pattern_weights.get(pattern['type'], 1.0)
                total_score += score
                patterns_detected['signals'].append({
                    'type': pattern['type'],
                    'score': score,
                    'direction': pattern['direction']
                })

        patterns_detected['combined_score'] = total_score
        return patterns_detected

    def _detect_classical_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """检测经典技术形态"""
        patterns = []

        # 获取局部极值点
        highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['low'].values, np.less, order=5)[0]

        # 1. 头肩顶/底
        hs_patterns = self._detect_head_shoulders(df, highs, lows)
        patterns.extend(hs_patterns)

        # 2. 双顶/双底
        double_patterns = self._detect_double_patterns(df, highs, lows)
        patterns.extend(double_patterns)

        # 3. 三角形态
        triangle_patterns = self._detect_triangles(df, highs, lows)
        patterns.extend(triangle_patterns)

        # 4. 旗形
        flag_patterns = self._detect_flags(df)
        patterns.extend(flag_patterns)

        return patterns

    def _detect_head_shoulders(self, df: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> List[Dict]:
        """检测头肩形态"""
        patterns = []

        # 头肩顶检测
        if len(highs) >= 5:
            for i in range(len(highs) - 4):
                # 获取5个高点
                points = highs[i:i + 5]
                if points[-1] < len(df) - 10:  # 确保有足够的未来数据

                    left_shoulder = df['high'].iloc[points[0]]
                    left_valley = df['low'].iloc[points[1]]
                    head = df['high'].iloc[points[2]]
                    right_valley = df['low'].iloc[points[3]]
                    right_shoulder = df['high'].iloc[points[4]]

                    # 头肩顶条件
                    if (head > left_shoulder and head > right_shoulder and
                            abs(left_shoulder - right_shoulder) / left_shoulder < 0.03 and
                            abs(left_valley - right_valley) / left_valley < 0.03):
                        neckline = (left_valley + right_valley) / 2
                        confidence = self._calculate_pattern_confidence(df, points[-1])

                        patterns.append({
                            'type': 'head_shoulders',
                            'direction': 'BEARISH',
                            'confidence': confidence,
                            'target': neckline - (head - neckline),
                            'stop_loss': head * 1.01,
                            'entry_point': neckline,
                            'timestamp': df.index[points[-1]]
                        })

        # 头肩底检测（反向逻辑）
        if len(lows) >= 5:
            for i in range(len(lows) - 4):
                points = lows[i:i + 5]
                if points[-1] < len(df) - 10:

                    left_shoulder = df['low'].iloc[points[0]]
                    left_peak = df['high'].iloc[points[1]]
                    head = df['low'].iloc[points[2]]
                    right_peak = df['high'].iloc[points[3]]
                    right_shoulder = df['low'].iloc[points[4]]

                    if (head < left_shoulder and head < right_shoulder and
                            abs(left_shoulder - right_shoulder) / left_shoulder < 0.03 and
                            abs(left_peak - right_peak) / left_peak < 0.03):
                        neckline = (left_peak + right_peak) / 2
                        confidence = self._calculate_pattern_confidence(df, points[-1])

                        patterns.append({
                            'type': 'inverse_head_shoulders',
                            'direction': 'BULLISH',
                            'confidence': confidence,
                            'target': neckline + (neckline - head),
                            'stop_loss': head * 0.99,
                            'entry_point': neckline,
                            'timestamp': df.index[points[-1]]
                        })

        return patterns

    def _detect_double_patterns(self, df: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> List[Dict]:
        """检测双顶/双底形态"""
        patterns = []

        # 双顶检测
        for i in range(len(highs) - 1):
            first_top = df['high'].iloc[highs[i]]
            second_top = df['high'].iloc[highs[i + 1]]

            # 两个顶部高度相近（差异小于2%）
            if abs(first_top - second_top) / first_top < 0.02:
                # 找到中间的低点
                valley_idx = df['low'].iloc[highs[i]:highs[i + 1]].idxmin()
                valley = df['low'].loc[valley_idx]

                if valley < first_top * 0.95:  # 回撤至少5%
                    confidence = self._calculate_pattern_confidence(df, highs[i + 1])

                    patterns.append({
                        'type': 'double_top',
                        'direction': 'BEARISH',
                        'confidence': confidence,
                        'target': valley - (first_top - valley),
                        'stop_loss': max(first_top, second_top) * 1.01,
                        'entry_point': valley,
                        'timestamp': df.index[highs[i + 1]]
                    })

        # 双底检测（类似逻辑）
        for i in range(len(lows) - 1):
            first_bottom = df['low'].iloc[lows[i]]
            second_bottom = df['low'].iloc[lows[i + 1]]

            if abs(first_bottom - second_bottom) / first_bottom < 0.02:
                peak_idx = df['high'].iloc[lows[i]:lows[i + 1]].idxmax()
                peak = df['high'].loc[peak_idx]

                if peak > first_bottom * 1.05:
                    confidence = self._calculate_pattern_confidence(df, lows[i + 1])

                    patterns.append({
                        'type': 'double_bottom',
                        'direction': 'BULLISH',
                        'confidence': confidence,
                        'target': peak + (peak - first_bottom),
                        'stop_loss': min(first_bottom, second_bottom) * 0.99,
                        'entry_point': peak,
                        'timestamp': df.index[lows[i + 1]]
                    })

        return patterns

    def _detect_triangles(self, df: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> List[Dict]:
        """检测三角形态"""
        patterns = []
        window = 20  # 检测窗口

        for i in range(window, len(df) - 5):
            recent_highs = df['high'].iloc[i - window:i].values
            recent_lows = df['low'].iloc[i - window:i].values

            # 计算趋势线斜率
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

            # 上升三角形：顶部水平，底部上升
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                patterns.append({
                    'type': 'ascending_triangle',
                    'direction': 'BULLISH',
                    'confidence': 0.7,
                    'target': recent_highs[-1] + (recent_highs[-1] - recent_lows[-1]),
                    'stop_loss': recent_lows[-1] * 0.98,
                    'entry_point': recent_highs[-1],
                    'timestamp': df.index[i]
                })

            # 下降三角形：底部水平，顶部下降
            elif abs(low_slope) < 0.001 and high_slope < -0.001:
                patterns.append({
                    'type': 'descending_triangle',
                    'direction': 'BEARISH',
                    'confidence': 0.7,
                    'target': recent_lows[-1] - (recent_highs[-1] - recent_lows[-1]),
                    'stop_loss': recent_highs[-1] * 1.02,
                    'entry_point': recent_lows[-1],
                    'timestamp': df.index[i]
                })

            # 对称三角形：顶部下降，底部上升
            elif high_slope < -0.001 and low_slope > 0.001:
                patterns.append({
                    'type': 'symmetrical_triangle',
                    'direction': 'NEUTRAL',
                    'confidence': 0.6,
                    'target': None,  # 方向未定
                    'stop_loss': None,
                    'entry_point': (recent_highs[-1] + recent_lows[-1]) / 2,
                    'timestamp': df.index[i]
                })

        return patterns

    def _detect_game_theory_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """检测博弈论相关形态"""
        patterns = []

        # 1. 止损猎杀
        stop_hunts = self._detect_stop_hunt(df)
        patterns.extend(stop_hunts)

        # 2. 流动性抓取
        liquidity_grabs = self._detect_liquidity_grab(df)
        patterns.extend(liquidity_grabs)

        # 3. 假突破
        false_breakouts = self._detect_false_breakout(df)
        patterns.extend(false_breakouts)

        # 4. 轧空/轧多
        squeezes = self._detect_squeezes(df)
        patterns.extend(squeezes)

        return patterns

    def _detect_stop_hunt(self, df: pd.DataFrame) -> List[Dict]:
        """检测止损猎杀形态"""
        patterns = []
        lookback = 20

        for i in range(lookback, len(df) - 1):
            # 检测下影线止损猎杀（看涨）
            support = df['low'].iloc[i - lookback:i].min()

            if (df['low'].iloc[i] < support * 0.995 and  # 突破支撑
                    df['close'].iloc[i] > support and  # 收回支撑上方
                    df['volume'].iloc[i] > df['volume'].iloc[i - 20:i].mean() * 1.5):  # 放量

                wick_ratio = (df['close'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i])
                if wick_ratio > 0.7:  # 下影线占比大
                    patterns.append({
                        'type': 'stop_hunt',
                        'direction': 'BULLISH',
                        'confidence': min(0.9, wick_ratio),
                        'target': df['high'].iloc[i - lookback:i].max(),
                        'stop_loss': df['low'].iloc[i] * 0.99,
                        'entry_point': df['close'].iloc[i],
                        'timestamp': df.index[i]
                    })

            # 检测上影线止损猎杀（看跌）
            resistance = df['high'].iloc[i - lookback:i].max()

            if (df['high'].iloc[i] > resistance * 1.005 and
                    df['close'].iloc[i] < resistance and
                    df['volume'].iloc[i] > df['volume'].iloc[i - 20:i].mean() * 1.5):

                wick_ratio = (df['high'].iloc[i] - df['close'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i])
                if wick_ratio > 0.7:
                    patterns.append({
                        'type': 'stop_hunt',
                        'direction': 'BEARISH',
                        'confidence': min(0.9, wick_ratio),
                        'target': df['low'].iloc[i - lookback:i].min(),
                        'stop_loss': df['high'].iloc[i] * 1.01,
                        'entry_point': df['close'].iloc[i],
                        'timestamp': df.index[i]
                    })

        return patterns

    def _detect_liquidity_grab(self, df: pd.DataFrame) -> List[Dict]:
        """检测流动性抓取形态"""
        patterns = []

        for i in range(50, len(df) - 1):
            # 检测向下流动性抓取后反转
            recent_low = df['low'].iloc[i - 50:i].min()

            if (df['low'].iloc[i] < recent_low * 0.998 and  # 创新低
                    df['close'].iloc[i] > df['open'].iloc[i] and  # 收阳
                    df['close'].iloc[i] > recent_low * 1.002):  # 快速收回

                # 计算抓取深度
                grab_depth = (recent_low - df['low'].iloc[i]) / recent_low

                if grab_depth > 0.002 and grab_depth < 0.02:  # 0.2%-2%的抓取深度
                    patterns.append({
                        'type': 'liquidity_grab',
                        'direction': 'BULLISH',
                        'confidence': min(0.9, grab_depth * 50),  # 深度越大置信度越高
                        'target': df['high'].iloc[i - 20:i].mean(),
                        'stop_loss': df['low'].iloc[i] * 0.995,
                        'entry_point': df['close'].iloc[i],
                        'timestamp': df.index[i]
                    })

        return patterns

    def _detect_squeezes(self, df: pd.DataFrame) -> List[Dict]:
        """检测轧空/轧多形态"""
        patterns = []

        # 需要RSI指标
        if 'RSI' not in df.columns:
            return patterns

        for i in range(20, len(df) - 1):
            # 轧空检测：超卖后快速反弹
            if (df['RSI'].iloc[i - 1] < 30 and  # 前期超卖
                    df['close'].iloc[i] > df['close'].iloc[i - 1] * 1.02 and  # 快速上涨2%+
                    df['volume'].iloc[i] > df['volume'].iloc[i - 20:i].mean() * 2):  # 成交量翻倍

                momentum = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]

                patterns.append({
                    'type': 'short_squeeze',
                    'direction': 'BULLISH',
                    'confidence': min(0.9, momentum * 20),
                    'target': df['close'].iloc[i] * 1.05,  # 目标5%
                    'stop_loss': df['low'].iloc[i - 1],
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

            # 轧多检测：超买后快速下跌
            if (df['RSI'].iloc[i - 1] > 70 and
                    df['close'].iloc[i] < df['close'].iloc[i - 1] * 0.98 and
                    df['volume'].iloc[i] > df['volume'].iloc[i - 20:i].mean() * 2):
                momentum = (df['close'].iloc[i - 1] - df['close'].iloc[i]) / df['close'].iloc[i - 1]

                patterns.append({
                    'type': 'long_squeeze',
                    'direction': 'BEARISH',
                    'confidence': min(0.9, momentum * 20),
                    'target': df['close'].iloc[i] * 0.95,
                    'stop_loss': df['high'].iloc[i - 1],
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

        return patterns

    def _detect_microstructure_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """检测市场微观结构形态"""
        patterns = []

        # 1. 吸收形态
        absorption = self._detect_absorption(df)
        patterns.extend(absorption)

        # 2. 衰竭形态
        exhaustion = self._detect_exhaustion(df)
        patterns.extend(exhaustion)

        return patterns

    def _detect_absorption(self, df: pd.DataFrame) -> List[Dict]:
        """检测吸收形态（大单吸收卖压/买压）"""
        patterns = []

        for i in range(10, len(df) - 1):
            # 买入吸收：价格不跌，成交量增加
            if (df['close'].iloc[i] > df['close'].iloc[i - 1] * 0.998 and  # 价格稳定
                    df['volume'].iloc[i] > df['volume'].iloc[i - 10:i].mean() * 1.8 and  # 成交量放大
                    df['low'].iloc[i] > df['low'].iloc[i - 5:i].min()):  # 低点抬高

                patterns.append({
                    'type': 'absorption',
                    'direction': 'BULLISH',
                    'confidence': 0.7,
                    'target': df['high'].iloc[i - 10:i].max(),
                    'stop_loss': df['low'].iloc[i],
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

        return patterns

    def _detect_exhaustion(self, df: pd.DataFrame) -> List[Dict]:
        """检测衰竭形态"""
        patterns = []

        for i in range(20, len(df) - 1):
            # 上涨衰竭：连续上涨后，成交量萎缩
            if (df['close'].iloc[i] > df['close'].iloc[i - 5] * 1.05 and  # 上涨5%
                    df['volume'].iloc[i] < df['volume'].iloc[i - 5:i].mean() * 0.7):  # 成交量萎缩

                patterns.append({
                    'type': 'exhaustion',
                    'direction': 'BEARISH',
                    'confidence': 0.6,
                    'target': df['close'].iloc[i] * 0.97,
                    'stop_loss': df['high'].iloc[i] * 1.01,
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

        return patterns

    def _detect_flags(self, df: pd.DataFrame) -> List[Dict]:
        """检测旗形形态"""
        patterns = []

        for i in range(30, len(df) - 5):
            # 检测前期强势上涨
            pre_move = df['close'].iloc[i - 10:i].pct_change().sum()

            # 牛旗：前期上涨后，小幅回调整理
            if pre_move > 0.1:  # 前期上涨10%+
                consolidation = df['close'].iloc[i:i + 5]
                if (consolidation.max() - consolidation.min()) / consolidation.mean() < 0.03:  # 整理幅度小于3%
                    patterns.append({
                        'type': 'bull_flag',
                        'direction': 'BULLISH',
                        'confidence': 0.75,
                        'target': consolidation.mean() * (1 + pre_move),  # 等幅上涨
                        'stop_loss': consolidation.min() * 0.98,
                        'entry_point': consolidation[-1],
                        'timestamp': df.index[i + 4]
                    })

            # 熊旗：前期下跌后，小幅反弹整理
            elif pre_move < -0.1:  # 前期下跌10%+
                consolidation = df['close'].iloc[i:i + 5]
                if (consolidation.max() - consolidation.min()) / consolidation.mean() < 0.03:
                    patterns.append({
                        'type': 'bear_flag',
                        'direction': 'BEARISH',
                        'confidence': 0.75,
                        'target': consolidation.mean() * (1 + pre_move),  # 等幅下跌
                        'stop_loss': consolidation.max() * 1.02,
                        'entry_point': consolidation[-1],
                        'timestamp': df.index[i + 4]
                    })

        return patterns

    def _calculate_pattern_confidence(self, df: pd.DataFrame, pattern_end_idx: int) -> float:
        """计算形态置信度"""
        # 基础置信度
        confidence = 0.5

        # 成交量确认
        if pattern_end_idx < len(df) - 1:
            vol_ratio = df['volume'].iloc[pattern_end_idx] / df['volume'].iloc[
                                                             pattern_end_idx - 20:pattern_end_idx].mean()
            if vol_ratio > 1.5:
                confidence += 0.2
            elif vol_ratio < 0.5:
                confidence -= 0.2

        # 趋势确认
        if pattern_end_idx >= 50:
            trend = df['close'].iloc[pattern_end_idx - 50:pattern_end_idx].pct_change().mean()
            if abs(trend) > 0.001:  # 明确趋势
                confidence += 0.1

        # 波动率环境
        if 'ATR' in df.columns and pattern_end_idx > 0:
            atr_ratio = df['ATR'].iloc[pattern_end_idx] / df['close'].iloc[pattern_end_idx]
            if atr_ratio < 0.02:  # 低波动环境
                confidence += 0.1
            elif atr_ratio > 0.05:  # 高波动环境
                confidence -= 0.1

        return max(0.1, min(0.95, confidence))

    def _detect_false_breakout(self, df: pd.DataFrame) -> List[Dict]:
        """检测假突破形态"""
        patterns = []
        lookback = 20

        for i in range(lookback, len(df) - 2):
            resistance = df['high'].iloc[i - lookback:i].max()
            support = df['low'].iloc[i - lookback:i].min()

            # 假突破上方
            if (df['high'].iloc[i] > resistance * 1.002 and  # 突破阻力
                    df['close'].iloc[i] < resistance and  # 收盘回落
                    df['close'].iloc[i + 1] < df['close'].iloc[i]):  # 次日继续下跌

                patterns.append({
                    'type': 'false_breakout',
                    'direction': 'BEARISH',
                    'confidence': 0.8,
                    'target': (resistance + support) / 2,
                    'stop_loss': df['high'].iloc[i] * 1.005,
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

            # 假突破下方
            if (df['low'].iloc[i] < support * 0.998 and  # 跌破支撑
                    df['close'].iloc[i] > support and  # 收盘回升
                    df['close'].iloc[i + 1] > df['close'].iloc[i]):  # 次日继续上涨

                patterns.append({
                    'type': 'false_breakout',
                    'direction': 'BULLISH',
                    'confidence': 0.8,
                    'target': (resistance + support) / 2,
                    'stop_loss': df['low'].iloc[i] * 0.995,
                    'entry_point': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })

        return patterns