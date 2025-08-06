import numpy as np
import pandas as pd
import talib
from logger_utils import Colors, print_colored


class TrendAwareRSI:
    """趋势感知的RSI指标"""

    def __init__(self, rsi_period=14, trend_period=50):
        self.rsi_period = rsi_period
        self.trend_period = trend_period
        print_colored("✅ 趋势感知RSI系统初始化完成", Colors.GREEN)

    def calculate_trend_strength(self, df):
        """计算趋势方向和强度"""
        df['ema_short'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=50)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # 确定趋势方向
        df['trend_direction'] = np.where(df['ema_short'] > df['ema_long'], 1, -1)

        # 分类趋势强度
        df['trend_strength'] = np.select(
            [df['adx'] > 30, df['adx'] > 20, df['adx'] <= 20],
            ['强势', '中等', '弱势']
        )
        return df

    def get_dynamic_thresholds(self, trend_direction, trend_strength):
        """根据趋势获取动态RSI阈值"""
        thresholds = {
            '强势上涨': {'超卖': 40, '超买': 80},
            '中等上涨': {'超卖': 35, '超买': 75},
            '弱势震荡': {'超卖': 30, '超买': 70},
            '中等下跌': {'超卖': 25, '超买': 65},
            '强势下跌': {'超卖': 20, '超买': 60}
        }

        if trend_direction == 1 and trend_strength == '强势':
            return thresholds['强势上涨']
        elif trend_direction == 1 and trend_strength == '中等':
            return thresholds['中等上涨']
        elif trend_direction == -1 and trend_strength == '强势':
            return thresholds['强势下跌']
        elif trend_direction == -1 and trend_strength == '中等':
            return thresholds['中等下跌']
        else:
            return thresholds['弱势震荡']

    def calculate_rsi_score(self, df):
        """计算考虑趋势的RSI得分"""
        df = self.calculate_trend_strength(df)
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)

        scores = []
        for idx in range(len(df)):
            if pd.isna(df['rsi'].iloc[idx]):
                scores.append(0)
                continue

            thresholds = self.get_dynamic_thresholds(
                df['trend_direction'].iloc[idx],
                df['trend_strength'].iloc[idx]
            )

            rsi_value = df['rsi'].iloc[idx]
            oversold = thresholds['超卖']
            overbought = thresholds['超买']

            # 根据趋势计算得分
            if df['trend_direction'].iloc[idx] == 1:  # 上涨趋势
                if rsi_value < oversold:
                    score = 100  # 强烈买入信号
                elif rsi_value > overbought:
                    score = -20  # 轻微警告，不是强烈卖出
                else:
                    # 在阈值之间缩放
                    score = 50 - 70 * (rsi_value - oversold) / (overbought - oversold)
            else:  # 下跌趋势
                if rsi_value > overbought:
                    score = -100  # 强烈卖出信号
                elif rsi_value < oversold:
                    score = 20  # 轻微机会，不是强烈买入
                else:
                    # 在阈值之间缩放
                    score = -50 + 70 * (overbought - rsi_value) / (overbought - oversold)

            scores.append(score)

        df['rsi_score'] = scores

        # 打印最新的RSI分析
        if len(df) > 0:
            latest = df.iloc[-1]
            print_colored(f"    📊 RSI分析:", Colors.CYAN)
            print_colored(f"      • RSI值: {latest['rsi']:.1f}", Colors.INFO)
            print_colored(
                f"      • 趋势: {'上涨' if latest['trend_direction'] == 1 else '下跌'} ({latest['trend_strength']})",
                Colors.INFO)
            print_colored(f"      • 动态阈值: 超卖={thresholds['超卖']}, 超买={thresholds['超买']}", Colors.INFO)
            print_colored(f"      • RSI得分: {latest['rsi_score']:.1f}", Colors.INFO)

        return df