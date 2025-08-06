
import numpy as np
import pandas as pd
import talib
from logger_utils import Colors, print_colored
class DynamicWeightManager:
    """市场状态自适应的指标权重管理"""

    def __init__(self):
        self.indicators = {
            'RSI': {'type': '先行', 'base_weight': 0.20},
            'CCI': {'type': '先行', 'base_weight': 0.15},
            'Williams_R': {'type': '先行', 'base_weight': 0.15},
            'EMA': {'type': '滞后', 'base_weight': 0.20},
            'MACD': {'type': '滞后', 'base_weight': 0.30}
        }
        self.performance_history = {}
        print_colored("✅ 动态权重管理系统初始化完成", Colors.GREEN)

    def detect_market_regime(self, df):
        """识别当前市场状态"""
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        volatility = df['close'].pct_change().rolling(20).std()

        current_adx = adx.iloc[-1]
        current_vol = volatility.iloc[-1]

        if current_adx > 30:
            regime = '强势趋势'
        elif current_adx > 20:
            regime = '中等趋势'
        elif current_vol > volatility.quantile(0.8):
            regime = '高波动'
        else:
            regime = '震荡'

        print_colored(f"    📈 市场状态: {regime} (ADX={current_adx:.1f})", Colors.CYAN)
        return regime

    def calculate_adaptive_weights(self, market_regime):
        """根据市场状态调整权重"""
        regime_adjustments = {
            '强势趋势': {'先行': 0.7, '滞后': 1.3},
            '中等趋势': {'先行': 0.85, '滞后': 1.15},
            '震荡': {'先行': 1.3, '滞后': 0.7},
            '高波动': {'先行': 1.1, '滞后': 0.9}
        }

        adjustments = regime_adjustments[market_regime]
        weights = {}

        for indicator, props in self.indicators.items():
            base_weight = props['base_weight']
            ind_type = props['type']
            adjustment = adjustments[ind_type]

            # 应用市场状态调整
            weights[indicator] = base_weight * adjustment

        # 归一化权重使总和为1
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}

        print_colored(f"    📊 调整后权重:", Colors.INFO)
        for ind, weight in normalized_weights.items():
            print_colored(f"      • {ind}: {weight:.2%}", Colors.INFO)

        return normalized_weights