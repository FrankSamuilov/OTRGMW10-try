# technical_analysis_patch.py
# 技术分析修复补丁

import pandas as pd
import numpy as np


def patch_technical_analysis_function(original_func):
    """修复技术分析函数的装饰器"""

    def fixed_function(self, symbol):
        # 调用原函数
        result = original_func(self, symbol)

        # 检查并修复 RSI
        if 'rsi' in result and result['rsi'] == 50:
            # 尝试从 DataFrame 获取真实的 RSI
            try:
                df = self.get_market_data_sync(symbol)
                if not df.empty:
                    from indicators_module import calculate_optimized_indicators
                    df = calculate_optimized_indicators(df)

                    if 'RSI' in df.columns:
                        actual_rsi = df['RSI'].dropna()
                        if len(actual_rsi) > 0:
                            result['rsi'] = float(actual_rsi.iloc[-1])
                            print(f"[PATCH] 修正 RSI: {result['rsi']:.2f}")
            except:
                pass

        # 检查并修复成交量
        if 'volume_ratio' not in result or result.get('volume_ratio', 0) == 0:
            try:
                df = self.get_market_data_sync(symbol)
                if not df.empty and 'volume' in df.columns and len(df) >= 20:
                    volume_mean = df['volume'].rolling(20).mean().iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    if volume_mean > 0:
                        result['volume_ratio'] = current_volume / volume_mean
                        print(f"[PATCH] 修正成交量比率: {result['volume_ratio']:.2f}x")
            except:
                result['volume_ratio'] = 1.0

        return result

    return fixed_function


# 使用方法：
# 在 SimpleTradingBot 类初始化时：
# self._perform_technical_analysis = patch_technical_analysis_function(self._perform_technical_analysis)
