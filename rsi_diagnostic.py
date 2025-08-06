# rsi_diagnostic.py
# RSI 诊断工具

import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored

def diagnose_rsi_calculation(df):
    """诊断 RSI 计算问题"""

    print_colored("\n=== RSI 诊断开始 ===", Colors.CYAN)

    # 1. 检查数据
    print_colored("\n1. 数据检查:", Colors.INFO)
    print(f"   - DataFrame 形状: {df.shape}")
    print(f"   - 列名: {df.columns.tolist()}")

    if 'close' in df.columns:
        print(f"   - close 列类型: {df['close'].dtype}")
        print(f"   - close 非空值数量: {df['close'].notna().sum()}")
        print(f"   - close 前5个值: {df['close'].head().tolist()}")

    # 2. 手动计算 RSI
    if 'close' in df.columns and len(df) >= 14:
        print_colored("\n2. 手动计算 RSI:", Colors.INFO)

        # 计算价格变化
        price_diff = df['close'].diff()

        # 分离涨跌
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        # 计算平均涨跌
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        print(f"   - 最后的 avg_gain: {avg_gain.iloc[-1]}")
        print(f"   - 最后的 avg_loss: {avg_loss.iloc[-1]}")
        print(f"   - 最后的 RS: {rs.iloc[-1]}")
        print(f"   - 计算的 RSI: {rsi.iloc[-1]}")

        # 检查 NaN 的位置
        nan_count = rsi.isna().sum()
        print(f"   - RSI 中 NaN 的数量: {nan_count}")

        if nan_count > 0:
            first_valid_idx = rsi.first_valid_index()
            print(f"   - 第一个有效 RSI 的索引: {first_valid_idx}")

    # 3. 检查现有的 RSI 列
    if 'RSI' in df.columns:
        print_colored("\n3. 现有 RSI 列检查:", Colors.INFO)
        print(f"   - RSI 列类型: {df['RSI'].dtype}")
        print(f"   - RSI 非空值数量: {df['RSI'].notna().sum()}")
        print(f"   - RSI 最后5个值: {df['RSI'].tail().tolist()}")
        print(f"   - RSI 最后一个值: {df['RSI'].iloc[-1]}")
        print(f"   - 是否为 NaN: {pd.isna(df['RSI'].iloc[-1])}")

    print_colored("\n=== RSI 诊断结束 ===", Colors.CYAN)

    return df


def test_rsi_calculation():
    """测试 RSI 计算"""
    try:
        from indicators_module import calculate_optimized_indicators

        # 创建测试数据
        print_colored("\n创建测试数据...", Colors.INFO)
        test_data = {
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000
        }

        df = pd.DataFrame(test_data)
        print(f"测试数据形状: {df.shape}")

        # 计算指标
        print_colored("\n计算指标...", Colors.INFO)
        df_with_indicators = calculate_optimized_indicators(df)

        # 诊断
        diagnose_rsi_calculation(df_with_indicators)

    except Exception as e:
        print_colored(f"\n❌ 测试失败: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rsi_calculation()
