# diagnose_technical_analysis.py
import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored

def test_technical_analysis():
    """测试技术分析流程"""

    # 创建测试数据
    test_data = {
        'open': [100, 101, 102, 103, 104] * 20,
        'high': [101, 102, 103, 104, 105] * 20,
        'low': [99, 100, 101, 102, 103] * 20,
        'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
        'volume': [1000, 1200, 800, 1500, 2000] * 20
    }
    df = pd.DataFrame(test_data)

    print_colored("测试数据创建完成", Colors.INFO)
    print(f"数据形状: {df.shape}")

    # 测试指标计算
    try:
        from indicators_module import calculate_optimized_indicators
        df_with_indicators = calculate_optimized_indicators(df)

        # 检查 RSI
        if 'RSI' in df_with_indicators.columns:
            rsi_values = df_with_indicators['RSI'].dropna()
            print_colored(f"\nRSI 分析:", Colors.CYAN)
            print(f"  - RSI 列存在: ✓")
            print(f"  - 非空 RSI 值数量: {len(rsi_values)}")
            print(f"  - RSI 范围: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
            print(f"  - 最后的 RSI: {df_with_indicators['RSI'].iloc[-1]:.2f}")
        else:
            print_colored("❌ RSI 列不存在！", Colors.ERROR)

    except Exception as e:
        print_colored(f"❌ 指标计算失败: {e}", Colors.ERROR)
        return

    # 测试技术分析函数
    try:
        from simple_trading_bot import SimpleTradingBot
        bot = SimpleTradingBot()

        # 模拟技术分析
        print_colored("\n测试技术分析函数...", Colors.INFO)

        # 创建一个包含指标的测试 DataFrame
        test_df = df_with_indicators.copy()

        # 手动设置一些值来测试
        if 'RSI' in test_df.columns:
            test_df['RSI'].iloc[-1] = 75.5  # 设置一个特定的 RSI 值

        # 测试获取 RSI
        print_colored("\n测试 RSI 获取:", Colors.CYAN)

        # 方法1：直接访问
        if 'RSI' in test_df.columns:
            rsi1 = test_df['RSI'].iloc[-1]
            print(f"  方法1 - 直接访问: {rsi1}")

        # 方法2：安全访问
        rsi2 = float(test_df['RSI'].iloc[-1]) if 'RSI' in test_df.columns else 50
        print(f"  方法2 - 安全访问: {rsi2}")

        # 方法3：检查 NaN
        if 'RSI' in test_df.columns and not test_df['RSI'].empty:
            rsi_value = test_df['RSI'].iloc[-1]
            rsi3 = float(rsi_value) if not pd.isna(rsi_value) else 50
        else:
            rsi3 = 50
        print(f"  方法3 - NaN 检查: {rsi3}")

        # 测试成交量分析
        print_colored("\n测试成交量分析:", Colors.CYAN)
        volume_mean = test_df['volume'].rolling(20).mean().iloc[-1]
        volume_current = test_df['volume'].iloc[-1]
        volume_ratio = volume_current / volume_mean if volume_mean > 0 else 0
        print(f"  - 当前成交量: {volume_current}")
        print(f"  - 平均成交量: {volume_mean:.2f}")
        print(f"  - 成交量比率: {volume_ratio:.2f}x")

    except Exception as e:
        print_colored(f"❌ 技术分析测试失败: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_technical_analysis()
