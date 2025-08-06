# test_indicators_display.py
# 测试指标显示是否正确

def test_display():
    """测试指标显示"""
    import pandas as pd

    # 创建测试数据
    test_data = {
        'RSI': [30, 40, 50, 60, 70, 28.36],  # 最后一个值是您日志中的实际值
        'volume': [1000, 1200, 1500, 2000, 2500, 3000]
    }
    df = pd.DataFrame(test_data)

    # 测试 RSI 获取
    print("测试 RSI 获取:")

    # 错误的方式（可能导致始终为 50）
    rsi_wrong = 50  # 硬编码
    print(f"  错误方式: RSI = {rsi_wrong}")

    # 正确的方式
    rsi_correct = df['RSI'].iloc[-1]
    print(f"  正确方式: RSI = {rsi_correct}")

    # 测试成交量比率
    print("\n测试成交量比率:")
    volume_mean = df['volume'].rolling(3).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / volume_mean if volume_mean > 0 else 0
    print(f"  平均成交量: {volume_mean}")
    print(f"  当前成交量: {current_volume}")
    print(f"  成交量比率: {volume_ratio:.2f}x")

    # 模拟技术分析字典
    technical_analysis = {
        'rsi': rsi_correct,  # 使用实际值，不是硬编码的 50
        'volume_ratio': volume_ratio  # 使用计算的值，不是 0
    }

    print("\n技术分析结果:")
    print(f"  • RSI: {technical_analysis['rsi']:.1f}")
    print(f"  • 成交量比率: {technical_analysis['volume_ratio']:.1f}x")


if __name__ == "__main__":
    test_display()
