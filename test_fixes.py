# test_fixes.py
# 测试修复是否成功的脚本

import pandas as pd
import numpy as np

# 测试导入
try:
    from logger_utils import Colors, print_colored
    print("✅ logger_utils 导入成功")

    # 测试新添加的颜色常量
    test_attrs = ['OVERBOUGHT', 'OVERSOLD', 'TREND_UP', 'TREND_DOWN', 'TREND_NEUTRAL']
    for attr in test_attrs:
        if hasattr(Colors, attr):
            print(f"  ✅ Colors.{attr} 存在")
        else:
            print(f"  ❌ Colors.{attr} 缺失")
except Exception as e:
    print(f"❌ logger_utils 导入失败: {e}")

# 测试 indicators_module
try:
    from indicators_module import calculate_optimized_indicators
    print("\n✅ indicators_module 导入成功")

    # 测试参数验证
    print("\n测试参数验证:")

    # 测试 None
    result = calculate_optimized_indicators(None)
    print("  ✅ None 参数处理正常")

    # 测试字符串
    result = calculate_optimized_indicators("invalid")
    print("  ✅ 字符串参数处理正常")

    # 测试空 DataFrame
    result = calculate_optimized_indicators(pd.DataFrame())
    print("  ✅ 空 DataFrame 处理正常")

    # 测试正常 DataFrame
    test_df = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100)
    })
    result = calculate_optimized_indicators(test_df)
    if not result.empty:
        print("  ✅ 正常 DataFrame 处理成功")

except Exception as e:
    print(f"❌ indicators_module 测试失败: {e}")

print("\n测试完成！")
