# comprehensive_fix.py
# 综合修复方案 - 修复所有交易机器人错误

import os
import re


def fix_all_errors():
    """执行所有必要的修复"""

    # 1. 修复 logger_utils.py - 添加缺失的颜色常量
    fix_logger_utils()

    # 2. 修复 indicators_module.py - 修复参数和类型问题
    fix_indicators_module()

    # 3. 修复 simple_trading_bot.py - 修复函数调用和 await 问题
    fix_simple_trading_bot()

    print("\n✅ 所有修复已完成！")


def fix_logger_utils():
    """修复 logger_utils.py"""
    if not os.path.exists('logger_utils.py'):
        print("⚠️ 找不到 logger_utils.py")
        return

    with open('logger_utils.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 在 Colors 类中添加缺失的属性
    if 'OVERBOUGHT' not in content:
        # 找到 Colors 类的结束位置（在 RESET 之后）
        insert_pos = content.find('RESET_ALL = \'\\033[0m\'')
        if insert_pos > 0:
            # 找到该行的结束
            insert_pos = content.find('\n', insert_pos) + 1

            # 要插入的新属性
            new_attributes = '''
    # 添加缺失的交易相关颜色
    OVERBOUGHT = '\\033[91m'  # 红色 - 超买
    OVERSOLD = '\\033[92m'  # 绿色 - 超卖
    TREND_UP = '\\033[92m'  # 绿色 - 上升趋势
    TREND_DOWN = '\\033[91m'  # 红色 - 下降趋势
    TREND_NEUTRAL = '\\033[90m'  # 灰色 - 中性趋势
'''

            # 插入新属性
            content = content[:insert_pos] + new_attributes + content[insert_pos:]

            with open('logger_utils.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 修复了 logger_utils.py - 添加了缺失的颜色常量")


def fix_indicators_module():
    """修复 indicators_module.py"""
    if not os.path.exists('indicators_module.py'):
        print("⚠️ 找不到 indicators_module.py")
        return

    with open('indicators_module.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 修复 calculate_optimized_indicators 函数定义
    # 确保函数签名只有两个参数
    content = re.sub(
        r'def calculate_optimized_indicators\([^)]+\):',
        'def calculate_optimized_indicators(df: pd.DataFrame, btc_df=None):',
        content
    )

    # 2. 在函数开始添加参数验证
    func_start = content.find('def calculate_optimized_indicators')
    if func_start > 0:
        # 找到函数体的开始（第一个 try: 或者函数文档字符串之后）
        func_body_start = content.find('try:', func_start)
        if func_body_start < 0:
            # 如果没有 try，找文档字符串结束位置
            doc_end = content.find('"""', content.find('"""', func_start) + 3) + 3
            func_body_start = content.find('\n', doc_end) + 1

        # 添加参数验证代码
        validation_code = '''    # 参数验证
    if df is None:
        print_colored("❌ calculate_optimized_indicators: DataFrame 为 None", Colors.ERROR)
        return pd.DataFrame()

    if isinstance(df, str):
        print_colored(f"❌ calculate_optimized_indicators: 错误的参数类型，期望 DataFrame，收到 str", Colors.ERROR)
        return pd.DataFrame()

    if hasattr(df, 'empty') and df.empty:
        print_colored("❌ calculate_optimized_indicators: DataFrame 为空", Colors.ERROR)
        return pd.DataFrame()

    '''

        # 只在还没有这些验证的情况下添加
        if 'isinstance(df, str)' not in content:
            content = content[:func_body_start] + validation_code + content[func_body_start:]

    # 3. 修复 Supertrend_Strength 计算中的类型比较问题
    content = re.sub(
        r"df\[f'{col_prefix}Supertrend_Strength'\] = abs\(df\['close'\] - supertrend\) / df\['ATR'\]",
        "df[f'{col_prefix}Supertrend_Strength'] = abs(df['close'].astype(float) - supertrend.astype(float)) / df['ATR'].astype(float)",
        content
    )

    with open('indicators_module.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 修复了 indicators_module.py - 添加了参数验证和类型转换")


def fix_simple_trading_bot():
    """修复 simple_trading_bot.py"""
    if not os.path.exists('simple_trading_bot.py'):
        print("⚠️ 找不到 simple_trading_bot.py")
        return

    with open('simple_trading_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 修复 calculate_optimized_indicators 调用
    content = re.sub(
        r'calculate_optimized_indicators\(df,\s*symbol\)',
        'calculate_optimized_indicators(df)',
        content
    )

    # 2. 修复错误的 await 使用
    # 查找所有 await self.analyze_symbol 并移除 await
    content = re.sub(
        r'await\s+self\.analyze_symbol\(',
        'self.analyze_symbol(',
        content
    )

    # 3. 修复 analyze_symbols 方法
    # 确保不是异步方法
    content = re.sub(
        r'async\s+def\s+analyze_symbols\(',
        'def analyze_symbols(',
        content
    )

    # 4. 修复 Colors.DEBUG 使用
    content = content.replace('Colors.DEBUG', 'Colors.INFO')

    with open('simple_trading_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 修复了 simple_trading_bot.py - 修正了函数调用和 await 问题")


def create_test_script():
    """创建测试脚本验证修复"""
    test_script = '''# test_fixes.py
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
    print("\\n✅ indicators_module 导入成功")

    # 测试参数验证
    print("\\n测试参数验证:")

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

print("\\n测试完成！")
'''

    with open('test_fixes.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    print("\n✅ 创建了测试脚本 test_fixes.py")


if __name__ == "__main__":
    print("开始执行综合修复...")
    print("=" * 60)

    fix_all_errors()
    create_test_script()

    print("\n" + "=" * 60)
    print("修复完成！")
    print("\n下一步操作：")
    print("1. 运行 python test_fixes.py 验证修复")
    print("2. 如果测试通过，重新运行您的交易机器人")
    print("3. 如果还有错误，请提供新的错误信息")