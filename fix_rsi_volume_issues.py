# fix_rsi_volume_issues.py
# 修复 RSI 始终为 50 和成交量分析失败的问题

import os
import re


def diagnose_issues():
    """诊断 RSI 和成交量问题"""
    print("诊断 RSI 和成交量问题...")
    print("=" * 60)

    # 创建诊断脚本
    diagnostic_script = '''# diagnose_technical_analysis.py
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
            print_colored(f"\\nRSI 分析:", Colors.CYAN)
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
        print_colored("\\n测试技术分析函数...", Colors.INFO)

        # 创建一个包含指标的测试 DataFrame
        test_df = df_with_indicators.copy()

        # 手动设置一些值来测试
        if 'RSI' in test_df.columns:
            test_df['RSI'].iloc[-1] = 75.5  # 设置一个特定的 RSI 值

        # 测试获取 RSI
        print_colored("\\n测试 RSI 获取:", Colors.CYAN)

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
        print_colored("\\n测试成交量分析:", Colors.CYAN)
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
'''

    with open('diagnose_technical_analysis.py', 'w', encoding='utf-8') as f:
        f.write(diagnostic_script)
    print("✅ 创建了诊断脚本 diagnose_technical_analysis.py")


def fix_rsi_transmission():
    """修复 RSI 值传递问题"""

    if not os.path.exists('simple_trading_bot.py'):
        print("❌ 找不到 simple_trading_bot.py")
        return

    with open('simple_trading_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    print("\n修复 RSI 传递问题...")

    # 查找 _perform_technical_analysis 函数
    tech_analysis_start = content.find("def _perform_technical_analysis")
    if tech_analysis_start < 0:
        print("❌ 找不到 _perform_technical_analysis 函数")
        return

    # 修复1：确保 RSI 从计算结果中正确获取
    # 查找 technical_analysis 字典构建部分
    pattern = r"technical_analysis\s*=\s*{[^}]+}"

    # 在函数内查找
    func_end = content.find("\n    def ", tech_analysis_start + 1)
    if func_end < 0:
        func_end = len(content)

    func_content = content[tech_analysis_start:func_end]

    # 检查是否正确传递了 RSI
    if "'rsi':" in func_content:
        print("✅ 找到 RSI 字段")

        # 确保使用的是计算出的 rsi 变量，而不是默认值
        # 查找 rsi 变量的获取
        if "rsi = 50" in func_content and "if pd.isna(rsi):" not in func_content:
            print("⚠️ 发现可能的问题：RSI 被硬编码为 50")

    # 创建修复补丁
    fix_patch = '''
# RSI 传递修复补丁
def fix_rsi_in_technical_analysis(df):
    """确保 RSI 正确传递"""
    # 获取实际的 RSI 值
    actual_rsi = 50  # 默认值

    if 'RSI' in df.columns and not df['RSI'].empty:
        rsi_series = df['RSI'].dropna()
        if len(rsi_series) > 0:
            actual_rsi = float(rsi_series.iloc[-1])
            print(f"[DEBUG] 获取到实际 RSI: {actual_rsi:.2f}")
        else:
            print("[DEBUG] RSI 列全是 NaN")
    else:
        print("[DEBUG] RSI 列不存在")

    return actual_rsi
'''

    # 写入补丁文件
    with open('rsi_fix_patch.py', 'w', encoding='utf-8') as f:
        f.write(fix_patch)
    print("✅ 创建了 RSI 修复补丁")


def fix_volume_analysis():
    """修复成交量分析"""

    volume_fix = '''# volume_analysis_fix.py
# 修复成交量分析

def calculate_volume_surge(df, lookback=20):
    """计算成交量激增"""
    try:
        if 'volume' not in df.columns or len(df) < lookback:
            return 0.0, False

        # 计算平均成交量
        volume_mean = df['volume'].rolling(lookback).mean()
        current_volume = df['volume'].iloc[-1]
        mean_volume = volume_mean.iloc[-1]

        # 计算比率
        if mean_volume > 0:
            volume_ratio = current_volume / mean_volume
        else:
            volume_ratio = 1.0

        # 判断是否激增（超过平均值的1.5倍）
        is_surge = volume_ratio > 1.5

        return volume_ratio, is_surge

    except Exception as e:
        print(f"成交量分析错误: {e}")
        return 0.0, False


def get_volume_analysis_text(volume_ratio, is_surge):
    """获取成交量分析文本"""
    if is_surge:
        return f"⚡ 成交量激增 (比率: {volume_ratio:.1f}x)"
    else:
        return f"成交量正常 (比率: {volume_ratio:.1f}x)"
'''

    with open('volume_analysis_fix.py', 'w', encoding='utf-8') as f:
        f.write(volume_fix)
    print("✅ 创建了成交量分析修复")


def create_comprehensive_fix():
    """创建综合修复方案"""

    comprehensive_fix = '''# comprehensive_technical_fix.py
# 综合修复技术分析问题

import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored


def get_technical_indicator_safely(df, indicator_name, default_value=50):
    """安全获取技术指标值"""
    try:
        if indicator_name not in df.columns:
            print_colored(f"⚠️ {indicator_name} 列不存在，使用默认值 {default_value}", Colors.WARNING)
            return default_value

        # 获取最后一个有效值
        series = df[indicator_name].dropna()
        if len(series) == 0:
            print_colored(f"⚠️ {indicator_name} 没有有效值，使用默认值 {default_value}", Colors.WARNING)
            return default_value

        value = float(series.iloc[-1])
        print_colored(f"✅ {indicator_name}: {value:.2f}", Colors.SUCCESS)
        return value

    except Exception as e:
        print_colored(f"❌ 获取 {indicator_name} 失败: {e}，使用默认值 {default_value}", Colors.ERROR)
        return default_value


def analyze_technical_indicators(df):
    """分析技术指标并返回结果"""
    result = {
        'rsi': get_technical_indicator_safely(df, 'RSI', 50),
        'macd': get_technical_indicator_safely(df, 'MACD', 0),
        'macd_signal': get_technical_indicator_safely(df, 'MACD_signal', 0),
        'bb_position': 50,
        'volume_ratio': 1.0,
        'volume_surge': False
    }

    # 计算布林带位置
    if all(col in df.columns for col in ['close', 'BB_Upper', 'BB_Lower']):
        try:
            close = df['close'].iloc[-1]
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            if upper > lower:
                result['bb_position'] = ((close - lower) / (upper - lower)) * 100
        except:
            pass

    # 计算成交量比率
    if 'volume' in df.columns and len(df) >= 20:
        try:
            volume_mean = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            if volume_mean > 0:
                result['volume_ratio'] = current_volume / volume_mean
                result['volume_surge'] = result['volume_ratio'] > 1.5
        except:
            pass

    return result


# 在 _perform_technical_analysis 中使用：
# indicators = analyze_technical_indicators(df)
# rsi = indicators['rsi']
# volume_ratio = indicators['volume_ratio']
'''

    with open('comprehensive_technical_fix.py', 'w', encoding='utf-8') as f:
        f.write(comprehensive_fix)
    print("✅ 创建了综合修复方案")


def main():
    print("开始修复 RSI 和成交量问题...")
    print("=" * 60)

    # 1. 诊断问题
    diagnose_issues()

    # 2. 修复 RSI 传递
    fix_rsi_transmission()

    # 3. 修复成交量分析
    fix_volume_analysis()

    # 4. 创建综合修复
    create_comprehensive_fix()

    print("\n" + "=" * 60)
    print("修复步骤：")
    print("\n1. 运行诊断脚本了解问题：")
    print("   python diagnose_technical_analysis.py")
    print("\n2. 在 simple_trading_bot.py 的 _perform_technical_analysis 函数中：")
    print("   - 导入: from comprehensive_technical_fix import analyze_technical_indicators")
    print("   - 使用: indicators = analyze_technical_indicators(df)")
    print("   - 获取: rsi = indicators['rsi']")
    print("\n3. 检查是否有其他地方覆盖了 RSI 值")
    print("\n4. 确保技术分析结果正确传递到显示函数")


if __name__ == "__main__":
    main()