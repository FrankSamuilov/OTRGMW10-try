# 修复交易机器人错误的补丁文件
# fix_errors.py

import os
import re


def fix_colors_class():
    """修复 Colors 类缺少的属性"""
    logger_utils_content = '''"""
日志工具模块：提供彩色日志输出和格式化功能
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime


# ANSI颜色代码
class Colors:
    # 基础颜色
    BLACK = '\\033[30m'
    RED = '\\033[31m'
    GREEN = '\\033[32m'
    YELLOW = '\\033[33m'
    BLUE = '\\033[34m'
    MAGENTA = '\\033[35m'
    CYAN = '\\033[36m'
    WHITE = '\\033[37m'
    GRAY = '\\033[90m'

    # 亮色版本
    BRIGHT_BLACK = '\\033[90m'
    BRIGHT_RED = '\\033[91m'
    BRIGHT_GREEN = '\\033[92m'
    BRIGHT_YELLOW = '\\033[93m'
    BRIGHT_BLUE = '\\033[94m'
    BRIGHT_MAGENTA = '\\033[95m'
    BRIGHT_CYAN = '\\033[96m'
    BRIGHT_WHITE = '\\033[97m'

    # 特殊用途（添加缺失的）
    INFO = '\\033[94m'  # 蓝色
    WARNING = '\\033[93m'  # 黄色
    ERROR = '\\033[91m'  # 红色
    SUCCESS = '\\033[92m'  # 绿色
    DEBUG = '\\033[90m'  # 灰色

    # 添加缺失的颜色常量
    OVERBOUGHT = '\\033[91m'  # 红色 - 超买
    OVERSOLD = '\\033[92m'  # 绿色 - 超卖
    TREND_UP = '\\033[92m'  # 绿色 - 上升趋势
    TREND_DOWN = '\\033[91m'  # 红色 - 下降趋势
    TREND_NEUTRAL = '\\033[90m'  # 灰色 - 中性趋势

    # 样式
    BOLD = '\\033[1m'
    DIM = '\\033[2m'
    ITALIC = '\\033[3m'
    UNDERLINE = '\\033[4m'
    BLINK = '\\033[5m'
    REVERSE = '\\033[7m'
    HIDDEN = '\\033[8m'
    STRIKETHROUGH = '\\033[9m'

    # 重置
    RESET = '\\033[0m'
    RESET_ALL = '\\033[0m'
'''

    # 写入修复后的logger_utils.py的开头部分
    if os.path.exists('logger_utils.py'):
        with open('logger_utils.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找Colors类的结束位置
        class_end = content.find('def format_log')
        if class_end > 0:
            # 保留原文件中Colors类之后的内容
            remaining_content = content[class_end:]

            # 写入新的内容
            with open('logger_utils.py', 'w', encoding='utf-8') as f:
                f.write(logger_utils_content + '\n\n' + remaining_content)
            print("✅ 修复了 logger_utils.py 中的 Colors 类")


def fix_calculate_optimized_indicators():
    """修复 calculate_optimized_indicators 函数调用"""
    if os.path.exists('simple_trading_bot.py'):
        with open('simple_trading_bot.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复函数调用参数
        # 将 calculate_optimized_indicators(df, symbol) 改为 calculate_optimized_indicators(df)
        content = re.sub(
            r'calculate_optimized_indicators\(df,\s*symbol\)',
            'calculate_optimized_indicators(df)',
            content
        )

        # 修复 await 问题
        # 移除错误的 await 调用
        content = re.sub(
            r'await\s+self\.analyze_symbol\(',
            'self.analyze_symbol(',
            content
        )

        with open('simple_trading_bot.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 修复了 simple_trading_bot.py 中的函数调用")


def fix_supertrend_warning():
    """修复 Supertrend 中的类型比较警告"""
    if os.path.exists('indicators_module.py'):
        with open('indicators_module.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            # 查找可能导致警告的行
            if "df[f'{col_prefix}Supertrend_Strength']" in line and "abs(df['close']" in line:
                # 确保使用正确的数据类型比较
                lines[i] = line.replace(
                    "abs(df['close'] - supertrend)",
                    "abs(df['close'].astype(float) - supertrend.astype(float))"
                )
                modified = True

        if modified:
            with open('indicators_module.py', 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("✅ 修复了 indicators_module.py 中的类型比较警告")


def add_error_handling():
    """添加错误处理以防止程序崩溃"""
    error_handling_code = '''
# 在 calculate_optimized_indicators 函数开始处添加
def calculate_optimized_indicators(df: pd.DataFrame, btc_df=None):
    """
    计算优化后的指标
    """
    try:
        # 参数验证
        if df is None:
            print_colored("❌ DataFrame 为 None", Colors.ERROR)
            return pd.DataFrame()

        if isinstance(df, str):
            print_colored(f"❌ 错误：期望 DataFrame，但收到字符串: {df}", Colors.ERROR)
            return pd.DataFrame()

        if df.empty:
            print_colored("❌ DataFrame 为空", Colors.ERROR)
            return pd.DataFrame()

        # ... 原有代码继续 ...
'''

    print("⚠️ 请在 indicators_module.py 的 calculate_optimized_indicators 函数开始处添加以下错误处理代码：")
    print(error_handling_code)


def main():
    """执行所有修复"""
    print("开始修复交易机器人错误...")
    print("=" * 60)

    # 1. 修复 Colors 类
    fix_colors_class()

    # 2. 修复函数调用
    fix_calculate_optimized_indicators()

    # 3. 修复类型警告
    fix_supertrend_warning()

    # 4. 提示添加错误处理
    add_error_handling()

    print("=" * 60)
    print("修复完成！请重新运行您的程序。")
    print("\n注意事项：")
    print("1. 确保所有文件都已保存")
    print("2. 检查 indicators_module.py 中的 calculate_optimized_indicators 函数参数")
    print("3. 如果还有错误，请检查具体的错误信息")


if __name__ == "__main__":
    main()