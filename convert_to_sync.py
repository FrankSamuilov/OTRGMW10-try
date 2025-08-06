# convert_to_sync.py
# 将所有异步代码转换为同步代码的修复脚本

import os
import re
import shutil
from datetime import datetime


def backup_files():
    """备份原始文件"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    files_to_backup = [
        'simple_trading_bot.py',
        'game_theory_module.py',
        'enhanced_game_theory.py',
        'indicators_module.py'
    ]

    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(backup_dir, file))
            print(f"✅ 备份 {file} 到 {backup_dir}")

    return backup_dir


def convert_simple_trading_bot():
    """转换 simple_trading_bot.py 为完全同步版本"""
    if not os.path.exists('simple_trading_bot.py'):
        print("⚠️ 找不到 simple_trading_bot.py")
        return

    with open('simple_trading_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 移除所有 async 关键字
    content = re.sub(r'async\s+def\s+', 'def ', content)

    # 2. 移除所有 await 关键字
    content = re.sub(r'await\s+', '', content)

    # 3. 将异步的 analyze_symbols 改为同步版本
    content = re.sub(
        r'async def analyze_symbols\(self, symbols: List\[str\], account_balance: float\) -> List\[Dict\]:',
        'def analyze_symbols(self, symbols: List[str], account_balance: float) -> List[Dict]:',
        content
    )

    # 4. 修复所有相关的函数调用
    # 移除 asyncio 相关的导入
    content = re.sub(r'import\s+asyncio\s*\n', '', content)
    content = re.sub(r'from\s+asyncio\s+import\s+.*\n', '', content)

    # 5. 修复 analyze_symbol 调用（确保不使用 await）
    content = re.sub(
        r'integrated_decision\s*=\s*await\s+self\.analyze_symbol\(',
        'integrated_decision = self.analyze_symbol(',
        content
    )

    # 6. 修复 perform_technical_analysis 调用
    content = re.sub(
        r'technical_analysis\s*=\s*await\s+self\._perform_technical_analysis\(',
        'technical_analysis = self._perform_technical_analysis(',
        content
    )

    # 7. 确保 analyze_symbol 方法是同步的
    analyze_symbol_pattern = r'def analyze_symbol\(self,[^)]+\)[^:]*:'
    if re.search(analyze_symbol_pattern, content):
        # 找到 analyze_symbol 方法并确保它调用的都是同步方法
        content = re.sub(
            r'game_theory_analysis\s*=\s*await\s+self\.game_theory\.analyze_market\(',
            'game_theory_analysis = self.game_theory.analyze_market(',
            content
        )

    with open('simple_trading_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 转换 simple_trading_bot.py 完成")


def convert_game_theory_module():
    """转换 game_theory_module.py 为同步版本"""
    if not os.path.exists('game_theory_module.py'):
        print("⚠️ 找不到 game_theory_module.py")
        return

    with open('game_theory_module.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 移除所有 async 关键字
    content = re.sub(r'async\s+def\s+', 'def ', content)

    # 2. 移除所有 await 关键字
    content = re.sub(r'await\s+', '', content)

    # 3. 修复特定的方法
    # analyze_market 方法
    content = re.sub(
        r'async def analyze_market\(self, symbol: str\) -> Dict:',
        'def analyze_market(self, symbol: str) -> Dict:',
        content
    )

    # 4. 修复内部调用
    content = re.sub(
        r'order_book\s*=\s*await\s+self\.get_order_book_async\(',
        'order_book = self.get_order_book(',
        content
    )

    content = re.sub(
        r'long_short_data\s*=\s*await\s+self\.get_long_short_ratio\(',
        'long_short_data = self.get_long_short_ratio(',
        content
    )

    # 5. 将异步方法改为同步
    content = re.sub(
        r'async def get_order_book_async\(',
        'def get_order_book(',
        content
    )

    content = re.sub(
        r'async def get_long_short_ratio\(',
        'def get_long_short_ratio(',
        content
    )

    with open('game_theory_module.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 转换 game_theory_module.py 完成")


def fix_indicators_module():
    """修复 indicators_module.py 中的 RuntimeWarning"""
    if not os.path.exists('indicators_module.py'):
        print("⚠️ 找不到 indicators_module.py")
        return

    with open('indicators_module.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复 Supertrend_Strength 计算中的类型问题
    # 找到问题行并修复
    old_line = "df[f'{col_prefix}Supertrend_Strength'] = abs(df['close'] - supertrend) / df['ATR']"
    new_line = "df[f'{col_prefix}Supertrend_Strength'] = abs(df['close'].astype(float) - supertrend.astype(float)) / df['ATR'].astype(float)"

    content = content.replace(old_line, new_line)

    # 额外的安全检查：确保在比较前转换类型
    # 在计算 supertrend 的循环中添加类型转换
    content = re.sub(
        r"if close\.iloc\[i\] > upperband\.iloc\[i - 1\]:",
        "if float(close.iloc[i]) > float(upperband.iloc[i - 1]):",
        content
    )

    content = re.sub(
        r"elif close\.iloc\[i\] < lowerband\.iloc\[i - 1\]:",
        "elif float(close.iloc[i]) < float(lowerband.iloc[i - 1]):",
        content
    )

    with open('indicators_module.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 修复 indicators_module.py 的 RuntimeWarning")


def create_sync_wrapper():
    """创建一个同步包装器文件，确保所有调用都是同步的"""
    wrapper_content = '''# sync_wrapper.py
# 同步包装器 - 确保所有函数调用都是同步的

import time
from typing import Dict, List, Any
from logger_utils import Colors, print_colored


class SyncTradingBot:
    """完全同步的交易机器人包装器"""

    def __init__(self, original_bot):
        self.bot = original_bot
        self.logger = original_bot.logger

    def analyze_symbol(self, symbol: str, account_balance: float) -> Dict[str, Any]:
        """同步分析单个交易对"""
        try:
            # 直接调用原始方法，不使用 await
            result = self.bot.analyze_symbol(symbol, account_balance)
            return result
        except Exception as e:
            self.logger.error(f"分析 {symbol} 失败: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'error': str(e)
            }

    def analyze_symbols(self, symbols: List[str], account_balance: float) -> List[Dict]:
        """同步分析多个交易对"""
        trading_opportunities = []

        for i, symbol in enumerate(symbols, 1):
            try:
                print_colored(f"\\n{'=' * 60}", Colors.BLUE)
                print_colored(f"📊 综合分析 {symbol} ({i}/{len(symbols)})", Colors.CYAN + Colors.BOLD)
                print_colored(f"{'=' * 60}", Colors.BLUE)

                # 同步调用分析
                result = self.analyze_symbol(symbol, account_balance)

                if result.get('action') != 'HOLD' and 'error' not in result:
                    if result.get('trade_params'):
                        trading_opportunities.append(result)

            except Exception as e:
                self.logger.error(f"分析 {symbol} 失败: {e}")
                print_colored(f"\\n❌ 分析失败: {str(e)}", Colors.ERROR)

        return trading_opportunities

    def run_trading_cycle(self):
        """运行交易循环 - 完全同步版本"""
        try:
            # 获取账户余额
            account_balance = self.bot.get_account_balance()

            # 分析交易对
            opportunities = self.analyze_symbols(self.bot.symbols_to_scan, account_balance)

            # 处理交易机会
            if opportunities:
                print_colored(f"\\n找到 {len(opportunities)} 个交易机会", Colors.GREEN)
                # 这里可以添加执行交易的逻辑
            else:
                print_colored("\\n未找到合适的交易机会", Colors.YELLOW)

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}")
            print_colored(f"\\n❌ 交易循环错误: {str(e)}", Colors.ERROR)


def ensure_sync_execution(bot_instance):
    """确保机器人以同步方式执行"""
    return SyncTradingBot(bot_instance)
'''

    with open('sync_wrapper.py', 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    print("✅ 创建同步包装器 sync_wrapper.py")


def create_new_main():
    """创建新的主程序入口，使用同步方式运行"""
    main_content = '''# main_sync.py
# 同步版本的主程序入口

import sys
import time
from datetime import datetime
from logger_utils import Colors, print_colored

# 导入必要的模块
from simple_trading_bot import SimpleTradingBot
from sync_wrapper import ensure_sync_execution


def main():
    """主函数 - 完全同步版本"""
    print_colored(f"{'=' * 60}", Colors.BLUE)
    print_colored(f"🚀 启动交易机器人 (同步版本)", Colors.BLUE + Colors.BOLD)
    print_colored(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.INFO)
    print_colored(f"{'=' * 60}", Colors.BLUE)

    try:
        # 创建机器人实例
        print_colored("初始化交易机器人...", Colors.INFO)
        bot = SimpleTradingBot()

        # 确保同步执行
        sync_bot = ensure_sync_execution(bot)

        print_colored("✅ 机器人初始化成功", Colors.GREEN)

        # 运行主循环
        print_colored("\\n📊 开始运行交易循环...", Colors.INFO)

        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                print_colored(f"\\n🔄 第 {cycle_count} 轮扫描开始", Colors.CYAN)

                # 运行一个交易循环
                sync_bot.run_trading_cycle()

                # 等待下一轮
                wait_time = 300  # 5分钟
                print_colored(f"\\n⏳ 等待 {wait_time} 秒后进行下一轮扫描...", Colors.INFO)

                # 显示倒计时
                for remaining in range(wait_time, 0, -30):
                    print(f"\\r剩余时间: {remaining} 秒", end='', flush=True)
                    time.sleep(min(30, remaining))

            except KeyboardInterrupt:
                print_colored("\\n\\n⚠️ 收到中断信号，正在安全退出...", Colors.WARNING)
                break
            except Exception as e:
                print_colored(f"\\n❌ 交易循环错误: {e}", Colors.ERROR)
                print_colored("⏳ 30秒后重试...", Colors.WARNING)
                time.sleep(30)

    except Exception as e:
        print_colored(f"\\n❌ 严重错误: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()
        return 1

    print_colored("\\n👋 交易机器人已停止", Colors.INFO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    with open('main_sync.py', 'w', encoding='utf-8') as f:
        f.write(main_content)
    print("✅ 创建新的同步主程序 main_sync.py")


def main():
    """执行所有转换"""
    print("开始将异步代码转换为同步代码...")
    print("=" * 60)

    # 1. 备份原始文件
    backup_dir = backup_files()
    print(f"\n原始文件已备份到: {backup_dir}")

    # 2. 转换各个文件
    print("\n开始转换文件...")
    convert_simple_trading_bot()
    convert_game_theory_module()
    fix_indicators_module()

    # 3. 创建辅助文件
    print("\n创建辅助文件...")
    create_sync_wrapper()
    create_new_main()

    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("\n使用说明：")
    print("1. 运行新的同步版本程序：")
    print("   python main_sync.py")
    print("\n2. 如果需要恢复原始文件，可以从备份目录恢复：")
    print(f"   {backup_dir}")
    print("\n3. 确保已经修复了 logger_utils.py 中的 Colors 类问题")
    print("\n注意：新版本完全移除了所有异步代码，应该不会再出现 await 相关错误")


if __name__ == "__main__":
    main()