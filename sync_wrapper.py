# sync_wrapper.py
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
                print_colored(f"\n{'=' * 60}", Colors.BLUE)
                print_colored(f"📊 综合分析 {symbol} ({i}/{len(symbols)})", Colors.CYAN + Colors.BOLD)
                print_colored(f"{'=' * 60}", Colors.BLUE)

                # 同步调用分析
                result = self.analyze_symbol(symbol, account_balance)

                if result.get('action') != 'HOLD' and 'error' not in result:
                    if result.get('trade_params'):
                        trading_opportunities.append(result)

            except Exception as e:
                self.logger.error(f"分析 {symbol} 失败: {e}")
                print_colored(f"\n❌ 分析失败: {str(e)}", Colors.ERROR)

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
                print_colored(f"\n找到 {len(opportunities)} 个交易机会", Colors.GREEN)
                # 这里可以添加执行交易的逻辑
            else:
                print_colored("\n未找到合适的交易机会", Colors.YELLOW)

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}")
            print_colored(f"\n❌ 交易循环错误: {str(e)}", Colors.ERROR)


def ensure_sync_execution(bot_instance):
    """确保机器人以同步方式执行"""
    return SyncTradingBot(bot_instance)
