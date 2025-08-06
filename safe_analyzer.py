# safe_analyzer.py
# 安全的分析函数，避免 await 错误

from typing import Dict, List, Any
from logger_utils import Colors, print_colored
import traceback


class SafeAnalyzer:
    """安全的分析器，确保所有调用都是同步的"""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.logger = bot_instance.logger if hasattr(bot_instance, 'logger') else None

    def analyze_symbol_safe(self, symbol: str, account_balance: float) -> Dict[str, Any]:
        """安全地分析交易对 - 完全同步"""
        try:
            # 确保使用同步方式调用
            if hasattr(self.bot, 'game_theory') and hasattr(self.bot.game_theory, 'analyze_market'):
                # 调用游戏理论分析（同步）
                game_theory_analysis = self.bot.game_theory.analyze_market(symbol)
            else:
                game_theory_analysis = {'action': 'HOLD', 'confidence': 0}

            # 调用技术分析（同步）
            if hasattr(self.bot, '_perform_technical_analysis'):
                technical_analysis = self.bot._perform_technical_analysis(symbol)
            else:
                technical_analysis = {'signal_strength': 0, 'trend': 'NEUTRAL'}

            # 整合分析结果
            if hasattr(self.bot, '_integrate_analyses_trend_first'):
                result = self.bot._integrate_analyses_trend_first(
                    game_theory_analysis,
                    technical_analysis,
                    symbol
                )
            else:
                result = {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'reason': '分析方法不可用'
                }

            return result

        except Exception as e:
            error_msg = f"分析 {symbol} 时出错: {str(e)}"
            print_colored(f"❌ {error_msg}", Colors.ERROR)

            # 打印详细的错误信息帮助调试
            if "await" in str(e):
                print_colored("检测到 await 错误，请确保所有方法都是同步的", Colors.WARNING)

            traceback.print_exc()

            return {
                'symbol': symbol,
                'action': 'HOLD',
                'error': error_msg
            }

    def analyze_symbols_safe(self, symbols: List[str], account_balance: float) -> List[Dict]:
        """安全地分析多个交易对"""
        trading_opportunities = []

        for i, symbol in enumerate(symbols, 1):
            try:
                print_colored(f"\n{'=' * 60}", Colors.BLUE)
                print_colored(f"📊 综合分析 {symbol} ({i}/{len(symbols)})", Colors.CYAN + Colors.BOLD)
                print_colored(f"{'=' * 60}", Colors.BLUE)

                # 使用安全的分析方法
                result = self.analyze_symbol_safe(symbol, account_balance)

                if result.get('action') != 'HOLD' and 'error' not in result:
                    if result.get('trade_params'):
                        trading_opportunities.append(result)

            except Exception as e:
                print_colored(f"\n❌ 分析 {symbol} 失败: {str(e)}", Colors.ERROR)
                continue

        return trading_opportunities


def patch_bot_instance(bot):
    """为现有的机器人实例打补丁"""
    analyzer = SafeAnalyzer(bot)

    # 替换原有方法
    bot.analyze_symbol = analyzer.analyze_symbol_safe
    bot.analyze_symbols = analyzer.analyze_symbols_safe

    return bot
