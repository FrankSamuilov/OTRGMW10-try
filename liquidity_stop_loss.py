import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from logger_utils import Colors, print_colored


class LiquidityAwareStopLoss:
    """
    流动性感知的智能止损系统
    """

    def __init__(self, liquidity_hunter=None, logger=None):
        self.liquidity_hunter = liquidity_hunter
        self.logger = logger

        # 移动止损参数
        self.trailing_config = {
            'activation_threshold': 0.618,  # 0.618%激活移动止损
            'base_trailing_distance': 0.8,  # 基础跟踪距离80%
            'min_trailing_distance': 0.5,  # 最小跟踪距离50%
            'max_trailing_distance': 0.95,  # 最大跟踪距离95%
        }

        # 流动性调整参数
        self.liquidity_config = {
            'liquidity_check_interval': 60,  # 每60秒检查一次流动性
            'high_liquidity_threshold': 1.5,  # 高流动性阈值（相对于平均）
            'adjustment_factor': 0.1,  # 每级流动性调整10%
            'max_adjustment': 0.3,  # 最大调整30%
        }

        # 缓存
        self.liquidity_cache = {}
        self.last_liquidity_check = {}

        print_colored("✅ 流动性感知止损系统初始化完成", Colors.GREEN)

    def update_position_stop_loss(self, position: Dict, current_price: float,
                                  market_data: Dict) -> Dict:
        """
        更新持仓的止损位置
        """
        symbol = position['symbol']
        position_side = position.get('position_side', 'LONG')
        entry_price = position['entry_price']
        current_stop = position.get('current_stop_level', 0)

        # 计算当前盈利百分比
        if position_side == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # 检查是否激活移动止损
        if profit_pct >= self.trailing_config['activation_threshold']:

            # 计算基础移动止损位置
            base_stop_level = self.calculate_base_trailing_stop(
                position_side, entry_price, current_price, profit_pct
            )

            # 检查反向流动性
            liquidity_adjustment = self.check_reverse_liquidity(
                symbol, position_side, current_price, profit_pct
            )

            # 应用流动性调整
            adjusted_stop_level = self.apply_liquidity_adjustment(
                base_stop_level, liquidity_adjustment, position_side,
                current_price, profit_pct
            )

            # 确保止损只向有利方向移动
            if self.is_better_stop_level(adjusted_stop_level, current_stop, position_side):
                # 打印更新信息
                self.print_stop_update(
                    symbol, position_side, current_stop, adjusted_stop_level,
                    profit_pct, liquidity_adjustment
                )

                return {
                    'should_update': True,
                    'new_stop_level': adjusted_stop_level,
                    'trailing_active': True,
                    'profit_pct': profit_pct,
                    'liquidity_adjusted': liquidity_adjustment['adjusted'],
                    'adjustment_reason': liquidity_adjustment['reason']
                }

        return {
            'should_update': False,
            'trailing_active': position.get('trailing_active', False),
            'profit_pct': profit_pct
        }

    def calculate_base_trailing_stop(self, position_side: str, entry_price: float,
                                     current_price: float, profit_pct: float) -> float:
        """
        计算基础移动止损位置
        """
        # 根据盈利程度调整跟踪距离
        if profit_pct < 1.0:
            # 0.618% - 1%: 使用基础跟踪距离
            trailing_distance = self.trailing_config['base_trailing_distance']
        elif profit_pct < 2.0:
            # 1% - 2%: 收紧到70%
            trailing_distance = 0.7
        elif profit_pct < 3.0:
            # 2% - 3%: 收紧到60%
            trailing_distance = 0.6
        else:
            # 3%以上: 最紧50%
            trailing_distance = 0.5

        # 计算止损位置
        profit_to_keep = profit_pct * trailing_distance / 100

        if position_side == 'LONG':
            stop_level = entry_price * (1 + profit_to_keep)
        else:
            stop_level = entry_price * (1 - profit_to_keep)

        return stop_level

    def check_reverse_liquidity(self, symbol: str, position_side: str,
                                current_price: float, profit_pct: float) -> Dict:
        """
        检查反向流动性情况
        """
        result = {
            'adjusted': False,
            'liquidity_level': 'normal',
            'adjustment_factor': 0,
            'reason': ''
        }

        try:
            # 检查是否需要更新流动性数据
            if self.should_update_liquidity(symbol):
                self.update_liquidity_data(symbol, current_price)

            # 获取缓存的流动性数据
            liquidity_data = self.liquidity_cache.get(symbol, {})
            if not liquidity_data:
                return result

            # 判断反向流动性
            if position_side == 'LONG':
                # 多头持仓，检查下方（空头方向）流动性
                reverse_liquidity = liquidity_data.get('below_liquidity', {})
                reverse_targets = liquidity_data.get('below_targets', [])
            else:
                # 空头持仓，检查上方（多头方向）流动性
                reverse_liquidity = liquidity_data.get('above_liquidity', {})
                reverse_targets = liquidity_data.get('above_targets', [])

            # 计算流动性强度
            avg_liquidity = liquidity_data.get('avg_liquidity', 1)
            current_liquidity = reverse_liquidity.get('total_volume', 0)
            liquidity_ratio = current_liquidity / avg_liquidity if avg_liquidity > 0 else 1

            # 判断是否为高流动性
            if liquidity_ratio > self.liquidity_config['high_liquidity_threshold']:
                result['adjusted'] = True
                result['liquidity_level'] = 'high'

                # 计算调整系数
                # 流动性越高，调整越大
                adjustment = min(
                    (liquidity_ratio - 1) * self.liquidity_config['adjustment_factor'],
                    self.liquidity_config['max_adjustment']
                )
                result['adjustment_factor'] = adjustment

                # 检查最近的反向目标
                if reverse_targets:
                    nearest_target = reverse_targets[0]
                    distance_pct = abs(nearest_target['distance_pct'])

                    # 距离越近，调整越大
                    if distance_pct < 1.0:
                        result['adjustment_factor'] *= 1.5
                        result['reason'] = f"检测到近距离反向流动性 ({distance_pct:.2f}%)"
                    elif distance_pct < 2.0:
                        result['adjustment_factor'] *= 1.2
                        result['reason'] = f"检测到中距离反向流动性 ({distance_pct:.2f}%)"
                    else:
                        result['reason'] = f"检测到远距离反向流动性 ({distance_pct:.2f}%)"
                else:
                    result['reason'] = "反向流动性增加"

            # 特殊情况：如果盈利较小且反向流动性激增
            if profit_pct < 1.5 and liquidity_ratio > 2.0:
                result['adjustment_factor'] *= 1.3
                result['reason'] += " (小幅盈利+高流动性风险)"

        except Exception as e:
            self.logger.error(f"检查反向流动性失败: {e}")

        return result

    def apply_liquidity_adjustment(self, base_stop: float, adjustment: Dict,
                                   position_side: str, current_price: float,
                                   profit_pct: float) -> float:
        """
        应用流动性调整到止损位置
        """
        if not adjustment['adjusted']:
            return base_stop

        # 计算调整后的保留利润比例
        # 例如：原本保留80%利润，高流动性时可能调整为88%或90%
        adjustment_factor = adjustment['adjustment_factor']

        if position_side == 'LONG':
            # 多头：提高止损价格（减少回撤空间）
            # 原止损到当前价的距离
            original_distance = current_price - base_stop
            # 减少距离（提高止损）
            new_distance = original_distance * (1 - adjustment_factor)
            adjusted_stop = current_price - new_distance

        else:
            # 空头：降低止损价格（减少回撤空间）
            original_distance = base_stop - current_price
            new_distance = original_distance * (1 - adjustment_factor)
            adjusted_stop = current_price + new_distance

        # 确保调整后仍然保护部分利润
        min_profit_to_keep = profit_pct * 0.3  # 至少保留30%的利润

        if position_side == 'LONG':
            min_stop = entry_price * (1 + min_profit_to_keep / 100)
            adjusted_stop = max(adjusted_stop, min_stop)
        else:
            max_stop = entry_price * (1 - min_profit_to_keep / 100)
            adjusted_stop = min(adjusted_stop, max_stop)

        return adjusted_stop

    def should_update_liquidity(self, symbol: str) -> bool:
        """检查是否需要更新流动性数据"""
        last_check = self.last_liquidity_check.get(symbol, 0)
        current_time = datetime.now().timestamp()

        return (current_time - last_check) > self.liquidity_config['liquidity_check_interval']

    def update_liquidity_data(self, symbol: str, current_price: float):
        """更新流动性数据缓存"""
        if not self.liquidity_hunter:
            return

        try:
            # 获取最新流动性分布
            liquidity_levels = self.liquidity_hunter.calculate_liquidation_levels(
                symbol, current_price
            )

            # 分离上下方流动性
            above_targets = [t for t in liquidity_levels.get('major_targets', [])
                             if t['side'] == 'above']
            below_targets = [t for t in liquidity_levels.get('major_targets', [])
                             if t['side'] == 'below']

            # 计算总流动性
            above_volume = sum(t['volume'] for t in above_targets)
            below_volume = sum(t['volume'] for t in below_targets)
            total_volume = above_volume + below_volume
            avg_volume = total_volume / 2 if total_volume > 0 else 1

            # 更新缓存
            self.liquidity_cache[symbol] = {
                'above_liquidity': {'total_volume': above_volume},
                'below_liquidity': {'total_volume': below_volume},
                'above_targets': above_targets,
                'below_targets': below_targets,
                'avg_liquidity': avg_volume,
                'timestamp': datetime.now()
            }

            self.last_liquidity_check[symbol] = datetime.now().timestamp()

        except Exception as e:
            self.logger.error(f"更新流动性数据失败: {e}")

    def is_better_stop_level(self, new_stop: float, current_stop: float,
                             position_side: str) -> bool:
        """判断新止损是否更好"""
        if current_stop == 0:
            return True

        if position_side == 'LONG':
            return new_stop > current_stop
        else:
            return new_stop < current_stop

    def print_stop_update(self, symbol: str, position_side: str,
                          old_stop: float, new_stop: float,
                          profit_pct: float, adjustment: Dict):
        """打印止损更新信息"""
        if adjustment['adjusted']:
            print_colored(
                f"\n📊 {symbol} 流动性调整止损更新:",
                Colors.YELLOW + Colors.BOLD
            )
            print_colored(
                f"   • 持仓方向: {position_side}",
                Colors.INFO
            )
            print_colored(
                f"   • 当前盈利: {profit_pct:.2f}%",
                Colors.GREEN
            )
            print_colored(
                f"   • 原止损: ${old_stop:.4f}",
                Colors.INFO
            )
            print_colored(
                f"   • 新止损: ${new_stop:.4f}",
                Colors.CYAN
            )
            print_colored(
                f"   • 调整原因: {adjustment['reason']}",
                Colors.WARNING
            )
            print_colored(
                f"   • 调整幅度: {adjustment['adjustment_factor']:.1%}",
                Colors.INFO
            )
        else:
            print_colored(
                f"📈 {symbol} {position_side} 更新移动止损: "
                f"${old_stop:.4f} → ${new_stop:.4f} "
                f"(盈利: {profit_pct:.2f}%)",
                Colors.GREEN
            )


# =====================================================
# 集成到现有的持仓管理系统
# =====================================================

def enhance_position_management(self):
    """
    增强的持仓管理 - 添加流动性感知止损
    """
    if not self.open_positions:
        return

    positions_to_remove = []

    for position in self.open_positions:
        try:
            symbol = position['symbol']

            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 获取市场数据
            df = self.get_historical_data(symbol)

            # 检查是否需要更新止损
            if hasattr(self, 'liquidity_stop_loss') and self.liquidity_stop_loss:
                stop_update = self.liquidity_stop_loss.update_position_stop_loss(
                    position, current_price, {'df': df}
                )

                if stop_update['should_update']:
                    # 更新持仓信息
                    position['current_stop_level'] = stop_update['new_stop_level']
                    position['trailing_active'] = True
                    position['last_stop_update'] = datetime.now()

                    # 如果是流动性调整，记录原因
                    if stop_update.get('liquidity_adjusted'):
                        position['stop_adjustment_reason'] = stop_update['adjustment_reason']

            # 检查是否触发止损
            position_side = position.get('position_side', 'LONG')
            stop_level = position.get('current_stop_level', 0)

            if stop_level > 0:
                if (position_side == 'LONG' and current_price <= stop_level) or \
                        (position_side == 'SHORT' and current_price >= stop_level):

                    print_colored(
                        f"\n⚠️ {symbol} 触发止损!",
                        Colors.RED + Colors.BOLD
                    )

                    # 执行平仓
                    success, order = self.close_position(position)
                    if success:
                        positions_to_remove.append(position)

                        # 记录止损原因
                        reason = position.get('stop_adjustment_reason', '正常移动止损')
                        self.logger.info(
                            f"止损平仓: {symbol} {position_side} @ {current_price}, "
                            f"原因: {reason}"
                        )

        except Exception as e:
            self.logger.error(f"管理{symbol}持仓时出错: {e}")

    # 移除已平仓的持仓
    for pos in positions_to_remove:
        self.open_positions.remove(pos)