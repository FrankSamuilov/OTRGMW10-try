import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from logger_utils import Colors, print_colored


class PositionVisualizer:
    """
    持仓可视化系统
    """

    def __init__(self, liquidity_hunter=None, logger=None):
        self.liquidity_hunter = liquidity_hunter
        self.logger = logger

        # 可视化配置
        self.display_config = {
            'refresh_interval': 5,  # 5秒刷新一次
            'price_decimals': 4,  # 价格小数位
            'pct_decimals': 2,  # 百分比小数位
        }

        print_colored("✅ 持仓可视化系统初始化完成", Colors.GREEN)

    def display_position_dashboard(self, positions: List[Dict], market_data: Dict):
        """
        显示持仓仪表板
        """
        if not positions:
            print_colored("\n📊 当前无持仓", Colors.GRAY)
            return

        # 清空屏幕（可选）
        # os.system('cls' if os.name == 'nt' else 'clear')

        print_colored("\n" + "=" * 100, Colors.BLUE)
        print_colored("📊 实时持仓监控面板", Colors.CYAN + Colors.BOLD)
        print_colored("=" * 100, Colors.BLUE)
        print_colored(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.GRAY)

        total_pnl = 0
        total_value = 0

        for idx, position in enumerate(positions, 1):
            # 获取实时数据
            symbol = position['symbol']
            current_price = market_data.get(symbol, {}).get('current_price', 0)

            if current_price == 0:
                continue

            # 计算盈亏
            pnl_info = self.calculate_position_pnl(position, current_price)
            total_pnl += pnl_info['pnl_amount']
            total_value += pnl_info['position_value']

            # 显示单个持仓
            self.display_single_position(idx, position, pnl_info, current_price)

            # 显示流动性突破预测
            if self.liquidity_hunter:
                self.display_liquidity_prediction(symbol, position, current_price)

            print_colored("-" * 100, Colors.GRAY)

        # 显示汇总信息
        self.display_summary(total_pnl, total_value, len(positions))

    def calculate_position_pnl(self, position: Dict, current_price: float) -> Dict:
        """
        计算持仓盈亏
        """
        entry_price = position['entry_price']
        quantity = position['quantity']
        position_side = position.get('position_side', 'LONG')
        leverage = position.get('leverage', 1)

        # 计算盈亏
        if position_side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_amount = (current_price - entry_price) * quantity
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            pnl_amount = (entry_price - current_price) * quantity

        # 计算持仓价值
        position_value = current_price * quantity

        return {
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'position_value': position_value,
            'leveraged_pnl': pnl_amount * leverage
        }

    def display_single_position(self, idx: int, position: Dict,
                                pnl_info: Dict, current_price: float):
        """
        显示单个持仓信息
        """
        symbol = position['symbol']
        position_side = position.get('position_side', 'LONG')
        entry_price = position['entry_price']
        quantity = position['quantity']
        leverage = position.get('leverage', 1)

        # 标题
        side_icon = "🔼" if position_side == 'LONG' else "🔽"
        side_color = Colors.GREEN if position_side == 'LONG' else Colors.RED

        print_colored(f"\n{idx}. {side_icon} {symbol} {position_side}",
                      side_color + Colors.BOLD)

        # 基础信息
        print_colored("   📍 基础信息:", Colors.CYAN)
        print_colored(f"      • 入场价格: ${entry_price:.{self.display_config['price_decimals']}f}", Colors.INFO)
        print_colored(f"      • 当前价格: ${current_price:.{self.display_config['price_decimals']}f}", Colors.INFO)
        print_colored(f"      • 持仓数量: {quantity:.6f}", Colors.INFO)
        print_colored(f"      • 杠杆倍数: {leverage}x", Colors.INFO)
        print_colored(f"      • 持仓价值: ${pnl_info['position_value']:.2f}", Colors.INFO)

        # 盈亏信息
        pnl_pct = pnl_info['pnl_pct']
        pnl_amount = pnl_info['pnl_amount']
        pnl_color = Colors.GREEN if pnl_pct > 0 else Colors.RED if pnl_pct < 0 else Colors.GRAY

        print_colored("   💰 盈亏状态:", Colors.CYAN)
        print_colored(f"      • 盈亏比例: {pnl_color}{pnl_pct:+.{self.display_config['pct_decimals']}f}%{Colors.RESET}",
                      Colors.INFO)
        print_colored(f"      • 盈亏金额: {pnl_color}${pnl_amount:+.2f}{Colors.RESET}", Colors.INFO)
        print_colored(f"      • 杠杆盈亏: {pnl_color}${pnl_info['leveraged_pnl']:+.2f}{Colors.RESET}", Colors.INFO)

        # 止损信息
        self.display_stop_loss_info(position, current_price, pnl_pct)

        # 持仓时间
        open_time = position.get('open_time', 0)
        if open_time:
            holding_time = datetime.now() - datetime.fromtimestamp(open_time)
            hours = holding_time.total_seconds() / 3600
            print_colored(f"   ⏱️  持仓时间: {hours:.1f}小时", Colors.INFO)

    def display_stop_loss_info(self, position: Dict, current_price: float, pnl_pct: float):
        """
        显示止损信息
        """
        print_colored("   🛡️  止损保护:", Colors.CYAN)

        position_side = position.get('position_side', 'LONG')
        current_stop = position.get('current_stop_level', 0)
        trailing_active = position.get('trailing_active', False)
        entry_price = position['entry_price']

        # 是否激活移动止损
        if trailing_active:
            print_colored(f"      • 移动止损: {Colors.GREEN}已激活{Colors.RESET}", Colors.INFO)

            # 当前止损位
            if current_stop > 0:
                # 计算止损距离
                if position_side == 'LONG':
                    stop_distance = (current_price - current_stop) / current_price * 100
                    stop_protection = (current_stop - entry_price) / entry_price * 100
                else:
                    stop_distance = (current_stop - current_price) / current_price * 100
                    stop_protection = (entry_price - current_stop) / entry_price * 100

                print_colored(f"      • 止损价格: ${current_stop:.{self.display_config['price_decimals']}f}",
                              Colors.INFO)
                print_colored(f"      • 距离现价: {stop_distance:.{self.display_config['pct_decimals']}f}%",
                              Colors.INFO)
                print_colored(
                    f"      • 保护利润: {Colors.GREEN}{stop_protection:.{self.display_config['pct_decimals']}f}%{Colors.RESET}",
                    Colors.INFO)

                # 流动性调整信息
                if position.get('stop_adjustment_reason'):
                    print_colored(
                        f"      • 调整原因: {Colors.YELLOW}{position['stop_adjustment_reason']}{Colors.RESET}",
                        Colors.INFO)

        else:
            # 未激活原因
            activation_threshold = 0.618
            if pnl_pct < activation_threshold:
                needed = activation_threshold - pnl_pct
                print_colored(f"      • 移动止损: {Colors.GRAY}未激活{Colors.RESET}", Colors.INFO)
                print_colored(
                    f"      • 激活条件: 盈利达到{activation_threshold}% (还需{needed:.{self.display_config['pct_decimals']}f}%)",
                    Colors.INFO)

            # 固定止损
            initial_stop = position.get('stop_loss', 0)
            if initial_stop > 0:
                print_colored(f"      • 固定止损: ${initial_stop:.{self.display_config['price_decimals']}f}",
                              Colors.INFO)

    def display_liquidity_prediction(self, symbol: str, position: Dict, current_price: float):
        """
        显示流动性突破预测
        """
        try:
            # 获取流动性数据
            liquidity_levels = self.liquidity_hunter.calculate_liquidation_levels(symbol, current_price)
            major_targets = liquidity_levels.get('major_targets', [])

            if not major_targets:
                return

            print_colored("   🎯 流动性突破预测:", Colors.CYAN)

            position_side = position.get('position_side', 'LONG')

            # 根据持仓方向选择相关目标
            if position_side == 'LONG':
                # 多头关注上方目标（获利）和下方目标（风险）
                profit_targets = [t for t in major_targets if t['side'] == 'above'][:2]
                risk_targets = [t for t in major_targets if t['side'] == 'below'][:1]
            else:
                # 空头关注下方目标（获利）和上方目标（风险）
                profit_targets = [t for t in major_targets if t['side'] == 'below'][:2]
                risk_targets = [t for t in major_targets if t['side'] == 'above'][:1]

            # 显示获利目标
            if profit_targets:
                print_colored("      📈 获利目标:", Colors.GREEN)
                for i, target in enumerate(profit_targets, 1):
                    self.display_liquidity_target(target, current_price, i, 'profit')

            # 显示风险目标
            if risk_targets:
                print_colored("      ⚠️  风险区域:", Colors.RED)
                for i, target in enumerate(risk_targets, 1):
                    self.display_liquidity_target(target, current_price, i, 'risk')

            # 突破概率分析
            self.analyze_breakout_probability(liquidity_levels, position_side, current_price)

        except Exception as e:
            self.logger.error(f"显示流动性预测失败: {e}")

    def display_liquidity_target(self, target: Dict, current_price: float,
                                 idx: int, target_type: str):
        """
        显示单个流动性目标
        """
        price = target['price']
        distance_pct = target['distance_pct']
        volume = target['volume']
        strength = target['strength']

        # 估算影响人数
        estimated_traders = self.estimate_affected_traders(volume, strength)

        # 选择图标
        if target_type == 'profit':
            icon = "🎯"
            color = Colors.GREEN
        else:
            icon = "⚠️"
            color = Colors.RED

        print_colored(
            f"         {icon} 目标{idx}: ${price:.{self.display_config['price_decimals']}f} "
            f"({distance_pct:+.{self.display_config['pct_decimals']}f}%) "
            f"强度:{strength:.1f} "
            f"预计{estimated_traders}人触发",
            color
        )

    def estimate_affected_traders(self, volume: float, strength: float) -> str:
        """
        估算受影响的交易者数量
        """
        # 基于成交量和强度估算
        # 这是一个简化的估算模型
        base_traders = volume / 10000  # 假设平均每人10000单位

        if strength > 0.8:
            multiplier = 2.0
            category = "大量"
        elif strength > 0.5:
            multiplier = 1.5
            category = "较多"
        elif strength > 0.3:
            multiplier = 1.0
            category = "部分"
        else:
            multiplier = 0.5
            category = "少量"

        estimated = int(base_traders * multiplier)

        if estimated > 1000:
            return f"{category}(>1000)"
        elif estimated > 500:
            return f"{category}(500-1000)"
        elif estimated > 100:
            return f"{category}(100-500)"
        else:
            return f"{category}(<100)"

    def analyze_breakout_probability(self, liquidity_levels: Dict,
                                     position_side: str, current_price: float):
        """
        分析突破概率
        """
        print_colored("      📊 突破分析:", Colors.CYAN)

        # 获取最近的目标
        targets = liquidity_levels.get('major_targets', [])
        if not targets:
            return

        nearest_above = next((t for t in targets if t['side'] == 'above'), None)
        nearest_below = next((t for t in targets if t['side'] == 'below'), None)

        # 计算突破压力
        above_pressure = sum(t['attraction_score'] for t in targets if t['side'] == 'above')
        below_pressure = sum(t['attraction_score'] for t in targets if t['side'] == 'below')

        total_pressure = above_pressure + below_pressure
        if total_pressure > 0:
            up_probability = above_pressure / total_pressure * 100
            down_probability = below_pressure / total_pressure * 100
        else:
            up_probability = down_probability = 50

        # 显示突破概率
        print_colored(f"         • 向上突破概率: {up_probability:.1f}%",
                      Colors.GREEN if up_probability > 60 else Colors.INFO)
        print_colored(f"         • 向下突破概率: {down_probability:.1f}%",
                      Colors.RED if down_probability > 60 else Colors.INFO)

        # 推荐操作
        if position_side == 'LONG':
            if down_probability > 65:
                print_colored("         💡 建议: 考虑减仓或收紧止损", Colors.YELLOW)
            elif up_probability > 70:
                print_colored("         💡 建议: 可以适当放宽止损，等待突破", Colors.GREEN)
        else:  # SHORT
            if up_probability > 65:
                print_colored("         💡 建议: 考虑减仓或收紧止损", Colors.YELLOW)
            elif down_probability > 70:
                print_colored("         💡 建议: 可以适当放宽止损，等待突破", Colors.GREEN)

    def display_summary(self, total_pnl: float, total_value: float, position_count: int):
        """
        显示汇总信息
        """
        print_colored("\n" + "=" * 100, Colors.BLUE)
        print_colored("📊 账户汇总", Colors.CYAN + Colors.BOLD)
        print_colored("=" * 100, Colors.BLUE)

        pnl_color = Colors.GREEN if total_pnl > 0 else Colors.RED if total_pnl < 0 else Colors.GRAY

        print_colored(f"   • 持仓数量: {position_count}个", Colors.INFO)
        print_colored(f"   • 总持仓价值: ${total_value:.2f}", Colors.INFO)
        print_colored(f"   • 总盈亏: {pnl_color}${total_pnl:+.2f}{Colors.RESET}", Colors.INFO)
        print_colored(
            f"   • 平均盈亏率: {pnl_color}{(total_pnl / total_value * 100 if total_value > 0 else 0):+.2f}%{Colors.RESET}",
            Colors.INFO)

    def create_visual_chart(self, position: Dict, market_data: Dict) -> Optional[str]:
        """
        创建可视化图表（可选功能）
        """
        try:
            symbol = position['symbol']
            df = market_data.get('df')
            if df is None:
                return None

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={'height_ratios': [3, 1]})

            # 价格图
            ax1.plot(df.index[-100:], df['close'].iloc[-100:],
                     label='价格', color='blue', linewidth=2)

            # 标记入场点
            entry_price = position['entry_price']
            ax1.axhline(y=entry_price, color='green', linestyle='--',
                        label=f'入场价: ${entry_price:.4f}')

            # 标记止损线
            current_stop = position.get('current_stop_level', 0)
            if current_stop > 0:
                ax1.axhline(y=current_stop, color='red', linestyle='--',
                            label=f'止损: ${current_stop:.4f}')

            # 标记流动性目标
            if hasattr(self, 'liquidity_hunter'):
                current_price = df['close'].iloc[-1]
                liquidity_levels = self.liquidity_hunter.calculate_liquidation_levels(
                    symbol, current_price
                )

                for target in liquidity_levels.get('major_targets', [])[:3]:
                    color = 'orange' if target['side'] == 'above' else 'purple'
                    ax1.axhline(y=target['price'], color=color, linestyle=':',
                                alpha=0.7, label=f"流动性: ${target['price']:.4f}")

            ax1.set_title(f'{symbol} - 持仓可视化', fontsize=16)
            ax1.set_ylabel('价格', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # 成交量图
            ax2.bar(df.index[-100:], df['volume'].iloc[-100:],
                    alpha=0.7, color='gray')
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.set_xlabel('时间', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # 保存图表
            filename = f'position_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            plt.close()

            return filename

        except Exception as e:
            self.logger.error(f"创建图表失败: {e}")
            return None


# =====================================================
# 集成到主交易系统
# =====================================================

def enhanced_manage_positions(self):
    """
    增强的持仓管理 - 包含可视化
    """
    if not self.open_positions:
        print_colored("\n📊 当前无持仓", Colors.GRAY)
        return

    # 收集所有持仓的市场数据
    market_data = {}
    for position in self.open_positions:
        symbol = position['symbol']
        try:
            # 获取实时价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 获取历史数据
            df = self.get_historical_data(symbol)

            market_data[symbol] = {
                'current_price': current_price,
                'df': df
            }
        except Exception as e:
            self.logger.error(f"获取{symbol}市场数据失败: {e}")

    # 显示可视化仪表板
    if hasattr(self, 'position_visualizer') and self.position_visualizer:
        self.position_visualizer.display_position_dashboard(
            self.open_positions,
            market_data
        )