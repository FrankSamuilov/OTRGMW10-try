"""
性能监控器类 - 基于市场微观结构博弈理论的交易表现分析
追踪交易表现、分析策略效果、识别优化机会
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict
from logger_utils import Colors, print_colored


class PerformanceMonitor:
    """
    交易性能监控与分析系统

    核心功能：
    1. 实时交易表现追踪
    2. 基于市场微观结构的策略效果分析
    3. 聪明钱跟随效果评估
    4. 操纵环境下的表现统计
    5. 多维度性能指标计算
    """

    def __init__(self, save_dir: str = "performance_data"):
        """
        初始化性能监控器

        参数:
            save_dir: 性能数据保存目录
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 交易记录
        self.trades = []
        self.open_positions = {}
        self.closed_positions = []

        # 性能统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': 0.0,
            'current_balance': 0.0,
            'start_balance': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0
        }

        # 市场微观结构相关统计
        self.microstructure_stats = {
            'trades_with_manipulation': 0,
            'manipulation_win_rate': 0.0,
            'smart_money_follow_trades': 0,
            'smart_money_follow_win_rate': 0.0,
            'high_toxicity_trades': 0,
            'high_toxicity_win_rate': 0.0,
            'stop_hunt_avoided': 0,
            'fvg_trades': 0,
            'fvg_win_rate': 0.0
        }

        # 按交易对统计
        self.symbol_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'win_rate': 0.0,
            'avg_holding_time': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        })

        # 按时间段统计
        self.hourly_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0.0})
        self.daily_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0.0})

        # 策略标签统计
        self.strategy_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'profit': 0.0,
            'win_rate': 0.0
        })

        # 退出类型统计
        self.exit_stats = defaultdict(lambda: {
            'count': 0,
            'total_profit': 0.0,
            'avg_profit': 0.0
        })

        # 日志器
        self.logger = logging.getLogger('PerformanceMonitor')

        # 加载历史数据
        self._load_historical_data()

        print_colored("✅ 性能监控系统初始化完成", Colors.GREEN)

    def record_trade_open(self, trade_data: Dict[str, Any]):
        """
        记录开仓交易

        参数:
            trade_data: 交易数据，包含symbol, side, price, quantity,
                       leverage, stop_loss, take_profit, market_analysis等
        """
        try:
            trade_id = f"{trade_data['symbol']}_{datetime.now().timestamp()}"

            position = {
                'id': trade_id,
                'symbol': trade_data['symbol'],
                'side': trade_data['side'],
                'entry_price': trade_data['price'],
                'quantity': trade_data['quantity'],
                'entry_time': datetime.now(),
                'leverage': trade_data.get('leverage', 1),
                'stop_loss': trade_data.get('stop_loss'),
                'take_profit': trade_data.get('take_profit'),
                'market_analysis': trade_data.get('market_analysis', {}),
                'strategy_tags': trade_data.get('strategy_tags', []),
                'status': 'open',
                'unrealized_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'price_history': [(datetime.now(), trade_data['price'])]
            }

            # 记录市场微观结构相关标签
            if 'market_analysis' in trade_data:
                analysis = trade_data['market_analysis']

                # 操纵环境标记
                if analysis.get('manipulation_score', 0) > 0.7:
                    position['strategy_tags'].append('high_manipulation')
                    self.microstructure_stats['trades_with_manipulation'] += 1

                # 聪明钱跟随标记
                if analysis.get('smart_money_signal') == 'follow':
                    position['strategy_tags'].append('smart_money_follow')
                    self.microstructure_stats['smart_money_follow_trades'] += 1

                # 高毒性环境标记
                if analysis.get('order_flow_toxicity', 0) > 0.35:
                    position['strategy_tags'].append('high_toxicity')
                    self.microstructure_stats['high_toxicity_trades'] += 1

                # FVG交易标记
                if analysis.get('fvg_signal'):
                    position['strategy_tags'].append('fvg_trade')
                    self.microstructure_stats['fvg_trades'] += 1

            self.open_positions[trade_id] = position
            self.stats['total_trades'] += 1

            # 更新时间统计
            hour = datetime.now().hour
            date_str = datetime.now().strftime('%Y-%m-%d')
            self.hourly_stats[hour]['trades'] += 1
            self.daily_stats[date_str]['trades'] += 1

            # 更新交易对统计
            self.symbol_stats[trade_data['symbol']]['trades'] += 1

            print_colored(f"📊 记录开仓: {trade_data['symbol']} {trade_data['side']} @ {trade_data['price']}",
                          Colors.INFO)

            # 保存数据
            self._save_current_state()

        except Exception as e:
            self.logger.error(f"记录开仓失败: {e}")

    def update_position(self, trade_id: str, current_price: float, market_data: Dict[str, Any] = None):
        """
        更新持仓状态

        参数:
            trade_id: 交易ID
            current_price: 当前价格
            market_data: 当前市场数据（可选）
        """
        if trade_id not in self.open_positions:
            return

        position = self.open_positions[trade_id]

        # 计算未实现盈亏
        if position['side'] == 'BUY':
            pnl = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - current_price) / position['entry_price']

        position['unrealized_pnl'] = pnl
        position['current_price'] = current_price
        position['last_update'] = datetime.now()

        # 更新最大盈利/亏损
        if pnl > position['max_profit']:
            position['max_profit'] = pnl
        if pnl < position['max_loss']:
            position['max_loss'] = pnl

        # 记录价格历史
        position['price_history'].append((datetime.now(), current_price))
        if len(position['price_history']) > 100:  # 限制历史记录长度
            position['price_history'] = position['price_history'][-100:]

        # 如果提供了市场数据，记录关键变化
        if market_data:
            position['market_updates'] = position.get('market_updates', [])
            position['market_updates'].append({
                'time': datetime.now(),
                'manipulation_score': market_data.get('manipulation_score'),
                'order_flow_toxicity': market_data.get('order_flow_toxicity'),
                'smart_money_signal': market_data.get('smart_money_signal')
            })

    def record_trade_close(self, trade_id: str, exit_price: float, exit_reason: str,
                           exit_analysis: Dict[str, Any] = None):
        """
        记录平仓交易

        参数:
            trade_id: 交易ID
            exit_price: 退出价格
            exit_reason: 退出原因
            exit_analysis: 退出时的市场分析（可选）
        """
        if trade_id not in self.open_positions:
            self.logger.warning(f"未找到交易ID: {trade_id}")
            return

        position = self.open_positions[trade_id]
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['exit_analysis'] = exit_analysis or {}
        position['status'] = 'closed'

        # 计算最终盈亏
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - exit_price) / position['entry_price']

        position['realized_pnl'] = pnl
        position['profit_amount'] = pnl * position['entry_price'] * position['quantity']

        # 计算持仓时间
        holding_time = (position['exit_time'] - position['entry_time']).total_seconds() / 3600
        position['holding_hours'] = holding_time

        # 更新统计
        self._update_stats_on_close(position)

        # 移动到已平仓列表
        self.closed_positions.append(position)
        del self.open_positions[trade_id]

        # 打印交易结果
        profit_color = Colors.GREEN if pnl > 0 else Colors.RED
        print_colored(f"📊 平仓: {position['symbol']} {position['side']}", Colors.INFO)
        print_colored(f"  盈亏: {profit_color}{pnl * 100:.2f}%{Colors.RESET}", Colors.INFO)
        print_colored(f"  持仓时间: {holding_time:.1f}小时", Colors.INFO)
        print_colored(f"  退出原因: {exit_reason}", Colors.INFO)

        # 保存数据
        self._save_current_state()

        # 定期生成报告
        if self.stats['total_trades'] % 10 == 0:
            self.generate_performance_report()

    def _update_stats_on_close(self, position: Dict[str, Any]):
        """更新平仓后的统计数据"""
        pnl = position['realized_pnl']
        symbol = position['symbol']

        # 基础统计
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['total_profit'] += pnl
            self.stats['current_consecutive_wins'] += 1
            self.stats['current_consecutive_losses'] = 0

            if self.stats['current_consecutive_wins'] > self.stats['max_consecutive_wins']:
                self.stats['max_consecutive_wins'] = self.stats['current_consecutive_wins']
        else:
            self.stats['losing_trades'] += 1
            self.stats['total_loss'] += abs(pnl)
            self.stats['current_consecutive_losses'] += 1
            self.stats['current_consecutive_wins'] = 0

            if self.stats['current_consecutive_losses'] > self.stats['max_consecutive_losses']:
                self.stats['max_consecutive_losses'] = self.stats['current_consecutive_losses']

        # 计算胜率
        total_closed = self.stats['winning_trades'] + self.stats['losing_trades']
        if total_closed > 0:
            self.stats['win_rate'] = self.stats['winning_trades'] / total_closed

        # 计算盈利因子
        if self.stats['total_loss'] > 0:
            self.stats['profit_factor'] = self.stats['total_profit'] / self.stats['total_loss']

        # 计算平均盈亏
        if self.stats['winning_trades'] > 0:
            self.stats['average_win'] = self.stats['total_profit'] / self.stats['winning_trades']
        if self.stats['losing_trades'] > 0:
            self.stats['average_loss'] = self.stats['total_loss'] / self.stats['losing_trades']

        # 更新交易对统计
        symbol_stat = self.symbol_stats[symbol]
        if pnl > 0:
            symbol_stat['wins'] += 1
        else:
            symbol_stat['losses'] += 1

        symbol_stat['profit'] += pnl
        symbol_stat['win_rate'] = symbol_stat['wins'] / symbol_stat['trades'] if symbol_stat['trades'] > 0 else 0

        if pnl > symbol_stat['best_trade']:
            symbol_stat['best_trade'] = pnl
        if pnl < symbol_stat['worst_trade']:
            symbol_stat['worst_trade'] = pnl

        # 更新策略标签统计
        for tag in position.get('strategy_tags', []):
            tag_stat = self.strategy_stats[tag]
            tag_stat['trades'] += 1
            if pnl > 0:
                tag_stat['wins'] += 1
            tag_stat['profit'] += pnl
            tag_stat['win_rate'] = tag_stat['wins'] / tag_stat['trades']

        # 更新退出类型统计
        exit_reason = position['exit_reason']
        self.exit_stats[exit_reason]['count'] += 1
        self.exit_stats[exit_reason]['total_profit'] += pnl
        self.exit_stats[exit_reason]['avg_profit'] = (
                self.exit_stats[exit_reason]['total_profit'] /
                self.exit_stats[exit_reason]['count']
        )

        # 更新时间统计
        hour = position['exit_time'].hour
        date_str = position['exit_time'].strftime('%Y-%m-%d')

        if pnl > 0:
            self.hourly_stats[hour]['wins'] += 1
            self.daily_stats[date_str]['wins'] += 1

        self.hourly_stats[hour]['profit'] += pnl
        self.daily_stats[date_str]['profit'] += pnl

        # 更新市场微观结构相关统计
        self._update_microstructure_stats(position)

    def _update_microstructure_stats(self, position: Dict[str, Any]):
        """更新市场微观结构相关统计"""
        tags = position.get('strategy_tags', [])
        pnl = position['realized_pnl']

        # 操纵环境胜率
        if 'high_manipulation' in tags:
            total_manip = self.microstructure_stats['trades_with_manipulation']
            if total_manip > 0:
                manip_wins = len([p for p in self.closed_positions
                                  if 'high_manipulation' in p.get('strategy_tags', [])
                                  and p['realized_pnl'] > 0])
                self.microstructure_stats['manipulation_win_rate'] = manip_wins / total_manip

        # 聪明钱跟随胜率
        if 'smart_money_follow' in tags:
            total_smart = self.microstructure_stats['smart_money_follow_trades']
            if total_smart > 0:
                smart_wins = len([p for p in self.closed_positions
                                  if 'smart_money_follow' in p.get('strategy_tags', [])
                                  and p['realized_pnl'] > 0])
                self.microstructure_stats['smart_money_follow_win_rate'] = smart_wins / total_smart

        # 高毒性环境胜率
        if 'high_toxicity' in tags:
            total_toxic = self.microstructure_stats['high_toxicity_trades']
            if total_toxic > 0:
                toxic_wins = len([p for p in self.closed_positions
                                  if 'high_toxicity' in p.get('strategy_tags', [])
                                  and p['realized_pnl'] > 0])
                self.microstructure_stats['high_toxicity_win_rate'] = toxic_wins / total_toxic

        # FVG交易胜率
        if 'fvg_trade' in tags:
            total_fvg = self.microstructure_stats['fvg_trades']
            if total_fvg > 0:
                fvg_wins = len([p for p in self.closed_positions
                                if 'fvg_trade' in p.get('strategy_tags', [])
                                and p['realized_pnl'] > 0])
                self.microstructure_stats['fvg_win_rate'] = fvg_wins / total_fvg

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率

        参数:
            risk_free_rate: 无风险利率（年化）
        """
        if not self.closed_positions:
            return 0.0

        # 获取所有收益率
        returns = [p['realized_pnl'] for p in self.closed_positions]

        if len(returns) < 2:
            return 0.0

        # 计算平均收益和标准差
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 计算夏普比率（假设每天交易）
        daily_risk_free = risk_free_rate / 365
        sharpe = (avg_return - daily_risk_free) / std_return * np.sqrt(365)

        self.stats['sharpe_ratio'] = sharpe
        return sharpe

    def calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.closed_positions:
            return 0.0

        # 计算累积收益曲线
        cumulative_returns = []
        cumulative = 0

        for position in sorted(self.closed_positions, key=lambda x: x['exit_time']):
            cumulative += position['realized_pnl']
            cumulative_returns.append(cumulative)

        # 计算最大回撤
        peak = cumulative_returns[0]
        max_dd = 0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        self.stats['max_drawdown'] = max_dd
        return max_dd

    def generate_performance_report(self, detailed: bool = True) -> Dict[str, Any]:
        """
        生成性能报告

        参数:
            detailed: 是否生成详细报告
        """
        # 计算额外指标
        self.calculate_sharpe_ratio()
        self.calculate_max_drawdown()

        report = {
            'generated_at': datetime.now().isoformat(),
            'basic_stats': self.stats.copy(),
            'microstructure_stats': self.microstructure_stats.copy(),
            'symbol_performance': dict(self.symbol_stats),
            'strategy_performance': dict(self.strategy_stats),
            'exit_analysis': dict(self.exit_stats)
        }

        if detailed:
            # 添加时间分析
            report['hourly_performance'] = dict(self.hourly_stats)
            report['daily_performance'] = dict(self.daily_stats)

            # 添加最佳和最差交易
            if self.closed_positions:
                sorted_trades = sorted(self.closed_positions,
                                       key=lambda x: x['realized_pnl'],
                                       reverse=True)
                report['best_trades'] = [
                    {
                        'symbol': t['symbol'],
                        'side': t['side'],
                        'pnl': t['realized_pnl'],
                        'holding_hours': t['holding_hours'],
                        'exit_reason': t['exit_reason']
                    }
                    for t in sorted_trades[:5]
                ]
                report['worst_trades'] = [
                    {
                        'symbol': t['symbol'],
                        'side': t['side'],
                        'pnl': t['realized_pnl'],
                        'holding_hours': t['holding_hours'],
                        'exit_reason': t['exit_reason']
                    }
                    for t in sorted_trades[-5:]
                ]

        # 打印报告摘要
        self._print_report_summary(report)

        # 保存报告
        report_path = os.path.join(
            self.save_dir,
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _print_report_summary(self, report: Dict[str, Any]):
        """打印报告摘要"""
        stats = report['basic_stats']
        micro_stats = report['microstructure_stats']

        print_colored("\n" + "=" * 60, Colors.BLUE)
        print_colored("📊 交易性能报告", Colors.BLUE + Colors.BOLD)
        print_colored("=" * 60, Colors.BLUE)

        # 基础统计
        print_colored("\n📈 基础统计:", Colors.CYAN)
        print_colored(f"  总交易次数: {stats['total_trades']}", Colors.INFO)
        print_colored(f"  胜率: {stats['win_rate'] * 100:.1f}%", Colors.INFO)
        print_colored(f"  盈利因子: {stats['profit_factor']:.2f}", Colors.INFO)
        print_colored(f"  夏普比率: {stats['sharpe_ratio']:.2f}", Colors.INFO)
        print_colored(f"  最大回撤: {stats['max_drawdown'] * 100:.1f}%", Colors.INFO)
        print_colored(f"  平均盈利: {stats['average_win'] * 100:.2f}%", Colors.GREEN)
        print_colored(f"  平均亏损: {stats['average_loss'] * 100:.2f}%", Colors.RED)
        print_colored(f"  最大连续盈利: {stats['max_consecutive_wins']}", Colors.GREEN)
        print_colored(f"  最大连续亏损: {stats['max_consecutive_losses']}", Colors.RED)

        # 市场微观结构统计
        print_colored("\n🔬 市场微观结构分析:", Colors.CYAN)
        print_colored(f"  操纵环境交易: {micro_stats['trades_with_manipulation']}次, "
                      f"胜率: {micro_stats['manipulation_win_rate'] * 100:.1f}%", Colors.INFO)
        print_colored(f"  聪明钱跟随: {micro_stats['smart_money_follow_trades']}次, "
                      f"胜率: {micro_stats['smart_money_follow_win_rate'] * 100:.1f}%", Colors.INFO)
        print_colored(f"  高毒性环境: {micro_stats['high_toxicity_trades']}次, "
                      f"胜率: {micro_stats['high_toxicity_win_rate'] * 100:.1f}%", Colors.INFO)
        print_colored(f"  FVG交易: {micro_stats['fvg_trades']}次, "
                      f"胜率: {micro_stats['fvg_win_rate'] * 100:.1f}%", Colors.INFO)

        # 最佳交易对
        if report['symbol_performance']:
            print_colored("\n💰 最佳交易对:", Colors.CYAN)
            sorted_symbols = sorted(report['symbol_performance'].items(),
                                    key=lambda x: x[1]['profit'],
                                    reverse=True)
            for symbol, perf in sorted_symbols[:3]:
                print_colored(f"  {symbol}: {perf['trades']}次交易, "
                              f"胜率: {perf['win_rate'] * 100:.1f}%, "
                              f"总盈利: {perf['profit'] * 100:.2f}%", Colors.INFO)

        print_colored("=" * 60 + "\n", Colors.BLUE)

    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计数据"""
        return {
            'basic_stats': self.stats.copy(),
            'microstructure_stats': self.microstructure_stats.copy(),
            'open_positions': len(self.open_positions),
            'total_closed': len(self.closed_positions)
        }

    def _save_current_state(self):
        """保存当前状态到文件"""
        try:
            state = {
                'stats': self.stats,
                'microstructure_stats': self.microstructure_stats,
                'open_positions': self.open_positions,
                'closed_positions': self.closed_positions[-100:],  # 只保存最近100笔
                'symbol_stats': dict(self.symbol_stats),
                'strategy_stats': dict(self.strategy_stats),
                'exit_stats': dict(self.exit_stats),
                'last_update': datetime.now().isoformat()
            }

            state_path = os.path.join(self.save_dir, 'current_state.json')
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")

    def _load_historical_data(self):
        """加载历史数据"""
        try:
            state_path = os.path.join(self.save_dir, 'current_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)

                # 恢复统计数据
                self.stats.update(state.get('stats', {}))
                self.microstructure_stats.update(state.get('microstructure_stats', {}))

                # 恢复交易记录
                self.closed_positions = state.get('closed_positions', [])

                # 恢复各类统计
                for symbol, stats in state.get('symbol_stats', {}).items():
                    self.symbol_stats[symbol].update(stats)

                for strategy, stats in state.get('strategy_stats', {}).items():
                    self.strategy_stats[strategy].update(stats)

                for exit_type, stats in state.get('exit_stats', {}).items():
                    self.exit_stats[exit_type].update(stats)

                print_colored(f"✅ 加载历史数据: {len(self.closed_positions)}笔已完成交易", Colors.GREEN)

        except Exception as e:
            self.logger.warning(f"加载历史数据失败: {e}")
            print_colored("⚠️ 无法加载历史数据，从零开始记录", Colors.WARNING)