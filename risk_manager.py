"""
风险管理器类 - 基于市场微观结构博弈理论
整合了操纵检测、订单流毒性分析、动态仓位管理等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from logger_utils import Colors, print_colored


class RiskManager:
    """
    市场微观结构博弈风险管理器

    核心功能：
    1. 基于操纵环境的动态风险调整
    2. 订单流毒性评估与仓位控制
    3. 止损猎杀区域识别与规避
    4. 聪明钱跟随的风险优化
    5. 多维度风险评分系统
    """

    def __init__(self,
                 max_position_size: float = 30.0,
                 max_daily_loss: float = 5.0,
                 max_drawdown: float = 10.0,
                 base_risk_per_trade: float = 2.0):
        """
        初始化风险管理器

        参数:
            max_position_size: 最大单个仓位占比(%)
            max_daily_loss: 最大日亏损比例(%)
            max_drawdown: 最大回撤比例(%)
            base_risk_per_trade: 每笔交易基础风险(%)
        """
        # 基础风险参数
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.base_risk_per_trade = base_risk_per_trade

        # 账户状态跟踪
        self.daily_loss = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()

        # 市场微观结构参数
        self.manipulation_threshold = 0.7  # 操纵检测阈值
        self.toxicity_threshold = 0.35  # 订单流毒性阈值
        self.smart_money_divergence_threshold = 0.3  # 聪明钱分歧阈值

        # 风险调整系数
        self.risk_multipliers = {
            'manipulation': 0.5,  # 检测到操纵时减仓50%
            'high_toxicity': 0.6,  # 高毒性减仓40%
            'smart_follow': 1.2,  # 跟随聪明钱增仓20%
            'stop_hunt_zone': 0.3,  # 止损猎杀区减仓70%
            'extreme_divergence': 1.5  # 极端分歧增仓50%
        }

        # 历史数据缓存
        self.risk_history = []
        self.manipulation_events = []
        self.stop_hunt_zones = {}

        self.logger = logging.getLogger('RiskManager')
        print_colored("✅ 市场微观结构博弈风险管理器初始化完成", Colors.GREEN)

    def calculate_position_size(self,
                                account_balance: float,
                                entry_price: float,
                                stop_loss: float,
                                market_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        基于市场微观结构分析计算仓位大小

        参数:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            market_analysis: 市场分析结果（包含操纵检测、毒性等）

        返回:
            包含仓位大小和风险参数的字典
        """
        try:
            # 1. 计算基础仓位
            risk_amount = account_balance * (self.base_risk_per_trade / 100)
            stop_loss_distance = abs(entry_price - stop_loss) / entry_price
            base_position_value = risk_amount / stop_loss_distance
            base_position_size = base_position_value / entry_price

            # 2. 风险乘数初始化
            risk_multiplier = 1.0
            risk_factors = []

            # 3. 操纵环境检测调整
            if 'manipulation_score' in market_analysis:
                manip_score = market_analysis['manipulation_score']
                if manip_score > self.manipulation_threshold:
                    risk_multiplier *= self.risk_multipliers['manipulation']
                    risk_factors.append(f"操纵环境检测 (评分: {manip_score:.2f})")

            # 4. 订单流毒性调整
            if 'order_flow_toxicity' in market_analysis:
                toxicity = market_analysis['order_flow_toxicity']
                if toxicity > self.toxicity_threshold:
                    risk_multiplier *= self.risk_multipliers['high_toxicity']
                    risk_factors.append(f"高订单流毒性 (VPIN: {toxicity:.2f})")

            # 5. 聪明钱分析调整
            if 'smart_money_divergence' in market_analysis:
                divergence = market_analysis['smart_money_divergence']
                if abs(divergence) > self.smart_money_divergence_threshold:
                    if divergence > 0:  # 聪明钱看多，散户看空
                        risk_multiplier *= self.risk_multipliers['smart_follow']
                        risk_factors.append(f"跟随聪明钱 (分歧度: {divergence:.2f})")
                    else:  # 极端情况可能需要反向
                        if abs(divergence) > 0.5:
                            risk_multiplier *= self.risk_multipliers['extreme_divergence']
                            risk_factors.append(f"极端分歧机会 (分歧度: {divergence:.2f})")

            # 6. 止损猎杀区域检测
            if self._is_stop_hunt_zone(entry_price, market_analysis):
                risk_multiplier *= self.risk_multipliers['stop_hunt_zone']
                risk_factors.append("止损猎杀区域")

            # 7. 账户状态限制
            if self.daily_loss >= self.max_daily_loss * 0.8:
                risk_multiplier *= 0.5
                risk_factors.append("接近日亏损限制")

            if self.current_drawdown >= self.max_drawdown * 0.8:
                risk_multiplier *= 0.3
                risk_factors.append("接近最大回撤限制")

            # 8. 计算最终仓位
            adjusted_position_size = base_position_size * risk_multiplier
            adjusted_position_value = adjusted_position_size * entry_price
            position_percent = (adjusted_position_value / account_balance) * 100

            # 9. 确保不超过最大仓位限制
            if position_percent > self.max_position_size:
                scale_factor = self.max_position_size / position_percent
                adjusted_position_size *= scale_factor
                adjusted_position_value *= scale_factor
                position_percent = self.max_position_size
                risk_factors.append(f"仓位上限限制 ({self.max_position_size}%)")

            # 10. 记录风险评估
            risk_assessment = {
                'position_size': adjusted_position_size,
                'position_value': adjusted_position_value,
                'position_percent': position_percent,
                'risk_amount': risk_amount * risk_multiplier,
                'risk_multiplier': risk_multiplier,
                'risk_factors': risk_factors,
                'stop_loss_distance': stop_loss_distance * 100,
                'timestamp': datetime.now()
            }

            self.risk_history.append(risk_assessment)

            # 打印风险评估结果
            print_colored("=" * 50, Colors.BLUE)
            print_colored("📊 市场微观结构风险评估", Colors.BLUE + Colors.BOLD)
            print_colored(f"基础仓位: {base_position_size:.6f}", Colors.INFO)
            print_colored(f"风险乘数: {risk_multiplier:.2f}", Colors.INFO)
            print_colored(f"调整后仓位: {adjusted_position_size:.6f}", Colors.INFO)
            print_colored(f"仓位占比: {position_percent:.2f}%", Colors.INFO)

            if risk_factors:
                print_colored("风险因素:", Colors.WARNING)
                for factor in risk_factors:
                    print_colored(f"  - {factor}", Colors.WARNING)

            return risk_assessment

        except Exception as e:
            self.logger.error(f"仓位计算错误: {e}")
            return {
                'position_size': 0,
                'position_value': 0,
                'position_percent': 0,
                'risk_amount': 0,
                'error': str(e)
            }

    def _is_stop_hunt_zone(self, price: float, market_analysis: Dict) -> bool:
        """
        检测是否处于止损猎杀区域

        基于多空比推测止损密集区，识别潜在的止损猎杀行为
        """
        try:
            # 检查是否有止损密集区数据
            if 'stop_loss_clusters' in market_analysis:
                clusters = market_analysis['stop_loss_clusters']
                for cluster in clusters:
                    cluster_price = cluster['price']
                    cluster_strength = cluster['strength']

                    # 如果当前价格接近止损密集区（±0.5%）
                    if abs(price - cluster_price) / price < 0.005:
                        if cluster_strength > 0.7:  # 强止损密集区
                            return True

            # 检查是否有插针行为
            if 'recent_wicks' in market_analysis:
                wicks = market_analysis['recent_wicks']
                for wick in wicks[-3:]:  # 检查最近3根K线
                    if wick['ratio'] > 2.0:  # 影线是实体的2倍以上
                        return True

            return False

        except Exception as e:
            self.logger.error(f"止损猎杀检测错误: {e}")
            return False

    def update_daily_stats(self, profit_loss: float, account_balance: float):
        """更新日统计数据"""
        # 检查是否需要重置日统计
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_loss = 0.0
            self.daily_trades = []
            self.last_reset_date = current_date

        # 更新日亏损
        if profit_loss < 0:
            self.daily_loss += abs(profit_loss) / account_balance * 100

        # 更新峰值和回撤
        if account_balance > self.peak_balance:
            self.peak_balance = account_balance

        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - account_balance) / self.peak_balance * 100

        # 记录交易
        self.daily_trades.append({
            'time': datetime.now(),
            'profit_loss': profit_loss,
            'balance': account_balance
        })

    def can_open_position(self) -> Tuple[bool, str]:
        """
        检查是否可以开新仓位

        返回:
            (是否可以开仓, 原因说明)
        """
        # 检查日亏损限制
        if self.daily_loss >= self.max_daily_loss:
            return False, f"已达到日亏损限制 ({self.daily_loss:.2f}% >= {self.max_daily_loss}%)"

        # 检查最大回撤限制
        if self.current_drawdown >= self.max_drawdown:
            return False, f"已达到最大回撤限制 ({self.current_drawdown:.2f}% >= {self.max_drawdown}%)"

        # 检查日交易次数（可选）
        max_daily_trades = 20  # 可配置
        if len(self.daily_trades) >= max_daily_trades:
            return False, f"已达到日交易次数限制 ({len(self.daily_trades)} >= {max_daily_trades})"

        return True, "风险参数正常"

    def calculate_dynamic_stop_loss(self,
                                    entry_price: float,
                                    side: str,
                                    market_analysis: Dict[str, Any]) -> float:
        """
        基于市场微观结构计算动态止损

        避开止损猎杀区域，考虑市场操纵行为
        """
        # 基础止损百分比
        base_stop_loss_pct = 0.008  # 0.8%

        # 根据市场条件调整
        if 'volatility' in market_analysis:
            volatility = market_analysis['volatility']
            if volatility > 0.02:  # 高波动
                base_stop_loss_pct *= 1.5
            elif volatility < 0.005:  # 低波动
                base_stop_loss_pct *= 0.8

        # 避开止损猎杀区
        if 'stop_loss_clusters' in market_analysis:
            clusters = market_analysis['stop_loss_clusters']

            # 找到最近的止损密集区
            nearest_cluster = None
            min_distance = float('inf')

            for cluster in clusters:
                distance = abs(entry_price - cluster['price']) / entry_price
                if distance < min_distance and distance < 0.02:  # 2%范围内
                    min_distance = distance
                    nearest_cluster = cluster

            if nearest_cluster:
                # 将止损设置在密集区之外
                cluster_price = nearest_cluster['price']
                if side == "BUY":
                    # 做多时，止损设在密集区下方
                    stop_loss = min(
                        entry_price * (1 - base_stop_loss_pct),
                        cluster_price * 0.998  # 密集区下方0.2%
                    )
                else:
                    # 做空时，止损设在密集区上方
                    stop_loss = max(
                        entry_price * (1 + base_stop_loss_pct),
                        cluster_price * 1.002  # 密集区上方0.2%
                    )

                print_colored(f"⚠️ 检测到止损密集区 @ {cluster_price:.6f}, 调整止损位置", Colors.WARNING)
            else:
                # 正常计算止损
                if side == "BUY":
                    stop_loss = entry_price * (1 - base_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + base_stop_loss_pct)
        else:
            # 正常计算止损
            if side == "BUY":
                stop_loss = entry_price * (1 - base_stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + base_stop_loss_pct)

        return stop_loss

    def evaluate_exit_conditions(self,
                                 position: Dict[str, Any],
                                 current_price: float,
                                 market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估退出条件，包括止盈、止损和基于市场微观结构的退出信号
        """
        exit_signal = {
            'should_exit': False,
            'exit_type': None,
            'reason': '',
            'urgency': 'normal'  # normal, high, immediate
        }

        entry_price = position['entry_price']
        side = position['side']
        current_profit_pct = ((current_price - entry_price) / entry_price) if side == "BUY" else (
                    (entry_price - current_price) / entry_price)

        # 1. 检查止损条件
        if 'stop_loss' in position:
            if (side == "BUY" and current_price <= position['stop_loss']) or \
                    (side == "SELL" and current_price >= position['stop_loss']):
                exit_signal['should_exit'] = True
                exit_signal['exit_type'] = 'stop_loss'
                exit_signal['reason'] = '达到止损位'
                exit_signal['urgency'] = 'immediate'
                return exit_signal

        # 2. 检查操纵信号
        if 'manipulation_score' in market_analysis:
            if market_analysis['manipulation_score'] > 0.8:
                if current_profit_pct > 0.005:  # 有小幅盈利就跑
                    exit_signal['should_exit'] = True
                    exit_signal['exit_type'] = 'manipulation_detected'
                    exit_signal['reason'] = '检测到严重市场操纵'
                    exit_signal['urgency'] = 'high'
                    return exit_signal

        # 3. 检查订单流毒性
        if 'order_flow_toxicity' in market_analysis:
            toxicity = market_analysis['order_flow_toxicity']
            if toxicity > 0.4 and current_profit_pct > 0:
                exit_signal['should_exit'] = True
                exit_signal['exit_type'] = 'toxic_flow'
                exit_signal['reason'] = f'订单流毒性过高 (VPIN: {toxicity:.2f})'
                exit_signal['urgency'] = 'high'
                return exit_signal

        # 4. 检查聪明钱反转
        if 'smart_money_reversal' in market_analysis:
            if market_analysis['smart_money_reversal']:
                exit_signal['should_exit'] = True
                exit_signal['exit_type'] = 'smart_money_reversal'
                exit_signal['reason'] = '聪明钱出现反转信号'
                exit_signal['urgency'] = 'high'
                return exit_signal

        # 5. 动态止盈（基于市场结构）
        if current_profit_pct > 0.02:  # 2%以上盈利
            # 检查是否接近阻力/支撑
            if 'near_resistance' in market_analysis and market_analysis['near_resistance']:
                exit_signal['should_exit'] = True
                exit_signal['exit_type'] = 'resistance_take_profit'
                exit_signal['reason'] = '接近关键阻力位'
                exit_signal['urgency'] = 'normal'
                return exit_signal

        # 6. 时间止盈（可选）
        if 'holding_time' in position:
            holding_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_hours > 24 and current_profit_pct > 0.01:  # 持仓超过24小时且有盈利
                exit_signal['should_exit'] = True
                exit_signal['exit_type'] = 'time_exit'
                exit_signal['reason'] = f'持仓时间过长 ({holding_hours:.1f}小时)'
                exit_signal['urgency'] = 'normal'

        return exit_signal

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险管理摘要"""
        return {
            'daily_loss': self.daily_loss,
            'current_drawdown': self.current_drawdown,
            'peak_balance': self.peak_balance,
            'daily_trades_count': len(self.daily_trades),
            'can_trade': self.can_open_position()[0],
            'risk_status': self._get_risk_status(),
            'last_update': datetime.now()
        }

    def _get_risk_status(self) -> str:
        """获取当前风险状态"""
        if self.daily_loss >= self.max_daily_loss * 0.9:
            return "危险"
        elif self.daily_loss >= self.max_daily_loss * 0.7:
            return "警告"
        elif self.current_drawdown >= self.max_drawdown * 0.8:
            return "谨慎"
        else:
            return "正常"

    def record_manipulation_event(self, event: Dict[str, Any]):
        """记录操纵事件用于后续分析"""
        event['timestamp'] = datetime.now()
        self.manipulation_events.append(event)

        # 保留最近100个事件
        if len(self.manipulation_events) > 100:
            self.manipulation_events = self.manipulation_events[-100:]

    def update_stop_hunt_zones(self, symbol: str, zones: List[Dict[str, float]]):
        """更新止损猎杀区域"""
        self.stop_hunt_zones[symbol] = {
            'zones': zones,
            'updated_at': datetime.now()
        }