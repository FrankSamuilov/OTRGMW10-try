"""
enhanced_trading_main.py
增强版交易系统主文件 - 调用simple_trading_bot的基础功能
实现交易计划系统和流动性分析
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any

# 导入配置
from config import (
    API_KEY, API_SECRET, TRADE_PAIRS,
    MIN_MARGIN_BALANCE, ORDER_AMOUNT_PERCENT,
    MAX_POSITIONS, USE_GAME_THEORY
)

# 导入您的simple_trading_bot（清理评分后的版本）
from simple_trading_bot import SimpleTradingBot

# 导入现有的模块
from logger_utils import Colors, print_colored
from data_module import get_historical_data
from indicators_module import calculate_optimized_indicators

# 如果这些模块存在，导入它们
try:
    from liquidity_hunter import LiquidityHunterSystem
    from liquidity_stop_loss import LiquidityAwareStopLoss

    LIQUIDITY_MODULES_AVAILABLE = True
except ImportError:
    print_colored("⚠️ 流动性模块不可用，将使用基础功能", Colors.WARNING)
    LIQUIDITY_MODULES_AVAILABLE = False


class TradingPlanManager:
    """交易计划管理器 - 完整版本，包含所有方法"""

    def __init__(self):
        self.logger = logging.getLogger('TradingPlan')
        self.active_plans = {}
        self.executed_plans = []

    def create_plan(self, analysis: Dict, symbol: str) -> Optional[Dict]:
        """创建交易计划"""

        # 识别是否是回调交易
        is_pullback_trade = self._identify_pullback_trade(analysis)

        # 根据交易类型调整验证条件
        if is_pullback_trade:
            min_confidence = 0.35
            min_risk_reward = 1.5
        else:
            min_confidence = 0.4
            min_risk_reward = 1.2

        # 验证条件
        if not self._validate_conditions_flexible(analysis, min_confidence, min_risk_reward):
            return None

        current_price = analysis.get('current_price', 0)
        direction = analysis.get('direction', 'NEUTRAL')

        if direction == 'NEUTRAL' or current_price == 0:
            return None

        # 创建计划
        plan = {
            'symbol': symbol,
            'direction': direction,
            'trade_type': 'PULLBACK' if is_pullback_trade else 'TREND',
            'created_at': datetime.now(),
            'valid_until': datetime.now() + timedelta(hours=1 if is_pullback_trade else 2),

            # 入场策略
            'entry': self._plan_entry_with_type(analysis, is_pullback_trade),

            # 出场策略
            'exit': self._plan_exit_with_type(analysis, is_pullback_trade),

            # 风险管理
            'risk': self._plan_risk_with_warnings(analysis, is_pullback_trade),

            # 执行参数
            'execution': {
                'max_slippage': 0.003,
                'timeout_minutes': 10,
                'retry_count': 2
            },

            # 分析数据
            'analysis': analysis,
            'confidence': analysis.get('confidence', 0.5),
            'reasoning': analysis.get('reasoning', []),

            # 风险警告
            'warnings': self._generate_warnings(analysis, is_pullback_trade)
        }

        return plan

    def _validate_conditions(self, analysis: Dict) -> bool:
        """基础验证（保留兼容性）"""
        if analysis.get('direction') not in ['LONG', 'SHORT']:
            return False
        if analysis.get('confidence', 0) < 0.6:
            return False
        if analysis.get('risk_reward_ratio', 0) < 1.5:
            return False
        return True

    def _validate_conditions_flexible(self, analysis: Dict, min_confidence: float, min_risk_reward: float) -> bool:
        """灵活的条件验证"""

        if analysis.get('direction') not in ['LONG', 'SHORT']:
            print_colored("    ❌ 无明确方向", Colors.WARNING)
            return False

        confidence = analysis.get('confidence', 0)
        if confidence < min_confidence:
            print_colored(f"    ⚠️ 置信度偏低: {confidence:.1%} (要求≥{min_confidence:.1%})", Colors.YELLOW)
            if confidence < min_confidence * 0.8:
                print_colored(f"    ❌ 置信度过低，取消交易", Colors.RED)
                return False

        risk_reward = analysis.get('risk_reward_ratio', 0)
        if risk_reward < min_risk_reward:
            print_colored(f"    ⚠️ 风险回报比: {risk_reward:.1f} (要求≥{min_risk_reward:.1f})", Colors.YELLOW)
            if risk_reward < 1.0:
                print_colored(f"    ❌ 风险回报比过低", Colors.RED)
                return False

        print_colored(f"    ✅ 条件满足 - 方向:{analysis['direction']}, 置信度:{confidence:.1%}, RR:{risk_reward:.1f}",
                      Colors.SUCCESS)
        return True

    def _identify_pullback_trade(self, analysis: Dict) -> bool:
        """识别是否是超买/超卖回调交易"""

        indicators = analysis.get('indicators', {})
        rsi = indicators.get('RSI', 50)

        # 如果indicators为空，返回False
        if not indicators:
            return False

        # 超买做空或超卖做多
        if analysis.get('direction') == 'SHORT' and rsi > 75:
            print_colored("    ⚠️ 识别为：超买回调交易（高风险）", Colors.YELLOW)
            return True
        elif analysis.get('direction') == 'LONG' and rsi < 25:
            print_colored("    ⚠️ 识别为：超卖反弹交易（高风险）", Colors.YELLOW)
            return True

        return False

    def _plan_entry_with_type(self, analysis: Dict, is_pullback: bool) -> Dict:
        """根据交易类型规划入场策略"""

        current_price = analysis['current_price']
        direction = analysis['direction']
        atr = analysis.get('atr', current_price * 0.01)

        entry = {
            'primary': {
                'price': current_price,
                'size_percent': 20 if is_pullback else 30,
                'type': 'LIMIT'
            },
            'scaling': []
        }

        # 基于流动性区域调整
        if 'liquidity_zones' in analysis and analysis['liquidity_zones']:
            zone = analysis['liquidity_zones'][0]

            if direction == 'LONG' and zone['type'] == 'support':
                entry['primary']['price'] = zone['price'] * 1.001
                entry['scaling'] = [
                    {
                        'price': zone['price'] * 0.995,
                        'size_percent': 20,
                        'condition': '触及强支撑'
                    }
                ]
            elif direction == 'SHORT' and zone['type'] == 'resistance':
                entry['primary']['price'] = zone['price'] * 0.999
                entry['scaling'] = [
                    {
                        'price': zone['price'] * 1.005,
                        'size_percent': 20,
                        'condition': '触及强阻力'
                    }
                ]

        return entry

    def _plan_exit_with_type(self, analysis: Dict, is_pullback: bool) -> Dict:
        """根据交易类型规划出场策略"""

        current_price = analysis['current_price']
        direction = analysis['direction']
        atr = analysis.get('atr', current_price * 0.01)

        if is_pullback:
            print_colored("    📌 使用回调交易止损策略（更紧）", Colors.INFO)

            if direction == 'SHORT':
                stop_loss = current_price + (atr * 0.5)
                take_profits = [
                    current_price - (atr * 0.5),
                    current_price - (atr * 1.0),
                    current_price - (atr * 1.5),
                ]
            else:
                stop_loss = current_price - (atr * 0.5)
                take_profits = [
                    current_price + (atr * 0.5),
                    current_price + (atr * 1.0),
                    current_price + (atr * 1.5),
                ]

            return {
                'stop_loss': {
                    'initial': stop_loss,
                    'trailing': True,
                    'trail_percent': 1.0,
                    'trail_activation': 0.5
                },
                'take_profit': {
                    'targets': take_profits,
                    'partial_exits': [50, 30, 20]
                }
            }
        else:
            print_colored("    📌 使用趋势交易止损策略（标准）", Colors.INFO)

            if direction == 'LONG':
                stop_loss = current_price - (atr * 1.5)
                take_profits = [
                    current_price + (atr * 1),
                    current_price + (atr * 2),
                    current_price + (atr * 3)
                ]
            else:
                stop_loss = current_price + (atr * 1.5)
                take_profits = [
                    current_price - (atr * 1),
                    current_price - (atr * 2),
                    current_price - (atr * 3)
                ]

            return {
                'stop_loss': {
                    'initial': stop_loss,
                    'trailing': True,
                    'trail_percent': 2.0,
                    'trail_activation': 1.0
                },
                'take_profit': {
                    'targets': take_profits,
                    'partial_exits': [30, 30, 40]
                }
            }

    def _plan_risk_with_warnings(self, analysis: Dict, is_pullback: bool) -> Dict:
        """规划风险管理"""

        entry_price = analysis['current_price']

        # 基础风险计算
        risk_percent = 1.0 if is_pullback else 2.0

        return {
            'max_risk_percent': risk_percent,
            'risk_reward_ratio': analysis.get('risk_reward_ratio', 2.0),
            'position_size_multiplier': 0.5 if is_pullback else 1.0,
            'strict_stop': is_pullback
        }

    def _generate_warnings(self, analysis: Dict, is_pullback: bool) -> List[str]:
        """生成风险警告"""

        warnings = []

        if is_pullback:
            indicators = analysis.get('indicators', {})
            rsi = indicators.get('RSI', 50)

            if analysis.get('direction') == 'SHORT' and rsi > 80:
                warnings.append("⚠️ 极度超买回调交易 - 如果突破继续上涨立即止损！")
                warnings.append("⚠️ 建议仓位减半，严格止损")
                warnings.append("⚠️ RSI > 80 在强势市场可能继续上涨")
            elif analysis.get('direction') == 'LONG' and rsi < 20:
                warnings.append("⚠️ 极度超卖反弹交易 - 如果跌破支撑立即止损！")
                warnings.append("⚠️ 建议仓位减半，严格止损")
                warnings.append("⚠️ RSI < 20 在弱势市场可能继续下跌")

        # 置信度警告
        confidence = analysis.get('confidence', 0)
        if confidence < 0.5:
            warnings.append(f"⚠️ 置信度偏低 ({confidence:.1%}) - 建议观望或减小仓位")

        # 风险回报比警告
        risk_reward = analysis.get('risk_reward_ratio', 0)
        if risk_reward < 1.5:
            warnings.append(f"⚠️ 风险回报比偏低 ({risk_reward:.1f}) - 注意风险控制")

        return warnings

    def _plan_entry(self, analysis: Dict) -> Dict:
        """基础入场规划（兼容旧代码）"""
        return self._plan_entry_with_type(analysis, False)

    def _plan_exit(self, analysis: Dict) -> Dict:
        """基础出场规划（兼容旧代码）"""
        return self._plan_exit_with_type(analysis, False)

    def _plan_risk(self, analysis: Dict) -> Dict:
        """基础风险规划（兼容旧代码）"""
        return self._plan_risk_with_warnings(analysis, False)

    def _calculate_entry_price(self, analysis: Dict) -> float:
        """计算入场价格（兼容旧代码）"""
        return analysis.get('current_price', 0)

    def _calculate_stop_loss(self, analysis: Dict) -> float:
        """计算止损价格（兼容旧代码）"""
        entry_price = self._calculate_entry_price(analysis)
        direction = analysis.get('direction')
        atr = analysis.get('atr', entry_price * 0.02)

        if direction == 'LONG':
            return entry_price - (atr * 1.5)
        else:
            return entry_price + (atr * 1.5)

    def _calculate_profit_targets(self, analysis: Dict) -> List[float]:
        """计算止盈目标（兼容旧代码）"""
        entry_price = self._calculate_entry_price(analysis)
        direction = analysis.get('direction')
        atr = analysis.get('atr', entry_price * 0.02)

        if direction == 'LONG':
            return [
                entry_price + (atr * 1),
                entry_price + (atr * 2),
                entry_price + (atr * 3)
            ]
        else:
            return [
                entry_price - (atr * 1),
                entry_price - (atr * 2),
                entry_price - (atr * 3)
            ]

    def _calculate_risk_reward(self, analysis: Dict) -> float:
        """计算风险回报比（兼容旧代码）"""
        entry = self._calculate_entry_price(analysis)
        stop = self._calculate_stop_loss(analysis)
        targets = self._calculate_profit_targets(analysis)

        risk = abs(entry - stop)
        reward = abs(targets[1] - entry) if targets else risk * 2

        if risk > 0:
            return reward / risk
        return 0



class EnhancedGameAnalyzer:
    """增强的博弈分析器"""

    def __init__(self):
        self.logger = logging.getLogger('GameAnalyzer')
        self.order_book_history = deque(maxlen=10)

    def analyze(self, df: pd.DataFrame, order_book: Dict = None,
                liquidity_data: Dict = None) -> Dict:
        """综合分析，返回交易方向和置信度"""

        analysis = {
            'direction': 'NEUTRAL',
            'confidence': 0,
            'current_price': 0,
            'atr': 0,
            'liquidity_zones': [],
            'stop_hunt_zones': [],
            'risk_reward_ratio': 0,
            'reasoning': [],
            'indicators': {}  # 添加指标存储
        }

        if df is None or df.empty:
            print_colored("  ⚠️ DataFrame为空", Colors.WARNING)
            return analysis

        try:
            analysis['current_price'] = df['close'].iloc[-1]
            analysis['atr'] = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0

            # 存储关键指标值
            if 'RSI' in df.columns:
                analysis['indicators']['RSI'] = df['RSI'].iloc[-1]
            if 'ADX' in df.columns:
                analysis['indicators']['ADX'] = df['ADX'].iloc[-1]
            if 'CCI' in df.columns:
                analysis['indicators']['CCI'] = df['CCI'].iloc[-1]

            # 1. 价格行为分析
            price_signal = self._analyze_price_action(df)
            print_colored(f"  📊 价格信号: {price_signal}", Colors.INFO)

            # 2. 流动性分析
            if liquidity_data:
                analysis['liquidity_zones'] = liquidity_data.get('zones', [])
                analysis['stop_hunt_zones'] = liquidity_data.get('hunt_zones', [])
                liquidity_signal = self._analyze_liquidity_signal(liquidity_data, analysis['current_price'])
            else:
                liquidity_signal = {'direction': 'NEUTRAL', 'strength': 0}
            print_colored(f"  💧 流动性信号: {liquidity_signal}", Colors.INFO)

            # 3. 订单簿分析
            if order_book:
                self.order_book_history.append(order_book)
                ob_signal = self._analyze_smoothed_orderbook()
            else:
                ob_signal = {'direction': 'NEUTRAL', 'strength': 0}
            print_colored(f"  📖 订单簿信号: {ob_signal}", Colors.INFO)

            # 4. 技术指标分析
            tech_signal = self._analyze_technical_indicators(df)
            print_colored(f"  📈 技术信号: {tech_signal}", Colors.INFO)

            # ===== 修复：在这里定义 signals 变量 =====
            # 根据RSI极端值调整权重
            rsi = analysis['indicators'].get('RSI', 50)

            if rsi > 80 or rsi < 20:
                # RSI极端值时的权重
                print_colored(f"    ⚠️ RSI极端值 ({rsi:.1f})，调整权重", Colors.YELLOW)
                signals = [
                    (price_signal, 0.2, '价格行为'),  # 降低价格权重
                    (liquidity_signal, 0.25, '流动性'),  # 降低流动性权重
                    (tech_signal, 0.45, '技术指标'),  # 提高技术指标权重
                    (ob_signal, 0.1, '订单簿')  # 降低订单簿权重
                ]
            else:
                # 正常权重
                signals = [
                    (price_signal, 0.3, '价格行为'),
                    (liquidity_signal, 0.3, '流动性'),
                    (tech_signal, 0.25, '技术指标'),
                    (ob_signal, 0.15, '订单簿')
                ]

            # 计算综合方向和置信度
            long_score = 0
            short_score = 0
            signal_count = 0

            for signal, weight, name in signals:
                if signal['direction'] != 'NEUTRAL':
                    signal_count += 1

                if signal['direction'] == 'LONG':
                    long_score += signal['strength'] * weight
                    if signal['strength'] > 0.3:
                        analysis['reasoning'].append(f"{name}看多")
                elif signal['direction'] == 'SHORT':
                    short_score += signal['strength'] * weight
                    if signal['strength'] > 0.3:
                        analysis['reasoning'].append(f"{name}看空")

            print_colored(f"  📊 综合评分 - 多头: {long_score:.2f}, 空头: {short_score:.2f}", Colors.CYAN)

            # 改进的置信度计算
            if long_score > 0.25 and long_score > short_score:
                analysis['direction'] = 'LONG'
                # 提升置信度计算
                base_confidence = long_score

                # 根据信号一致性加成
                if signal_count >= 3:
                    base_confidence *= 1.3

                # 根据信号强度差异加成
                if long_score > short_score * 2:
                    base_confidence *= 1.2

                analysis['confidence'] = min(base_confidence, 0.95)

            elif short_score > 0.25 and short_score > long_score:
                analysis['direction'] = 'SHORT'
                base_confidence = short_score

                if signal_count >= 3:
                    base_confidence *= 1.3

                if short_score > long_score * 2:
                    base_confidence *= 1.2

                analysis['confidence'] = min(base_confidence, 0.95)

            # 为超买超卖情况增加置信度
            if analysis['direction'] != 'NEUTRAL':
                rsi = analysis['indicators'].get('RSI', 50)
                if (analysis['direction'] == 'SHORT' and rsi > 75) or \
                        (analysis['direction'] == 'LONG' and rsi < 25):
                    analysis['confidence'] = min(analysis['confidence'] * 1.2, 0.85)
                    print_colored(f"    📈 超买/超卖加成，置信度提升至: {analysis['confidence']:.1%}", Colors.INFO)

            # 计算风险回报比
            if analysis['direction'] != 'NEUTRAL':
                analysis['risk_reward_ratio'] = self._calculate_risk_reward(
                    df, analysis['direction'], analysis.get('liquidity_zones', [])
                )

        except Exception as e:
            self.logger.error(f"分析错误: {e}")
            print_colored(f"  ❌ 分析错误: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()

        return analysis

    def explain_decision(self, analysis: Dict, df: pd.DataFrame):
        """解释交易决策"""

        if analysis['direction'] == 'SHORT' and 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi > 80:
                print_colored("\n📌 决策解释:", Colors.CYAN)
                print_colored("  虽然动量指标显示上升趋势，但是：", Colors.INFO)
                print_colored(f"  • RSI {rsi:.1f} 极度超买，短期回调概率高", Colors.RED)
                print_colored("  • 建议做空捕捉回调利润", Colors.RED)
                print_colored("  • 或等待回调后再做多", Colors.YELLOW)

        elif analysis['direction'] == 'LONG' and 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                print_colored("\n📌 决策解释:", Colors.CYAN)
                print_colored("  虽然动量指标显示下降趋势，但是：", Colors.INFO)
                print_colored(f"  • RSI {rsi:.1f} 极度超卖，短期反弹概率高", Colors.GREEN)
                print_colored("  • 建议做多捕捉反弹利润", Colors.GREEN)

    def interpret_overbought_in_trend(self, df: pd.DataFrame) -> str:
        """
        解释超买但趋势向上的情况
        """
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0

        if rsi > 70 and adx > 40:
            # 强趋势中的超买
            print_colored("    ⚠️ 强趋势中的超买状态:", Colors.YELLOW)
            print_colored("      • 短期: 可能回调（SHORT信号）", Colors.RED)
            print_colored("      • 中期: 趋势可能继续", Colors.GREEN)
            print_colored("      • 策略: 等待回调或突破后再入场", Colors.INFO)

            # 在强趋势中，超买可能持续很久
            # 但短期内仍然有回调风险，所以SHORT是合理的
            return "SHORT_FOR_PULLBACK"

        return "NORMAL"

    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """分析价格行为 - 修复版"""
        try:
            price = df['close'].iloc[-1]
            print_colored(f"    当前价格: {price:.4f}", Colors.INFO)

            # 检查多种EMA组合
            ema_checks = []

            # 检查 EMA20 和 EMA52（您的数据有这两个）
            if 'EMA20' in df.columns and 'EMA52' in df.columns:
                ema20 = df['EMA20'].iloc[-1]
                ema52 = df['EMA52'].iloc[-1]

                print_colored(f"    EMA20: {ema20:.4f}, EMA52: {ema52:.4f}", Colors.GRAY)

                # 判断趋势
                if price > ema20 and ema20 > ema52:
                    trend_strength = min(((price - ema52) / ema52) * 20, 1.0)
                    print_colored(f"    ✅ 上升趋势 (价格>EMA20>EMA52), 强度: {trend_strength:.2f}", Colors.GREEN)
                    return {'direction': 'LONG', 'strength': trend_strength}
                elif price < ema20 and ema20 < ema52:
                    trend_strength = min(((ema52 - price) / ema52) * 20, 1.0)
                    print_colored(f"    ✅ 下降趋势 (价格<EMA20<EMA52), 强度: {trend_strength:.2f}", Colors.RED)
                    return {'direction': 'SHORT', 'strength': trend_strength}
                else:
                    # 检查价格相对于短期均线的位置
                    if price > ema20 * 1.01:  # 价格明显高于EMA20
                        print_colored(f"    价格高于EMA20", Colors.INFO)
                        return {'direction': 'LONG', 'strength': 0.4}
                    elif price < ema20 * 0.99:  # 价格明显低于EMA20
                        print_colored(f"    价格低于EMA20", Colors.INFO)
                        return {'direction': 'SHORT', 'strength': 0.4}

            # 使用EMA5作为备用
            if 'EMA5' in df.columns:
                ema5 = df['EMA5'].iloc[-1]
                print_colored(f"    EMA5: {ema5:.4f}", Colors.GRAY)

                if price > ema5 * 1.005:
                    return {'direction': 'LONG', 'strength': 0.3}
                elif price < ema5 * 0.995:
                    return {'direction': 'SHORT', 'strength': 0.3}

            print_colored(f"    ⚠️ 无法确定趋势方向", Colors.WARNING)
            return {'direction': 'NEUTRAL', 'strength': 0}

        except Exception as e:
            print_colored(f"    ❌ 价格行为分析错误: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()
            return {'direction': 'NEUTRAL', 'strength': 0}

    def _analyze_liquidity_signal(self, liquidity_data: Dict, current_price: float) -> Dict:
        """分析流动性信号"""
        try:
            zones = liquidity_data.get('zones', [])
            if not zones:
                return {'direction': 'NEUTRAL', 'strength': 0}

            nearest_zone = zones[0]
            distance = abs(nearest_zone['price'] - current_price) / current_price

            # 接近支撑位
            if nearest_zone['type'] == 'support' and current_price > nearest_zone['price']:
                if distance < 0.005:  # 0.5%以内
                    return {'direction': 'LONG', 'strength': 0.8}
                elif distance < 0.01:  # 1%以内
                    return {'direction': 'LONG', 'strength': 0.6}

            # 接近阻力位
            elif nearest_zone['type'] == 'resistance' and current_price < nearest_zone['price']:
                if distance < 0.005:
                    return {'direction': 'SHORT', 'strength': 0.8}
                elif distance < 0.01:
                    return {'direction': 'SHORT', 'strength': 0.6}

            return {'direction': 'NEUTRAL', 'strength': 0}

        except Exception as e:
            self.logger.error(f"流动性信号分析错误: {e}")
            return {'direction': 'NEUTRAL', 'strength': 0}

    def _analyze_smoothed_orderbook(self) -> Dict:
        """分析平滑后的订单簿"""
        if len(self.order_book_history) < 3:
            return {'direction': 'NEUTRAL', 'strength': 0}

        try:
            ratios = []

            for ob in self.order_book_history:
                bid_volume = sum(ob.get('bid_sizes', [])[:10])
                ask_volume = sum(ob.get('ask_sizes', [])[:10])

                if bid_volume + ask_volume > 0:
                    ratio = bid_volume / (bid_volume + ask_volume)
                    ratios.append(ratio)

            if not ratios:
                return {'direction': 'NEUTRAL', 'strength': 0}

            # 加权平均
            weights = np.linspace(0.5, 1.0, len(ratios))
            weights = weights / weights.sum()
            weighted_ratio = np.average(ratios, weights=weights)

            # 判断方向
            if weighted_ratio > 0.6:
                return {'direction': 'LONG', 'strength': (weighted_ratio - 0.5) * 2}
            elif weighted_ratio < 0.4:
                return {'direction': 'SHORT', 'strength': (0.5 - weighted_ratio) * 2}

            return {'direction': 'NEUTRAL', 'strength': abs(weighted_ratio - 0.5)}

        except Exception as e:
            self.logger.error(f"订单簿分析错误: {e}")
            return {'direction': 'NEUTRAL', 'strength': 0}

    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """分析技术指标 - 修复版"""
        try:
            signals = []

            # RSI - 您的数据显示 82.40（强烈超买）
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                print_colored(f"    RSI: {rsi:.2f}", Colors.INFO)

                if not pd.isna(rsi):
                    if rsi > 80:  # 强烈超买
                        signals.append(('SHORT', 0.9))
                        print_colored(f"      → 强烈超买信号", Colors.RED)
                    elif rsi > 70:  # 超买
                        signals.append(('SHORT', 0.7))
                    elif rsi < 20:  # 强烈超卖
                        signals.append(('LONG', 0.9))
                    elif rsi < 30:  # 超卖
                        signals.append(('LONG', 0.7))
                    elif rsi > 60:  # 轻微超买
                        signals.append(('SHORT', 0.4))
                    elif rsi < 40:  # 轻微超卖
                        signals.append(('LONG', 0.4))

            # CCI - 您的数据显示 100.32（超买）
            if 'CCI' in df.columns:
                cci = df['CCI'].iloc[-1]
                print_colored(f"    CCI: {cci:.2f}", Colors.INFO)

                if not pd.isna(cci):
                    if cci > 100:  # 超买
                        signals.append(('SHORT', 0.7))
                        print_colored(f"      → CCI超买信号", Colors.RED)
                    elif cci < -100:  # 超卖
                        signals.append(('LONG', 0.7))

            # Williams %R - 您的数据显示 -11.29（超买）
            if 'Williams_%R' in df.columns:
                williams = df['Williams_%R'].iloc[-1]
                print_colored(f"    Williams %R: {williams:.2f}", Colors.INFO)

                if not pd.isna(williams):
                    if williams > -20:  # 超买
                        signals.append(('SHORT', 0.7))
                        print_colored(f"      → Williams超买信号", Colors.RED)
                    elif williams < -80:  # 超卖
                        signals.append(('LONG', 0.7))

            # MACD
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                signal = df['MACD_signal'].iloc[-1]

                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal and macd > 0:
                        signals.append(('LONG', 0.5))
                    elif macd < signal and macd < 0:
                        signals.append(('SHORT', 0.5))

            # ADX - 您的数据显示 43.24（强趋势）
            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[-1]
                print_colored(f"    ADX: {adx:.2f} (强趋势)", Colors.INFO)

                # ADX只表示趋势强度，需要结合其他指标判断方向
                if not pd.isna(adx) and adx > 25:
                    # 增强现有信号
                    if signals:
                        print_colored(f"      → ADX增强信号", Colors.CYAN)

            # 布林带位置 - 0.81（接近上轨）
            if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                price = df['close'].iloc[-1]
                upper = df['BB_Upper'].iloc[-1]
                lower = df['BB_Lower'].iloc[-1]

                if upper > lower:
                    bb_position = (price - lower) / (upper - lower)
                    print_colored(f"    BB位置: {bb_position:.2f}", Colors.INFO)

                    if bb_position > 0.8:  # 接近上轨
                        signals.append(('SHORT', 0.6))
                        print_colored(f"      → 接近布林带上轨", Colors.RED)
                    elif bb_position < 0.2:  # 接近下轨
                        signals.append(('LONG', 0.6))

            # 综合技术信号
            if not signals:
                print_colored("    ⚠️ 没有技术信号", Colors.WARNING)
                return {'direction': 'NEUTRAL', 'strength': 0}

            # 统计信号
            long_signals = [(s, w) for s, w in signals if s == 'LONG']
            short_signals = [(s, w) for s, w in signals if s == 'SHORT']

            print_colored(f"    📊 信号统计 - 多头: {len(long_signals)}, 空头: {len(short_signals)}", Colors.CYAN)

            if len(short_signals) > len(long_signals):
                strength = sum(w for _, w in short_signals) / max(len(short_signals), 1)
                print_colored(f"    → 技术面看空，强度: {strength:.2f}", Colors.RED)
                return {'direction': 'SHORT', 'strength': min(strength, 1.0)}
            elif len(long_signals) > len(short_signals):
                strength = sum(w for _, w in long_signals) / max(len(long_signals), 1)
                print_colored(f"    → 技术面看多，强度: {strength:.2f}", Colors.GREEN)
                return {'direction': 'LONG', 'strength': min(strength, 1.0)}
            else:
                # 如果信号数量相同，比较强度
                long_strength = sum(w for _, w in long_signals) if long_signals else 0
                short_strength = sum(w for _, w in short_signals) if short_signals else 0

                if short_strength > long_strength:
                    return {'direction': 'SHORT', 'strength': short_strength / max(len(short_signals), 1)}
                elif long_strength > short_strength:
                    return {'direction': 'LONG', 'strength': long_strength / max(len(long_signals), 1)}
                else:
                    return {'direction': 'NEUTRAL', 'strength': 0.3}

        except Exception as e:
            print_colored(f"    ❌ 技术指标分析错误: {e}", Colors.ERROR)
            return {'direction': 'NEUTRAL', 'strength': 0}

    def _calculate_risk_reward(self, df: pd.DataFrame, direction: str, liquidity_zones: List) -> float:
        """计算风险回报比"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02

            if direction == 'LONG':
                stop = current_price - (atr * 1.5)
                target = current_price + (atr * 3)
            else:
                stop = current_price + (atr * 1.5)
                target = current_price - (atr * 3)

            risk = abs(current_price - stop)
            reward = abs(target - current_price)

            if risk > 0:
                return reward / risk

            return 0

        except Exception as e:
            self.logger.error(f"计算风险回报比错误: {e}")
            return 0


class LiquidityAnalyzer:
    """流动性分析器 - 简化版"""

    def __init__(self):
        self.logger = logging.getLogger('Liquidity')

    def analyze(self, df: pd.DataFrame) -> Dict:
        """分析流动性景观"""

        result = {
            'zones': [],
            'hunt_zones': []
        }

        if df is None or len(df) < 50:
            return result

        try:
            current_price = df['close'].iloc[-1]

            # 找出支撑和阻力
            # 使用20周期的高低点
            for i in range(len(df) - 40, len(df) - 5, 5):
                window = df.iloc[i:i + 20]

                # 局部高点（阻力）
                high_point = window['high'].max()
                if abs(high_point - current_price) / current_price < 0.03:  # 3%范围内
                    result['zones'].append({
                        'price': high_point,
                        'type': 'resistance',
                        'strength': 0.7
                    })

                # 局部低点（支撑）
                low_point = window['low'].min()
                if abs(low_point - current_price) / current_price < 0.03:
                    result['zones'].append({
                        'price': low_point,
                        'type': 'support',
                        'strength': 0.7
                    })

            # 去重和排序
            result['zones'] = self._consolidate_zones(result['zones'], current_price)

            # 识别止损猎杀区域
            for zone in result['zones'][:3]:
                distance = abs(zone['price'] - current_price) / current_price
                if 0.002 < distance < 0.015:  # 0.2%到1.5%
                    result['hunt_zones'].append({
                        'target_price': zone['price'],
                        'type': zone['type'],
                        'probability': 0.6
                    })

        except Exception as e:
            self.logger.error(f"流动性分析错误: {e}")

        return result

    def _consolidate_zones(self, zones: List[Dict], current_price: float) -> List[Dict]:
        """整合相近的区域"""
        if not zones:
            return zones

        # 合并相近的区域
        consolidated = []
        for zone in zones:
            merged = False
            for existing in consolidated:
                if abs(zone['price'] - existing['price']) / existing['price'] < 0.005:
                    # 保留更强的
                    if zone['strength'] > existing['strength']:
                        existing['strength'] = zone['strength']
                    merged = True
                    break

            if not merged:
                consolidated.append(zone)

        # 按距离排序
        consolidated.sort(key=lambda x: abs(x['price'] - current_price))

        return consolidated[:5]

    def _generate_warnings(self, analysis: Dict, is_pullback: bool) -> List[str]:
        """生成风险警告"""
        warnings = []

        if is_pullback:
            indicators = analysis.get('indicators', {})
            rsi = indicators.get('RSI', 50)

            if analysis['direction'] == 'SHORT' and rsi > 80:
                warnings.append("⚠️ 极度超买回调交易 - 如果突破继续上涨立即止损！")
                warnings.append("⚠️ 建议仓位减半，严格止损")
            elif analysis['direction'] == 'LONG' and rsi < 20:
                warnings.append("⚠️ 极度超卖反弹交易 - 如果跌破支撑立即止损！")
                warnings.append("⚠️ 建议仓位减半，严格止损")

        confidence = analysis.get('confidence', 0)
        if confidence < 0.5:
            warnings.append(f"⚠️ 置信度偏低 ({confidence:.1%}) - 建议观望或减小仓位")

        return warnings


class EnhancedTradingSystem(SimpleTradingBot):
    """
    增强版交易系统 - 继承SimpleTradingBot的基础功能
    添加交易计划和流动性分析
    """

    def __init__(self):
        """初始化增强交易系统"""

        # ==================== 1. 基础初始化 ====================
        # 如果父类有初始化，先调用
        try:
            super().__init__()
        except:
            pass

        print_colored("\n🚀 初始化增强交易系统...", Colors.CYAN)

        # ==================== 2. 客户端和日志 ====================
        from binance.client import Client
        from config import API_KEY, API_SECRET
        import logging

        self.client = Client(API_KEY, API_SECRET)
        self.logger = logging.getLogger('EnhancedTrading')

        # 测试连接
        try:
            server_time = self.client.get_server_time()
            self.logger.info(f"成功连接到Binance，服务器时间: {server_time}")
        except Exception as e:
            print_colored(f"⚠️ 连接测试失败: {e}", Colors.WARNING)

        # ==================== 3. 核心属性（必须有）====================
        self.positions = {}  # 持仓记录
        self.active_plans = {}  # 活跃的交易计划
        self.plan_history = []  # 历史计划
        self.last_analysis_time = {}  # 最后分析时间

        # 从旧代码复制的重要属性
        self.trade_cycle = 0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.is_running = False
        self.last_scan_time = 0

        # 缓存相关
        self.historical_data_cache = {}
        self.cache_ttl = 300  # 5分钟缓存

        # 信号历史
        self.signal_history = {}
        self.order_book_history = []

        # 配置
        self.config = {
            'TRADE_PAIRS': TRADE_PAIRS,
            'MAX_POSITIONS': MAX_POSITIONS,
            'MIN_MARGIN_BALANCE': MIN_MARGIN_BALANCE,
            'USE_GAME_THEORY': USE_GAME_THEORY
        }

        # ==================== 4. 新系统组件 ====================
        self.plan_manager = TradingPlanManager()
        self.game_analyzer = EnhancedGameAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()

        # ==================== 5. 流动性模块（如果可用）====================
        if LIQUIDITY_MODULES_AVAILABLE:
            try:
                self.liquidity_hunter = LiquidityHunterSystem(self.client)
                self.liquidity_stop_loss = LiquidityAwareStopLoss()
                print_colored("✅ 流动性系统初始化成功", Colors.GREEN)
            except Exception as e:
                print_colored(f"⚠️ 流动性系统初始化失败: {e}", Colors.WARNING)
                self.liquidity_hunter = None
                self.liquidity_stop_loss = None

        # ==================== 6. 博弈论组件（如果需要）====================
        self.use_game_theory = self.config.get("USE_GAME_THEORY", True)
        if self.use_game_theory:
            try:
                # 这里只初始化您实际有的组件
                print_colored("✅ 博弈论系统启用", Colors.GREEN)
            except Exception as e:
                print_colored(f"⚠️ 博弈论系统初始化失败: {e}", Colors.WARNING)

        # ==================== 7. 风险管理（简化版）====================
        self.max_positions = 5
        self.min_balance = 10  # 最小余额要求

        print_colored("✅ 增强系统初始化完成", Colors.SUCCESS)

    def get_account_balance_simple(self) -> float:
        """简单版本 - 获取所有稳定币余额"""
        try:
            account = self.client.futures_account_balance()

            # 所有可能的稳定币
            stable_coins = {
                'USDT': 0.0,
                'USDC': 0.0,
                'BUSD': 0.0,
                'FDUSD': 0.0,
                'TUSD': 0.0
            }

            # 累加所有稳定币
            for asset in account:
                if asset['asset'] in stable_coins:
                    stable_coins[asset['asset']] = float(asset['balance'])

            # 打印找到的余额
            for coin, balance in stable_coins.items():
                if balance > 0:
                    print_colored(f"  • {coin}: {balance:.2f}", Colors.GREEN)

            total = sum(stable_coins.values())

            if total == 0:
                print_colored("  ⚠️ 没有找到稳定币余额", Colors.WARNING)
                print_colored("  💡 提示：请确保账户有 USDT/USDC/FDUSD 等稳定币", Colors.INFO)

            return total

        except Exception as e:
            print(f"获取余额错误: {e}")
            return 0.0

    def get_account_balance(self) -> float:
        """获取账户总余额 - 支持多种稳定币"""
        try:
            # 获取期货账户信息
            account_info = self.client.futures_account()

            # 支持的稳定币列表
            stable_coins = ['USDT', 'USDC', 'BUSD', 'FDUSD', 'TUSD', 'DAI', 'USDP']
            total_balance = 0.0
            found_balances = {}

            # 检查所有资产
            if 'assets' in account_info:
                for asset in account_info['assets']:
                    asset_name = asset['asset']
                    wallet_balance = float(asset.get('walletBalance', 0))

                    # 记录所有有余额的资产
                    if wallet_balance > 0.01:
                        found_balances[asset_name] = wallet_balance

                        # 如果是稳定币，加入总余额
                        if asset_name in stable_coins:
                            total_balance += wallet_balance

            # 如果没找到，尝试备用方法
            if total_balance == 0:
                try:
                    balance_list = self.client.futures_account_balance()
                    for balance in balance_list:
                        asset_name = balance['asset']
                        balance_amount = float(balance.get('balance', 0))

                        if balance_amount > 0.01:
                            if asset_name not in found_balances:
                                found_balances[asset_name] = balance_amount

                            if asset_name in stable_coins:
                                total_balance += balance_amount
                except:
                    pass

            # 打印找到的余额
            if found_balances:
                for asset, amount in found_balances.items():
                    if asset in stable_coins:
                        print_colored(f"  💰 {asset}: {amount:.2f}", Colors.GREEN)
                    else:
                        print_colored(f"  📊 {asset}: {amount:.4f}", Colors.INFO)

            return total_balance

        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_historical_data_safe(self, symbol: str) -> pd.DataFrame:
        """安全获取历史数据"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval='15m',
                limit=500
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            print_colored(f"  ✅ 获取 {len(df)} 条数据", Colors.SUCCESS)
            return df

        except Exception as e:
            self.logger.error(f"获取数据错误: {e}")
            print_colored(f"  ❌ 错误: {e}", Colors.ERROR)
            return None

    def calculate_indicators_safe(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """安全计算技术指标"""
        if df is None or df.empty:
            return df
        try:
            from indicators_module import calculate_optimized_indicators
            df = calculate_optimized_indicators(df)
            # 修复：使用新的填充方法
            df = df.ffill().fillna(0)  # 先前向填充，再填充0
            return df
        except Exception as e:
            self.logger.error(f"计算指标错误: {e}")
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = 50
            df['ATR'] = df['close'].std()
            # 这里也要修复
            df = df.ffill().fillna(0)
            return df

    def has_position(self, symbol: str) -> bool:
        """检查是否有持仓"""
        return symbol in self.positions

    def get_order_book(self, symbol: str) -> dict:
        """获取订单簿"""
        try:
            order_book = self.client.futures_order_book(symbol=symbol, limit=20)
            return {
                'bid_prices': [float(b[0]) for b in order_book.get('bids', [])],
                'bid_sizes': [float(b[1]) for b in order_book.get('bids', [])],
                'ask_prices': [float(a[0]) for a in order_book.get('asks', [])],
                'ask_sizes': [float(a[1]) for a in order_book.get('asks', [])]
            }
        except Exception as e:
            self.logger.error(f"获取订单簿错误: {e}")
            return {'bid_prices': [], 'bid_sizes': [], 'ask_prices': [], 'ask_sizes': []}

    def calculate_position_size(self, symbol: str, position_value: float, price: float) -> float:
        """计算交易数量"""
        try:
            # 获取交易规则
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

            if not symbol_info:
                return 0

            # 获取精度
            quantity_precision = 3  # 默认精度
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    if '.' in str(step_size):
                        quantity_precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    break

            # 计算数量
            quantity = position_value / price
            quantity = round(quantity, quantity_precision)

            return quantity

        except Exception as e:
            self.logger.error(f"计算数量错误: {e}")
            return 0

    def place_order(self, symbol: str, side: str, quantity: float):
        """下单 - 切换实盘/模拟"""

        # ========== 模式选择 ==========
        USE_REAL_TRADING = False  # ← 改为 True 启用实盘

        try:
            print_colored(f"    📤 下单: {side} {quantity} {symbol} @ 市价", Colors.CYAN)

            if USE_REAL_TRADING:
                # ===== 实盘交易 =====
                try:
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=quantity
                    )

                    if order and order.get('status'):
                        print_colored(f"    ✅ 实盘订单成功: {order['orderId']}", Colors.SUCCESS)
                        print_colored(f"    执行价格: {order.get('avgPrice', 'N/A')}", Colors.INFO)
                        print_colored(f"    状态: {order['status']}", Colors.INFO)
                        return order
                    else:
                        print_colored(f"    ❌ 订单失败", Colors.ERROR)
                        return None

                except BinanceAPIException as e:
                    print_colored(f"    ❌ 币安API错误: {e}", Colors.ERROR)
                    print_colored(f"    错误代码: {e.code}, 消息: {e.message}", Colors.ERROR)

                    # 常见错误处理
                    if e.code == -2010:  # 余额不足
                        print_colored("    💔 余额不足", Colors.RED)
                    elif e.code == -1111:  # 精度错误
                        print_colored("    ⚠️ 数量精度错误", Colors.YELLOW)
                    elif e.code == -1021:  # 时间戳错误
                        print_colored("    ⏰ 时间同步问题", Colors.YELLOW)

                    return None
            else:
                # ===== 模拟交易 =====
                print_colored(f"    [模拟] {side} {quantity} {symbol}", Colors.INFO)
                print_colored(f"    💡 提示: 设置 USE_REAL_TRADING = True 启用实盘", Colors.GRAY)

                return {
                    'orderId': f'SIM_{symbol}_{side}_{int(time.time())}',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'type': 'MARKET',
                    'status': 'FILLED',
                    'simulation': True
                }

        except Exception as e:
            print_colored(f"    ❌ 下单错误: {e}", Colors.ERROR)
            return None

    def close_position(self, symbol: str):
        """平仓"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return False

            close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            order = self.place_order(symbol, close_side, position['quantity'])

            if order:
                print_colored(f"    ✅ [模拟] 平仓成功: {symbol}", Colors.SUCCESS)
                return True
            return False

        except Exception as e:
            self.logger.error(f"平仓错误: {e}")
            return False

    def update_trailing_stop(self, symbol: str, current_price: float):
        """更新移动止损"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            trail_percent = 0.02  # 2%移动止损

            if position['side'] == 'LONG':
                if 'highest_price' not in position:
                    position['highest_price'] = current_price
                else:
                    position['highest_price'] = max(position['highest_price'], current_price)

                new_stop = position['highest_price'] * (1 - trail_percent)
                if new_stop > position.get('stop_loss', 0):
                    position['stop_loss'] = new_stop
                    print_colored(f"    📈 更新止损: {new_stop:.4f}", Colors.INFO)

            else:  # SHORT
                if 'lowest_price' not in position:
                    position['lowest_price'] = current_price
                else:
                    position['lowest_price'] = min(position['lowest_price'], current_price)

                new_stop = position['lowest_price'] * (1 + trail_percent)
                if new_stop < position.get('stop_loss', float('inf')):
                    position['stop_loss'] = new_stop
                    print_colored(f"    📉 更新止损: {new_stop:.4f}", Colors.INFO)

        except Exception as e:
            self.logger.error(f"更新止损错误: {e}")

    def run_trading_cycle(self):
        """
        运行交易循环 - 重写父类方法
        不使用评分，使用交易计划
        """

        print_colored("\n" + "=" * 60, Colors.BLUE)
        print_colored(f"🔄 增强交易循环 - {datetime.now().strftime('%H:%M:%S')}", Colors.CYAN)

        try:
            # 1. 获取账户信息（使用父类方法）
            balance = self.get_account_balance()
            print_colored(f"💰 账户余额: {balance:.2f} USDT", Colors.INFO)

            if balance < MIN_MARGIN_BALANCE:
                print_colored("⚠️ 余额不足", Colors.WARNING)
                return

            # 2. 管理现有持仓（使用父类方法）
            self._manage_existing_positions()

            # 3. 检查活跃计划
            self._check_active_plans()

            # 4. 寻找新机会
            if len(self.positions) < MAX_POSITIONS:
                self._find_new_opportunities(balance)

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}")
            print_colored(f"❌ 错误: {e}", Colors.ERROR)

    def _find_new_opportunities(self, balance: float):
        """寻找新的交易机会"""

        print_colored("\n🔍 扫描交易机会...", Colors.CYAN)

        for symbol in TRADE_PAIRS:
            try:
                # 跳过已有持仓或计划
                if self.has_position(symbol) or symbol in self.active_plans:
                    continue

                print_colored(f"\n📊 分析 {symbol}...", Colors.INFO)

                # 1. 获取数据（使用父类方法）
                df = self.get_historical_data_safe(symbol)
                if df is None or len(df) < 100:
                    print_colored("  ⚠️ 数据不足", Colors.WARNING)
                    continue

                # 2. 计算指标（使用父类方法）
                df = self.calculate_indicators_safe(df, symbol)

                # 3. 获取订单簿（使用父类方法）
                order_book = self.get_order_book(symbol)

                # 4. 流动性分析
                liquidity_data = self.liquidity_analyzer.analyze(df)

                # 5. 综合分析
                analysis = self.game_analyzer.analyze(df, order_book, liquidity_data)

                # 6. 打印分析结果
                self._print_analysis(symbol, analysis)

                # 7. 如果有信号，创建交易计划
                if analysis['direction'] != 'NEUTRAL':
                    plan = self.plan_manager.create_plan(analysis, symbol)

                    if plan:
                        self._print_plan(plan)

                        # 激活计划
                        if self._should_activate_plan(plan):
                            self.active_plans[symbol] = plan
                            print_colored(f"  ✅ 计划已激活", Colors.SUCCESS)

                            # 立即检查是否可以执行
                            self._try_execute_plan(symbol, plan)

            except Exception as e:
                self.logger.error(f"分析{symbol}错误: {e}")
                print_colored(f"  ❌ 错误: {e}", Colors.ERROR)

    def _check_active_plans(self):
        """检查并执行活跃的计划"""

        if not self.active_plans:
            return

        print_colored("\n📋 检查活跃计划...", Colors.INFO)

        for symbol, plan in list(self.active_plans.items()):
            try:
                # 检查是否过期
                if datetime.now() > plan['valid_until']:
                    print_colored(f"  ⏰ {symbol} 计划已过期", Colors.WARNING)
                    del self.active_plans[symbol]
                    continue

                # 尝试执行
                self._try_execute_plan(symbol, plan)

            except Exception as e:
                self.logger.error(f"检查计划错误: {e}")

    def _try_execute_plan(self, symbol: str, plan: Dict):
        """尝试执行交易计划"""

        try:
            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            entry_price = plan['entry']['primary']['price']
            direction = plan['direction']

            # 检查入场条件
            should_enter = False

            if direction == 'LONG':
                if current_price <= entry_price * 1.002:  # 允许0.2%滑点
                    should_enter = True
            else:
                if current_price >= entry_price * 0.998:
                    should_enter = True

            if should_enter:
                print_colored(f"  🎯 触发入场: {symbol} @ {current_price:.4f}", Colors.SUCCESS)

                # 执行交易（使用父类方法）
                success = self._execute_trade(symbol, plan, current_price)

                if success:
                    # 记录计划
                    self.plan_history.append({
                        'plan': plan,
                        'executed_at': datetime.now(),
                        'executed_price': current_price
                    })

                    # 移除活跃计划
                    del self.active_plans[symbol]
            else:
                diff = ((current_price - entry_price) / entry_price) * 100
                print_colored(f"  ⏳ {symbol}: 等待入场 (差距: {diff:+.2f}%)", Colors.GRAY)

        except Exception as e:
            self.logger.error(f"执行计划错误: {e}")

    def _execute_trade(self, symbol: str, plan: Dict, price: float) -> bool:
        """执行交易"""

        try:
            # 计算仓位大小
            balance = self.get_account_balance()
            position_value = balance * (plan['entry']['primary']['size_percent'] / 100)

            # 使用父类方法计算数量
            quantity = self.calculate_position_size(symbol, position_value, price)

            if quantity == 0:
                print_colored("  ⚠️ 数量计算为0", Colors.WARNING)
                return False

            # 确定方向
            side = 'BUY' if plan['direction'] == 'LONG' else 'SELL'

            print_colored(f"  📤 下单: {side} {quantity} {symbol} @ 市价", Colors.CYAN)

            # 使用父类方法下单
            order = self.place_order(symbol, side, quantity)

            if order:
                # 记录持仓
                self.positions[symbol] = {
                    'side': plan['direction'],
                    'entry_price': price,
                    'quantity': quantity,
                    'stop_loss': plan['exit']['stop_loss']['initial'],
                    'take_profit': plan['exit']['take_profit']['targets'],
                    'plan': plan,
                    'entry_time': datetime.now()
                }

                print_colored(f"  ✅ 交易成功", Colors.SUCCESS)
                return True

        except Exception as e:
            self.logger.error(f"执行交易错误: {e}")
            print_colored(f"  ❌ 执行失败: {e}", Colors.ERROR)

        return False

    def _manage_existing_positions(self):
        """管理现有持仓"""

        if not self.positions:
            return

        print_colored("\n📊 管理持仓...", Colors.INFO)

        for symbol, position in list(self.positions.items()):
            try:
                # 获取当前价格
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # 计算盈亏
                if position['side'] == 'LONG':
                    pnl = ((current_price - position['entry_price']) / position['entry_price']) * 100
                else:
                    pnl = ((position['entry_price'] - current_price) / position['entry_price']) * 100

                # 显示状态
                color = Colors.GREEN if pnl > 0 else Colors.RED
                print_colored(f"  {symbol}: {pnl:+.2f}%", color)

                # 检查出场条件
                should_exit, reason = self._check_exit_conditions(symbol, position, current_price)

                if should_exit:
                    print_colored(f"    🛑 触发{reason}", Colors.WARNING)
                    self.close_position(symbol)
                    del self.positions[symbol]
                else:
                    # 更新移动止损
                    self.update_trailing_stop(symbol, current_price)

            except Exception as e:
                self.logger.error(f"管理持仓错误: {e}")

    def _check_exit_conditions(self, symbol: str, position: Dict, current_price: float) -> tuple:
        """检查出场条件"""

        # 止损检查
        if position['side'] == 'LONG':
            if current_price <= position['stop_loss']:
                return True, "止损"
        else:
            if current_price >= position['stop_loss']:
                return True, "止损"

        # 止盈检查
        if position['take_profit']:
            if position['side'] == 'LONG':
                if current_price >= position['take_profit'][0]:
                    return True, "止盈"
            else:
                if current_price <= position['take_profit'][0]:
                    return True, "止盈"

        return False, ""

    def _should_activate_plan(self, plan: Dict) -> bool:
        """判断是否应该激活计划 - 更灵活的版本"""

        # 根据交易类型调整要求
        if plan.get('trade_type') == 'PULLBACK':
            min_confidence = 0.35  # 回调交易降低要求
        else:
            min_confidence = 0.4  # 趋势交易

        # 检查置信度
        if plan['confidence'] < min_confidence:
            print_colored(f"  ⚠️ 置信度 {plan['confidence']:.1%} < {min_confidence:.1%}", Colors.YELLOW)

            # 如果有强烈的超买/超卖信号，仍然可以执行
            if plan.get('trade_type') == 'PULLBACK' and plan['confidence'] > 0.3:
                print_colored(f"  ✅ 回调交易，降低置信度要求", Colors.GREEN)
                return True

            return False

        # 检查风险回报比
        if plan['risk']['risk_reward_ratio'] < 1.0:
            print_colored(f"  ❌ 风险回报比过低", Colors.WARNING)
            return False

        # 显示警告
        if plan.get('warnings'):
            print_colored("\n  ⚠️ 风险警告:", Colors.YELLOW)
            for warning in plan['warnings']:
                print_colored(f"    {warning}", Colors.RED)

        return True

    def _print_analysis(self, symbol: str, analysis: Dict):
        """打印分析结果"""

        if analysis['direction'] == 'NEUTRAL':
            print_colored(f"  😴 无信号", Colors.GRAY)
        else:
            color = Colors.GREEN if analysis['direction'] == 'LONG' else Colors.RED
            print_colored(f"  📈 方向: {analysis['direction']}", color)
            print_colored(f"  💯 置信度: {analysis['confidence']:.1%}", Colors.INFO)
            print_colored(f"  📊 风险回报比: {analysis['risk_reward_ratio']:.1f}", Colors.INFO)

            if analysis['reasoning']:
                print_colored("  📝 理由:", Colors.INFO)
                for reason in analysis['reasoning'][:3]:
                    print_colored(f"    • {reason}", Colors.GRAY)

    def _print_plan(self, plan: Dict):
        """打印交易计划 - 增强版"""

        print_colored("\n  📋 交易计划:", Colors.CYAN)

        # 显示交易类型
        trade_type = plan.get('trade_type', 'TREND')
        if trade_type == 'PULLBACK':
            print_colored(f"    ⚠️ 类型: 回调交易（高风险）", Colors.YELLOW)
        else:
            print_colored(f"    类型: 趋势交易", Colors.INFO)

        print_colored(f"    方向: {plan['direction']}", Colors.INFO)
        print_colored(f"    置信度: {plan['confidence']:.1%}", Colors.INFO)

        # 入场和出场
        print_colored(f"    入场: {plan['entry']['primary']['price']:.4f}", Colors.INFO)
        print_colored(f"    止损: {plan['exit']['stop_loss']['initial']:.4f}", Colors.WARNING)

        if plan['exit']['take_profit']['targets']:
            print_colored(f"    止盈1: {plan['exit']['take_profit']['targets'][0]:.4f}", Colors.SUCCESS)

        # 显示警告
        if plan.get('warnings'):
            print_colored("\n    ⛔ 重要警告:", Colors.RED)
            for warning in plan['warnings']:
                print_colored(f"      {warning}", Colors.YELLOW)


def main():
    """主函数"""

    print_colored("\n" + "=" * 60, Colors.CYAN)
    print_colored("🚀 启动增强交易系统", Colors.SUCCESS)
    print_colored("基于交易计划的决策系统 v2.0", Colors.INFO)
    print_colored("=" * 60, Colors.CYAN)

    try:
        # 创建增强系统实例
        trading_system = EnhancedTradingSystem()

        # 主循环
        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                print_colored(f"\n🔄 第 {cycle_count} 轮循环", Colors.BLUE)

                # 运行交易循环
                trading_system.run_trading_cycle()

                # 等待下一轮
                wait_time = 120  # 5分钟
                print_colored(f"\n⏳ 等待 {wait_time} 秒...", Colors.INFO)

                for remaining in range(wait_time, 0, -30):
                    print(f"\r剩余: {remaining}秒", end='', flush=True)
                    time.sleep(min(30, remaining))

            except KeyboardInterrupt:
                print_colored("\n\n⚠️ 收到中断信号", Colors.WARNING)
                break
            except Exception as e:
                print_colored(f"\n❌ 循环错误: {e}", Colors.ERROR)
                time.sleep(30)

    except Exception as e:
        print_colored(f"\n❌ 系统错误: {e}", Colors.ERROR)
        return 1

    print_colored("\n👋 系统已停止", Colors.INFO)
    return 0


if __name__ == "__main__":
    sys.exit(main())