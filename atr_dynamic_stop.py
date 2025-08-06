"""
基于ATR的动态止损系统 - 根据市场波动率自适应调整止损位
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from logger_utils import Colors, print_colored


class ATRDynamicStopLoss:
    """ATR动态止损系统"""

    def __init__(self, base_multiplier: float = 2.0, logger=None):
        """
        初始化ATR止损系统

        参数:
            base_multiplier: 基础ATR倍数（默认2倍）
            logger: 日志对象
        """
        self.base_multiplier = base_multiplier
        self.logger = logger
        self.atr_history = {}  # 记录ATR历史

    def calculate_atr_stop_loss(self, entry_price: float, current_atr: float,
                                leverage: int, side: str, market_conditions: Dict) -> Dict[str, Any]:
        """
        计算基于ATR的动态止损

        参数:
            entry_price: 入场价格
            current_atr: 当前ATR值
            leverage: 杠杆倍数
            side: 交易方向 ("BUY" or "SELL")
            market_conditions: 市场条件

        返回:
            止损参数字典
        """

        # 1. 基础ATR倍数调整
        atr_multiplier = self._adjust_atr_multiplier(leverage, market_conditions)

        # 2. 计算基础止损距离
        base_stop_distance = current_atr * atr_multiplier

        # 3. 根据市场环境微调
        adjusted_distance = self._adjust_for_market_environment(
            base_stop_distance,
            market_conditions
        )

        # 4. 计算最终止损价格
        if side.upper() == "BUY":
            stop_loss_price = entry_price - adjusted_distance
        else:  # SELL
            stop_loss_price = entry_price + adjusted_distance

        # 5. 计算止损百分比
        stop_loss_pct = adjusted_distance / entry_price * 100

        # 6. 安全检查 - 确保止损不会太近或太远
        min_stop_pct = 0.5  # 最小0.5%
        max_stop_pct = 5.0  # 最大5%

        if stop_loss_pct < min_stop_pct:
            stop_loss_pct = min_stop_pct
            adjusted_distance = entry_price * stop_loss_pct / 100
            if side.upper() == "BUY":
                stop_loss_price = entry_price - adjusted_distance
            else:
                stop_loss_price = entry_price + adjusted_distance
            print_colored(f"⚠️ ATR止损过小，调整为最小值 {min_stop_pct}%", Colors.WARNING)

        elif stop_loss_pct > max_stop_pct:
            stop_loss_pct = max_stop_pct
            adjusted_distance = entry_price * stop_loss_pct / 100
            if side.upper() == "BUY":
                stop_loss_price = entry_price - adjusted_distance
            else:
                stop_loss_price = entry_price + adjusted_distance
            print_colored(f"⚠️ ATR止损过大，调整为最大值 {max_stop_pct}%", Colors.WARNING)

        result = {
            'stop_loss_price': stop_loss_price,
            'stop_loss_pct': stop_loss_pct,
            'atr_value': current_atr,
            'atr_multiplier': atr_multiplier,
            'stop_distance': adjusted_distance,
            'method': 'ATR_DYNAMIC'
        }

        # 打印详细信息
        print_colored("🎯 ATR动态止损计算:", Colors.BLUE)
        print_colored(f"   当前ATR: {current_atr:.6f}", Colors.INFO)
        print_colored(f"   ATR倍数: {atr_multiplier:.2f}", Colors.INFO)
        print_colored(f"   杠杆: {leverage}x", Colors.INFO)
        print_colored(f"   止损距离: {adjusted_distance:.6f} ({stop_loss_pct:.2f}%)", Colors.INFO)
        print_colored(f"   止损价格: {stop_loss_price:.6f}", Colors.INFO)

        return result

    def _adjust_atr_multiplier(self, leverage: int, market_conditions: Dict) -> float:
        """
        根据杠杆和市场条件调整ATR倍数

        高杠杆需要更紧的止损，低杠杆可以更宽松
        """
        # 基础倍数
        multiplier = self.base_multiplier

        # 1. 根据杠杆调整
        if leverage >= 20:
            multiplier *= 0.6  # 高杠杆，收紧止损
        elif leverage >= 10:
            multiplier *= 0.8
        elif leverage >= 5:
            multiplier *= 0.9
        else:
            multiplier *= 1.1  # 低杠杆，可以放宽

        # 2. 根据市场波动率调整
        volatility = market_conditions.get('volatility_level', 'NORMAL')
        if volatility == 'EXTREME':
            multiplier *= 1.3  # 极端波动，需要更宽的止损
        elif volatility == 'HIGH':
            multiplier *= 1.15
        elif volatility == 'LOW':
            multiplier *= 0.85  # 低波动，可以收紧

        # 3. 根据趋势强度调整
        trend_strength = market_conditions.get('trend_strength', 0)
        if abs(trend_strength) > 0.7:
            # 强趋势，可以适当放宽止损
            multiplier *= 1.1

        # 确保倍数在合理范围内
        return max(1.0, min(3.0, multiplier))

    def _adjust_for_market_environment(self, base_distance: float,
                                       market_conditions: Dict) -> float:
        """根据市场环境微调止损距离"""

        distance = base_distance
        environment = market_conditions.get('environment', 'unknown')

        if environment == 'trending':
            # 趋势市场，可以适当放宽
            distance *= 1.1
        elif environment == 'ranging':
            # 震荡市场，需要收紧
            distance *= 0.9
        elif environment == 'breakout':
            # 突破初期，给予更多空间
            distance *= 1.2
        elif environment == 'reversal_risk':
            # 潜在反转，收紧止损
            distance *= 0.8

        # 考虑支撑阻力位
        if 'nearest_support' in market_conditions or 'nearest_resistance' in market_conditions:
            distance = self._adjust_for_sr_levels(distance, market_conditions)

        return distance

    def _adjust_for_sr_levels(self, distance: float, market_conditions: Dict) -> float:
        """根据支撑阻力位调整止损距离"""

        # 这里可以根据最近的支撑阻力位来微调止损距离
        # 确保止损设置在关键位置之外

        support = market_conditions.get('nearest_support')
        resistance = market_conditions.get('nearest_resistance')
        current_price = market_conditions.get('current_price')

        if support and current_price and abs(current_price - support) < distance * 1.5:
            # 如果止损会设置在支撑位附近，稍微调整
            distance = abs(current_price - support) * 1.1
            print_colored(f"📍 调整止损以避开支撑位 {support:.6f}", Colors.INFO)

        return distance

    def update_stop_loss_dynamically(self, position: Dict, current_atr: float,
                                     current_price: float, market_conditions: Dict) -> Dict[str, Any]:
        """
        动态更新现有持仓的止损

        参数:
            position: 持仓信息
            current_atr: 当前ATR
            current_price: 当前价格
            market_conditions: 市场条件

        返回:
            更新后的止损信息
        """

        symbol = position['symbol']
        entry_atr = position.get('entry_atr', current_atr)
        position_side = position.get('position_side', 'LONG')
        leverage = position.get('leverage', 1)

        # 1. 计算ATR变化率
        atr_change_ratio = current_atr / entry_atr if entry_atr > 0 else 1.0

        # 2. 决定是否需要调整止损
        should_adjust = False
        adjustment_reason = ""

        if atr_change_ratio > 1.3:
            # ATR显著增加，考虑放宽止损
            should_adjust = True
            adjustment_reason = f"ATR增加{(atr_change_ratio - 1) * 100:.1f}%，市场波动加大"
        elif atr_change_ratio < 0.7:
            # ATR显著减少，考虑收紧止损
            should_adjust = True
            adjustment_reason = f"ATR减少{(1 - atr_change_ratio) * 100:.1f}%，市场波动减小"

        # 3. 检查是否在盈利中
        entry_price = position['entry_price']
        if position_side == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # 4. 盈利状态下的特殊处理
        if profit_pct > 2:  # 盈利超过2%
            # 使用更紧的ATR倍数来保护利润
            atr_multiplier = max(1.0, self.base_multiplier * 0.7)
            should_adjust = True
            adjustment_reason += f" | 盈利{profit_pct:.1f}%，收紧止损保护利润"
        else:
            atr_multiplier = self._adjust_atr_multiplier(leverage, market_conditions)

        result = {
            'should_adjust': should_adjust,
            'reason': adjustment_reason,
            'current_atr': current_atr,
            'entry_atr': entry_atr,
            'atr_change_ratio': atr_change_ratio,
            'new_stop_loss': None
        }

        if should_adjust:
            # 计算新的止损
            new_stop_distance = current_atr * atr_multiplier

            if position_side == 'LONG':
                new_stop_loss = current_price - new_stop_distance
                # 确保止损只向上移动（保护利润）
                current_stop = position.get('current_stop_level', 0)
                if new_stop_loss > current_stop:
                    result['new_stop_loss'] = new_stop_loss
                    result['stop_loss_pct'] = new_stop_distance / current_price * 100
                else:
                    result['should_adjust'] = False
                    result['reason'] = "新止损低于当前止损，保持不变"
            else:  # SHORT
                new_stop_loss = current_price + new_stop_distance
                # 确保止损只向下移动（保护利润）
                current_stop = position.get('current_stop_level', float('inf'))
                if new_stop_loss < current_stop:
                    result['new_stop_loss'] = new_stop_loss
                    result['stop_loss_pct'] = new_stop_distance / current_price * 100
                else:
                    result['should_adjust'] = False
                    result['reason'] = "新止损高于当前止损，保持不变"

        if result['should_adjust'] and result['new_stop_loss']:
            print_colored(f"🔄 {symbol} ATR动态止损调整:", Colors.CYAN)
            print_colored(f"   原因: {adjustment_reason}", Colors.INFO)
            print_colored(f"   新止损: {result['new_stop_loss']:.6f} ({result['stop_loss_pct']:.2f}%)", Colors.INFO)

        return result

    def calculate_initial_stop_with_atr(self, df: pd.DataFrame, entry_price: float,
                                        side: str, leverage: int = 1) -> Dict[str, Any]:
        """
        使用历史数据计算初始ATR止损

        参数:
            df: 包含价格数据的DataFrame
            entry_price: 入场价格
            side: 交易方向
            leverage: 杠杆倍数

        返回:
            止损参数
        """

        # 确保有ATR数据
        if 'ATR' not in df.columns:
            # 计算ATR
            df['H-L'] = df['high'] - df['low']
            df['H-PC'] = abs(df['high'] - df['close'].shift(1))
            df['L-PC'] = abs(df['low'] - df['close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

        current_atr = df['ATR'].iloc[-1]

        # 获取市场条件
        market_conditions = self._analyze_market_conditions(df)

        # 计算止损
        return self.calculate_atr_stop_loss(
            entry_price=entry_price,
            current_atr=current_atr,
            leverage=leverage,
            side=side,
            market_conditions=market_conditions
        )

    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析市场条件用于ATR调整"""

        conditions = {
            'volatility_level': 'NORMAL',
            'trend_strength': 0,
            'environment': 'unknown'
        }

        # 计算波动率水平
        recent_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].iloc[-20:].mean()

        if recent_atr > avg_atr * 1.5:
            conditions['volatility_level'] = 'HIGH'
        elif recent_atr > avg_atr * 2:
            conditions['volatility_level'] = 'EXTREME'
        elif recent_atr < avg_atr * 0.7:
            conditions['volatility_level'] = 'LOW'

        # 简单的趋势强度计算
        if len(df) > 50:
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            price = df['close'].iloc[-1]

            if price > ema20 > ema50:
                conditions['trend_strength'] = 0.8
                conditions['environment'] = 'trending'
            elif price < ema20 < ema50:
                conditions['trend_strength'] = -0.8
                conditions['environment'] = 'trending'
            else:
                conditions['environment'] = 'ranging'

        return conditions