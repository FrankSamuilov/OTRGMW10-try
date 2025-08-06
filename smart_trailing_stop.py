"""
智能移动止盈系统 - 结合博弈论和技术指标的动态止盈策略
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from logger_utils import Colors, print_colored


class SmartTrailingStop:
    """智能移动止盈系统"""

    def __init__(self, logger=None):
        self.logger = logger
        self.trailing_records = {}  # 记录每个持仓的止盈历史

    def calculate_trailing_parameters(self, position: Dict, market_analysis: Dict,
                                      game_analysis: Dict, technical_analysis: Dict) -> Dict[str, Any]:
        """
        计算智能移动止盈参数

        参数:
            position: 当前持仓信息
            market_analysis: 市场分析数据
            game_analysis: 博弈论分析数据
            technical_analysis: 技术分析数据

        返回:
            移动止盈参数字典
        """

        # 获取基础信息
        symbol = position['symbol']
        entry_price = position['entry_price']
        current_price = position.get('current_price', entry_price)
        position_side = position.get('position_side', 'LONG')

        # 计算当前盈利百分比
        if position_side == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # 初始化参数
        params = {
            'should_activate': False,  # 是否激活移动止盈
            'should_update': False,  # 是否更新止盈位
            'new_stop_level': None,  # 新的止损位
            'trailing_distance': 0.8,  # 默认80%跟踪
            'reason': '',  # 决策原因
            'confidence': 0.0  # 决策置信度
        }

        # 1. 检查是否达到激活阈值（1%）
        if profit_pct < 1.0:
            params['reason'] = f'盈利未达到激活阈值 ({profit_pct:.2f}% < 1.0%)'
            return params

        # 2. 计算基于博弈论的跟踪距离
        game_trailing = self._calculate_game_theory_trailing(game_analysis, profit_pct)

        # 3. 计算基于技术指标的跟踪距离
        tech_trailing = self._calculate_technical_trailing(technical_analysis, profit_pct)

        # 4. 计算基于市场环境的跟踪距离
        market_trailing = self._calculate_market_trailing(market_analysis, profit_pct)

        # 5. 综合计算最终跟踪距离
        trailing_weights = {
            'game': 0.4,  # 博弈论权重
            'tech': 0.4,  # 技术指标权重
            'market': 0.2  # 市场环境权重
        }

        weighted_trailing = (
                game_trailing['distance'] * trailing_weights['game'] +
                tech_trailing['distance'] * trailing_weights['tech'] +
                market_trailing['distance'] * trailing_weights['market']
        )

        # 6. 决定是否需要上移止损
        confidence = (game_trailing['confidence'] + tech_trailing['confidence'] +
                      market_trailing['confidence']) / 3

        # 获取当前最高/最低价
        if position_side == 'LONG':
            highest_price = position.get('highest_price', current_price)
            if current_price > highest_price:
                # 创新高，考虑是否上移
                if confidence >= 0.5:  # 置信度超过50%才上移
                    params['should_update'] = True
                    params['new_stop_level'] = current_price * (1 - (1 - weighted_trailing) * profit_pct / 100)
                    params['trailing_distance'] = weighted_trailing
                    params['confidence'] = confidence
                    params['reason'] = self._build_reason(game_trailing, tech_trailing, market_trailing)
                else:
                    params['reason'] = f'置信度不足 ({confidence:.2f} < 0.5)'
        else:  # SHORT
            lowest_price = position.get('lowest_price', current_price)
            if current_price < lowest_price:
                # 创新低，考虑是否下移
                if confidence >= 0.5:
                    params['should_update'] = True
                    params['new_stop_level'] = current_price * (1 + (1 - weighted_trailing) * profit_pct / 100)
                    params['trailing_distance'] = weighted_trailing
                    params['confidence'] = confidence
                    params['reason'] = self._build_reason(game_trailing, tech_trailing, market_trailing)
                else:
                    params['reason'] = f'置信度不足 ({confidence:.2f} < 0.5)'

        # 7. 安全检查 - 确保止损只会向有利方向移动
        if params['should_update'] and params['new_stop_level']:
            current_stop = position.get('current_stop_level', 0)
            if position_side == 'LONG':
                if params['new_stop_level'] <= current_stop:
                    params['should_update'] = False
                    params['reason'] = '新止损位低于当前止损位，不更新'
            else:  # SHORT
                if params['new_stop_level'] >= current_stop:
                    params['should_update'] = False
                    params['reason'] = '新止损位高于当前止损位，不更新'

        return params

    def _calculate_game_theory_trailing(self, game_analysis: Dict, profit_pct: float) -> Dict[str, float]:
        """基于博弈论计算跟踪距离"""

        distance = 0.8  # 默认80%
        confidence = 0.5

        if not game_analysis:
            return {'distance': distance, 'confidence': confidence}

        # 1. 市场操纵检测
        manipulation_score = game_analysis.get('manipulation_detection', {}).get('total_manipulation_score', 0)
        if manipulation_score > 0.7:
            # 高操纵环境，收紧跟踪
            distance = max(0.6, 0.8 - manipulation_score * 0.2)
            confidence *= 0.8

        # 2. 订单流毒性
        toxicity = game_analysis.get('order_flow_toxicity', {}).get('toxicity_level', 'LOW')
        if toxicity == 'HIGH':
            distance = 0.65  # 高毒性，更紧密跟踪
            confidence *= 0.9
        elif toxicity == 'MEDIUM':
            distance = 0.75

        # 3. 聪明钱流向
        smart_money = game_analysis.get('smart_money_flow', {}).get('smart_money_direction', 'NEUTRAL')
        position_side = game_analysis.get('position_side', 'LONG')

        # 如果聪明钱与持仓方向一致，可以放松跟踪
        if (position_side == 'LONG' and 'ACCUMULATING' in smart_money) or \
                (position_side == 'SHORT' and 'DISTRIBUTING' in smart_money):
            distance = min(0.85, distance + 0.05)
            confidence *= 1.2

        # 4. 基于盈利调整
        if profit_pct > 5:
            # 高盈利时收紧跟踪
            distance = max(0.7, distance - 0.05)
        elif profit_pct > 10:
            distance = max(0.6, distance - 0.1)

        return {
            'distance': distance,
            'confidence': min(1.0, confidence)
        }

    def _calculate_technical_trailing(self, technical_analysis: Dict, profit_pct: float) -> Dict[str, float]:
        """基于技术指标计算跟踪距离"""

        distance = 0.8
        confidence = 0.5

        if not technical_analysis:
            return {'distance': distance, 'confidence': confidence}

        # 1. RSI
        rsi = technical_analysis.get('rsi', 50)
        if rsi > 70:
            # 超买，收紧跟踪
            distance = 0.7
            confidence *= 1.1
        elif rsi < 30:
            # 超卖（对于空头是好事）
            distance = 0.7
            confidence *= 1.1

        # 2. 波动率（使用ATR）
        atr_ratio = technical_analysis.get('atr_ratio', 1.0)  # ATR相对于价格的比率
        if atr_ratio > 0.02:  # 高波动
            distance = max(0.65, distance - atr_ratio * 5)
            confidence *= 0.9
        elif atr_ratio < 0.01:  # 低波动
            distance = min(0.85, distance + 0.05)
            confidence *= 1.1

        # 3. 趋势强度
        trend_strength = technical_analysis.get('trend_strength', 0)
        if abs(trend_strength) > 0.7:
            # 强趋势，可以放松跟踪
            distance = min(0.85, distance + 0.05)
            confidence *= 1.2

        # 4. MACD
        macd_signal = technical_analysis.get('macd_signal', 'NEUTRAL')
        position_side = technical_analysis.get('position_side', 'LONG')

        if (position_side == 'LONG' and macd_signal == 'BEARISH') or \
                (position_side == 'SHORT' and macd_signal == 'BULLISH'):
            # 反向信号，收紧跟踪
            distance = max(0.65, distance - 0.1)
            confidence *= 1.2

        return {
            'distance': distance,
            'confidence': min(1.0, confidence)
        }

    def _calculate_market_trailing(self, market_analysis: Dict, profit_pct: float) -> Dict[str, float]:
        """基于市场环境计算跟踪距离"""

        distance = 0.8
        confidence = 0.5

        if not market_analysis:
            return {'distance': distance, 'confidence': confidence}

        # 市场环境
        environment = market_analysis.get('environment', 'unknown')

        if environment == 'trending':
            # 趋势市场，可以放松跟踪
            distance = 0.85
            confidence = 0.8
        elif environment == 'ranging':
            # 震荡市场，收紧跟踪
            distance = 0.7
            confidence = 0.7
        elif environment == 'breakout':
            # 突破市场，根据盈利调整
            if profit_pct < 3:
                distance = 0.85  # 初期放松
            else:
                distance = 0.75  # 后期收紧
            confidence = 0.9
        elif environment == 'extreme_volatility':
            # 极端波动，大幅收紧
            distance = 0.6
            confidence = 0.6

        return {
            'distance': distance,
            'confidence': min(1.0, confidence)
        }

    def _build_reason(self, game: Dict, tech: Dict, market: Dict) -> str:
        """构建决策原因说明"""
        reasons = []

        # 添加主要影响因素
        factors = [
            (game, "博弈论"),
            (tech, "技术指标"),
            (market, "市场环境")
        ]

        for factor, name in factors:
            if factor['confidence'] > 0.7:
                reasons.append(f"{name}支持(置信度:{factor['confidence']:.2f})")

        return ", ".join(reasons) if reasons else "综合评估"

    def apply_trailing_stop(self, position: Dict, market_data: Dict) -> Dict[str, Any]:
        """
        应用移动止盈逻辑到持仓

        返回:
            包含是否需要平仓和更新信息的字典
        """
        symbol = position['symbol']
        position_side = position.get('position_side', 'LONG')
        current_price = market_data.get('current_price', 0)

        # 更新当前价格
        position['current_price'] = current_price

        # 获取各项分析数据
        game_analysis = market_data.get('game_analysis', {})
        technical_analysis = market_data.get('technical_analysis', {})
        market_analysis = market_data.get('market_analysis', {})

        # 计算移动止盈参数
        trailing_params = self.calculate_trailing_parameters(
            position, market_analysis, game_analysis, technical_analysis
        )

        result = {
            'should_close': False,
            'close_reason': '',
            'updated_position': position.copy(),
            'trailing_info': trailing_params
        }

        # 检查是否触发止损
        current_stop = position.get('current_stop_level', 0)
        if current_stop > 0:
            if (position_side == 'LONG' and current_price <= current_stop) or \
                    (position_side == 'SHORT' and current_price >= current_stop):
                result['should_close'] = True
                result['close_reason'] = f'触发移动止损 (价格:{current_price:.6f}, 止损:{current_stop:.6f})'
                return result

        # 更新止损位
        if trailing_params['should_update']:
            old_stop = position.get('current_stop_level', 0)
            new_stop = trailing_params['new_stop_level']

            # 更新持仓信息
            result['updated_position']['current_stop_level'] = new_stop
            result['updated_position']['trailing_distance'] = trailing_params['trailing_distance']
            result['updated_position']['trailing_active'] = True

            # 更新最高/最低价
            if position_side == 'LONG':
                result['updated_position']['highest_price'] = current_price
            else:
                result['updated_position']['lowest_price'] = current_price

            # 记录更新
            print_colored(
                f"📈 {symbol} {position_side} 更新移动止损: "
                f"{old_stop:.6f} → {new_stop:.6f} "
                f"(跟踪距离: {trailing_params['trailing_distance']:.1%})",
                Colors.GREEN
            )
            print_colored(f"   原因: {trailing_params['reason']}", Colors.INFO)

        return result