# enhanced_scoring_system.py
# 增强版动态评分系统 - 支持形态共振和自适应权重

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored

# enhanced_scoring_system_fixed.py
# 修复后的增强版动态评分系统

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored


class EnhancedScoringSystem:
    """
    修复版评分系统
    主要修改：
    1. 降低交易阈值，使系统更加积极
    2. 调整技术指标评分逻辑
    3. 增加调试信息输出
    """

    def __init__(self, logger=None):
        self.logger = logger

        # 调整基础评分权重 - 增加技术和博弈论权重
        self.base_weights = {
            'categories': {
                'trend': 0.35,  # 提高趋势权重（从0.20到0.35）
                'technical': 0.30,  # 降低技术权重（从0.35到0.30）
                'game_theory': 0.25,  # 降低博弈论权重（从0.30到0.25）
                'market_structure': 0.10  # 降低市场结构权重（从0.15到0.10）
            },
            'patterns': {
                # 反转形态权重
                'head_shoulders': 0.85,
                'inverse_head_shoulders': 0.90,
                'double_top': 0.75,
                'double_bottom': 0.80,

                # 持续形态权重
                'triangle': 0.65,
                'flag': 0.70,
                'wedge': 0.60,

                # 博弈论形态权重
                'stop_hunt': 0.80,
                'liquidity_grab': 0.85,
                'wyckoff': 0.90,
                'squeeze': 0.85,
                'accumulation': 0.85,  # 添加吸筹形态
                'distribution': 0.85,  # 添加派发形态

                # 市场结构权重
                'poc_reversion': 0.75,
                'value_area_trade': 0.70,
                'trend_day': 0.80
            }
        }

        # 调整市场环境配置 - 使权重更均衡
        self.market_regimes = {
            'TRENDING': {'trend': 0.45, 'technical': 0.25, 'game_theory': 0.20, 'market_structure': 0.10},
            'RANGING': {'trend': 0.15, 'technical': 0.35, 'game_theory': 0.35, 'market_structure': 0.15},
            'VOLATILE': {'trend': 0.25, 'technical': 0.30, 'game_theory': 0.35, 'market_structure': 0.10},
            'BREAKOUT': {'trend': 0.35, 'technical': 0.30, 'game_theory': 0.25, 'market_structure': 0.10}
        }

        print_colored("✅ 修复版评分系统初始化完成", Colors.GREEN)

    def calculate_comprehensive_score(self, analysis_data: Dict) -> Dict:
        """计算综合评分（带调试信息）"""

        # 添加调试信息
        print_colored("\n📊 === 评分系统调试信息 ===", Colors.CYAN)

        # 1. 检测市场环境
        market_regime = self._detect_market_regime(analysis_data)
        print_colored(f"市场环境: {market_regime}", Colors.INFO)

        # 2. 调整权重
        adjusted_weights = self._adjust_weights_for_regime(market_regime)

        # 3. 计算各类别得分
        category_scores = self._calculate_category_scores(analysis_data)

        # 打印各类别得分
        for cat, score in category_scores.items():
            print_colored(f"{cat} 原始得分: {score:.2f}", Colors.INFO)

        # 4. 检测形态共振
        resonance_bonus = self._detect_pattern_resonance(analysis_data)
        if resonance_bonus > 0:
            print_colored(f"形态共振加成: +{resonance_bonus:.2f}", Colors.GREEN)

        # 5. 多时间框架确认
        mtf_multiplier = self._calculate_mtf_multiplier(analysis_data)
        if mtf_multiplier != 1.0:
            print_colored(f"多时间框架乘数: ×{mtf_multiplier:.2f}", Colors.INFO)

        # 6. 计算最终得分
        final_score = self._compute_final_score(
            category_scores,
            adjusted_weights,
            resonance_bonus,
            mtf_multiplier
        )

        # 7. 生成详细报告
        report = self._generate_score_report(
            final_score,
            category_scores,
            adjusted_weights,
            resonance_bonus,
            mtf_multiplier,
            market_regime
        )

        return report

    def _calculate_technical_score(self, tech_data: Dict) -> float:
        """修复后的技术指标得分计算 - 考虑趋势背景"""
        score = 0

        # 获取趋势信息
        trend_direction = tech_data.get('trend_direction', 'NEUTRAL')
        if trend_direction == 'NEUTRAL':
            trend_info = tech_data.get('trend', {})
            if isinstance(trend_info, dict):
                trend_direction = trend_info.get('direction', 'NEUTRAL')

        # 获取ADX（趋势强度）
        adx = tech_data.get('ADX', tech_data.get('adx', 20))

        # 处理 RSI
        rsi = tech_data.get('RSI', tech_data.get('rsi', 50))

        print_colored(f"\n技术指标详情:", Colors.CYAN)
        print_colored(f"  RSI: {rsi:.1f}, 趋势: {trend_direction}, ADX: {adx:.1f}", Colors.INFO)

        # RSI 评分 - 根据趋势调整
        if trend_direction == 'UP' and adx > 25:  # 强势上涨趋势
            if rsi < 40:
                score += 3.0  # 回调是机会
            elif 40 <= rsi <= 60:
                score += 2.0  # 健康区间
            elif 60 < rsi <= 75:
                score += 1.0  # 仍可接受
            elif 75 < rsi <= 85:
                score += 0.5  # 轻微超买，但趋势向上
            else:
                score -= 1.0  # 极度超买
        elif trend_direction == 'DOWN' and adx > 25:  # 强势下跌趋势
            if rsi > 60:
                score -= 3.0  # 反弹是做空机会
            elif 40 <= rsi <= 60:
                score -= 2.0  # 继续看跌
            elif 25 <= rsi < 40:
                score -= 1.0  # 仍在下跌
            elif 15 <= rsi < 25:
                score -= 0.5  # 轻微超卖，但趋势向下
            else:
                score += 1.0  # 极度超卖可能反弹
        else:  # 震荡市场或弱趋势
            if rsi < 30:
                score += 3.0
            elif rsi < 40:
                score += 1.5
            elif rsi > 70:
                score -= 3.0
            elif rsi > 60:
                score -= 1.5
            else:
                # 中性区域
                if 45 <= rsi <= 55:
                    score += 0
                elif rsi < 45:
                    score += 0.5
                else:
                    score -= 0.5

        # MACD 评分
        macd = tech_data.get('MACD', 0)
        macd_signal = tech_data.get('MACD_signal', 0)

        if macd > macd_signal:
            score += 2.0
            print_colored(f"  MACD: 金叉", Colors.GREEN)
        else:
            score -= 2.0
            print_colored(f"  MACD: 死叉", Colors.RED)

        # 布林带位置 - 考虑趋势
        bb_position = tech_data.get('bb_position', 50)
        print_colored(f"  布林带位置: {bb_position:.1f}%", Colors.INFO)

        if trend_direction == 'UP':
            if bb_position < 30:
                score += 2.0  # 下轨附近是买入机会
            elif bb_position > 90:
                score -= 0.5  # 上轨附近只是轻微警告
            elif 50 <= bb_position <= 80:
                score += 1.0  # 中上部是健康的
        elif trend_direction == 'DOWN':
            if bb_position > 70:
                score -= 2.0  # 上轨附近是做空机会
            elif bb_position < 10:
                score += 0.5  # 下轨附近只是轻微机会
            elif 20 <= bb_position <= 50:
                score -= 1.0  # 中下部继续看跌
        else:
            # 震荡市场使用传统逻辑
            if bb_position < 20:
                score += 2.0
            elif bb_position < 30:
                score += 1.0
            elif bb_position > 80:
                score -= 2.0
            elif bb_position > 70:
                score -= 1.0

        # 成交量确认
        volume_ratio = tech_data.get('volume_ratio', 1.0)
        print_colored(f"  成交量比率: {volume_ratio:.2f}x", Colors.INFO)

        if volume_ratio > 2.0:
            score *= 1.5  # 大幅放量
        elif volume_ratio > 1.3:
            score *= 1.2  # 温和放量
        elif volume_ratio < 0.5:
            score *= 0.6  # 严重缩量

        # 其他指标（威廉指标和CCI）- 考虑趋势
        williams_r = tech_data.get('Williams_R', -50)
        if trend_direction == 'UP':
            if williams_r < -80:
                score += 1.0  # 超卖是机会
            elif williams_r > -20:
                score -= 0.5  # 超买只是轻微警告
        else:
            if williams_r < -80:
                score += 1.0
            elif williams_r > -20:
                score -= 1.0

        cci = tech_data.get('CCI', 0)
        if trend_direction == 'UP':
            if cci < -100:
                score += 1.0  # 超卖是机会
            elif cci > 150:
                score -= 0.5  # 极度超买才警告
        else:
            if cci < -100:
                score += 1.0
            elif cci > 100:
                score -= 1.0

        # 限制得分范围
        final_score = max(-8, min(8, score))
        print_colored(f"  技术指标最终得分: {final_score:.2f}", Colors.YELLOW)

        return final_score


    def _calculate_game_theory_score(self, patterns: List[Dict]) -> float:
        """计算博弈论形态得分 - 更积极"""
        score = 0

        print_colored(f"\n博弈论形态详情:", Colors.CYAN)

        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            confidence = pattern.get('confidence', 0)
            direction = pattern.get('direction', 'NEUTRAL')

            # 获取形态权重
            weight = self.base_weights['patterns'].get(pattern_type, 0.6)

            # 计算单个形态得分 - 提高基础分值
            pattern_score = confidence * weight * 6  # 从5提高到6

            # 根据方向调整
            if direction == 'BULLISH':
                score += pattern_score
                print_colored(f"  {pattern_type}: +{pattern_score:.2f} (置信度: {confidence:.1%})", Colors.GREEN)
            elif direction == 'BEARISH':
                score -= pattern_score
                print_colored(f"  {pattern_type}: -{pattern_score:.2f} (置信度: {confidence:.1%})", Colors.RED)

        final_score = max(-8, min(8, score))
        print_colored(f"  博弈论最终得分: {final_score:.2f}", Colors.YELLOW)

        return final_score

    def _compute_final_score(self, category_scores: Dict, weights: Dict,
                             resonance_bonus: float, mtf_multiplier: float) -> Dict:
        """计算最终得分 - 调整阈值"""
        # 基础加权得分
        weighted_score = sum(category_scores.get(cat, 0) * weights['categories'][cat]
                             for cat in weights['categories'])

        # 应用共振加成
        enhanced_score = weighted_score + resonance_bonus

        # 应用多时间框架乘数
        final_score = enhanced_score * mtf_multiplier

        # 计算置信度（0-1）- 更宽松的映射
        confidence = min(abs(final_score) / 8, 0.95)  # 从10改为8

        # 调整交易阈值 - 更积极
        if final_score > 1.2:  # 从2.0降低到1.2
            action = 'BUY'
            if final_score > 2.5:  # 从4.0降低到2.5
                action = 'STRONG_BUY'
        elif final_score < -1.2:  # 从-2.0提高到-1.2
            action = 'SELL'
            if final_score < -2.5:  # 从-4.0提高到-2.5
                action = 'STRONG_SELL'
        else:
            action = 'HOLD'

        print_colored(f"\n最终评分计算:", Colors.CYAN)
        print_colored(f"  加权得分: {weighted_score:.2f}", Colors.INFO)
        print_colored(f"  共振加成后: {enhanced_score:.2f}", Colors.INFO)
        print_colored(f"  最终得分: {final_score:.2f}", Colors.YELLOW)
        print_colored(f"  交易决策: {action} (置信度: {confidence:.1%})",
                      Colors.GREEN if 'BUY' in action else Colors.RED if 'SELL' in action else Colors.INFO)

        return {
            'final_score': final_score,
            'action': action,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'resonance_bonus': resonance_bonus,
            'mtf_multiplier': mtf_multiplier
        }

    def _detect_market_regime(self, analysis_data: Dict) -> str:
        """检测当前市场环境"""
        indicators = analysis_data.get('technical_indicators', {})

        # 获取关键指标
        adx = indicators.get('ADX', 25)
        atr_ratio = indicators.get('ATR_ratio', 1.0)
        volume_trend = indicators.get('volume_trend', 1.0)

        # 简化的市场环境判断
        if adx > 30:
            return 'TRENDING'
        elif atr_ratio > 1.5:
            return 'VOLATILE'
        elif volume_trend > 1.3:
            return 'BREAKOUT'
        else:
            return 'RANGING'

    def _adjust_weights_for_regime(self, market_regime: str) -> Dict:
        """根据市场环境调整权重"""
        return self.market_regimes.get(market_regime, self.base_weights['categories'])

    def _calculate_category_scores(self, analysis_data: Dict) -> Dict:
        """计算各类别得分"""
        scores = {}

        # 1. 趋势得分
        trend_data = analysis_data.get('trend', {})
        if trend_data:
            scores['trend'] = self._calculate_trend_score(trend_data)
        else:
            scores['trend'] = 0

        # 2. 技术指标得分
        tech_data = analysis_data.get('technical_indicators', {})
        if not tech_data:
            tech_data = analysis_data.get('technical', {})
        if tech_data:
            scores['technical'] = self._calculate_technical_score(tech_data)
        else:
            scores['technical'] = 0

        # 3. 博弈论得分
        game_patterns = analysis_data.get('game_theory_patterns', [])
        if not game_patterns:
            game_theory = analysis_data.get('game_theory', {})
            if game_theory:
                # 转换格式
                game_patterns = self._convert_game_theory_format(game_theory)
        if game_patterns:
            scores['game_theory'] = self._calculate_game_theory_score(game_patterns)
        else:
            scores['game_theory'] = 0

        # 4. 市场结构得分
        market_data = analysis_data.get('market_auction', {})
        if market_data:
            scores['market_structure'] = self._calculate_market_structure_score(market_data)
        else:
            scores['market_structure'] = 0

        return scores

    def _convert_game_theory_format(self, game_theory: Dict) -> List[Dict]:
        """转换博弈论数据格式"""
        patterns = []

        whale_intent = game_theory.get('whale_intent', 'NEUTRAL')
        confidence = game_theory.get('confidence', 0)

        if whale_intent == 'ACCUMULATION':
            patterns.append({
                'type': 'accumulation',
                'confidence': confidence,
                'direction': 'BULLISH'
            })
        elif whale_intent == 'DISTRIBUTION':
            patterns.append({
                'type': 'distribution',
                'confidence': confidence,
                'direction': 'BEARISH'
            })
        elif whale_intent == 'MANIPULATION_UP':
            patterns.append({
                'type': 'stop_hunt',
                'confidence': confidence * 0.8,
                'direction': 'BULLISH'
            })
        elif whale_intent == 'MANIPULATION_DOWN':
            patterns.append({
                'type': 'stop_hunt',
                'confidence': confidence * 0.8,
                'direction': 'BEARISH'
            })

        return patterns

    def _calculate_trend_score(self, trend_data: Dict) -> float:
        """计算趋势得分"""
        score = 0

        # 趋势方向和强度
        direction = trend_data.get('direction', 'NEUTRAL')
        strength = trend_data.get('strength', 0)

        if direction == 'UP':
            score = strength * 6  # 从10降低到6
        elif direction == 'DOWN':
            score = -strength * 6

        # 趋势持续性加分
        duration = trend_data.get('duration', 0)
        if duration > 15:  # 从20降低到15
            score *= 1.2
        elif duration < 5:
            score *= 0.8

        # 趋势质量
        quality = trend_data.get('quality', 0.5)
        score *= (0.5 + quality * 0.5)  # 质量影响降低

        return max(-8, min(8, score))

    def _calculate_market_structure_score(self, market_data: Dict) -> float:
        """计算市场结构得分"""
        score = 0

        # POC相关
        poc_distance = market_data.get('poc_distance', 0)
        if abs(poc_distance) > 0.01:
            score += 1.5 * min(abs(poc_distance) * 100, 1)

        # 价值区域位置
        va_position = market_data.get('value_area_position', 'IN_VALUE')
        if va_position == 'ABOVE_VALUE':
            score += 1.5
        elif va_position == 'BELOW_VALUE':
            score -= 1.5

        # 市场状态
        market_state = market_data.get('market_state', 'BALANCED')
        if market_state == 'TRENDING':
            score *= 1.2
        elif market_state == 'BREAKOUT':
            score *= 1.3

        return max(-8, min(8, score))

    def _detect_pattern_resonance(self, analysis_data: Dict) -> float:
        """检测形态共振"""
        resonance_bonus = 0

        # 获取所有形态
        tech_patterns = analysis_data.get('technical_patterns', [])
        game_patterns = analysis_data.get('game_theory_patterns', [])

        # 检查方向一致性
        bullish_count = 0
        bearish_count = 0

        for p in tech_patterns + game_patterns:
            if p.get('direction') == 'BULLISH':
                bullish_count += 1
            elif p.get('direction') == 'BEARISH':
                bearish_count += 1

        # 共振加成
        if bullish_count >= 3:
            resonance_bonus = min(bullish_count * 0.3, 1.5)
        elif bearish_count >= 3:
            resonance_bonus = -min(bearish_count * 0.3, 1.5)

        return resonance_bonus

    def _calculate_mtf_multiplier(self, analysis_data: Dict) -> float:
        """计算多时间框架确认乘数"""
        # 简化版本，暂时返回1.0
        return 1.0

    def _generate_score_report(self, final_score: Dict, category_scores: Dict,
                               weights: Dict, resonance_bonus: float,
                               mtf_multiplier: float, market_regime: str) -> Dict:
        """生成详细评分报告"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'market_regime': market_regime,
            'final_score': final_score['final_score'],
            'action': final_score['action'],
            'confidence': final_score['confidence'],
            'details': {
                'category_scores': category_scores,
                'adjusted_weights': weights,
                'resonance_bonus': resonance_bonus,
                'mtf_multiplier': mtf_multiplier,
                'weighted_score': final_score['weighted_score']
            }
        }

        return report