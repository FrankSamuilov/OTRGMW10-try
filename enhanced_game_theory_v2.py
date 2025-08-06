"""
enhanced_game_theory_v2.py
增强版博弈论分析系统 - 整合流动性猎杀和多维度分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import deque
from logger_utils import Colors, print_colored


class LiquidityBasedGameTheory:
    """基于流动性猎杀的博弈论分析系统"""

    def __init__(self):
        self.logger = logging.getLogger('LiquidityGameTheory')
        self.liquidity_zones = {}
        self.stop_hunt_history = deque(maxlen=50)  # 保存最近50次止损猎杀记录

    def analyze_liquidity_landscape(self, df: pd.DataFrame, order_book: Dict = None) -> Dict:
        """
        分析流动性景观 - 识别止损聚集区和突破位
        """
        print_colored("\n🎯 分析流动性景观...", Colors.CYAN)

        analysis = {
            'liquidity_zones': [],
            'stop_hunt_zones': [],
            'true_breakout_levels': [],
            'fake_breakout_risk': 0,
            'entry_recommendations': []
        }

        try:
            # 1. 识别流动性区域（止损聚集地）
            liquidity_zones = self._identify_liquidity_zones(df)
            analysis['liquidity_zones'] = liquidity_zones

            # 2. 检测潜在的止损猎杀区域
            stop_hunt_zones = self._detect_stop_hunt_zones(df, liquidity_zones)
            analysis['stop_hunt_zones'] = stop_hunt_zones

            # 3. 计算真实突破位（避开假突破陷阱）
            true_breakouts = self._calculate_true_breakout_levels(df, liquidity_zones, stop_hunt_zones)
            analysis['true_breakout_levels'] = true_breakouts

            # 4. 评估假突破风险
            analysis['fake_breakout_risk'] = self._assess_fake_breakout_risk(df, order_book)

            # 5. 生成入场建议
            analysis['entry_recommendations'] = self._generate_liquidity_based_entries(
                df, liquidity_zones, true_breakouts
            )

            # 打印分析结果
            self._print_liquidity_analysis(analysis)

        except Exception as e:
            self.logger.error(f"流动性景观分析错误: {e}")

        return analysis

    def _identify_liquidity_zones(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """
        识别流动性区域 - 散户止损聚集的地方
        """
        zones = []

        if len(df) < lookback:
            return zones

        try:
            # 获取最近的价格数据
            recent_df = df.tail(lookback).copy()

            # 1. 识别关键支撑/阻力位（大量止损会放在这里）
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values

            # 找出高成交量的价格区域
            volume_threshold = np.percentile(volumes, 70)
            high_volume_indices = np.where(volumes > volume_threshold)[0]

            for idx in high_volume_indices:
                if idx < len(recent_df) - 1:
                    # 检查是否是关键位置
                    price_level = closes[idx]

                    # 计算该价位附近的反弹次数
                    touch_count = self._count_price_touches(recent_df, price_level, tolerance=0.002)

                    if touch_count >= 2:  # 至少触及2次的位置
                        zone = {
                            'price': price_level,
                            'type': 'support' if price_level < closes[-1] else 'resistance',
                            'strength': min(touch_count / 3, 1.0),  # 强度评分
                            'volume': float(volumes[idx]),
                            'liquidity_score': self._calculate_liquidity_score(
                                price_level, recent_df, volumes[idx]
                            )
                        }
                        zones.append(zone)

            # 2. 识别整数关口（心理价位）
            current_price = df['close'].iloc[-1]
            psychological_levels = self._find_psychological_levels(current_price)

            for level in psychological_levels:
                zones.append({
                    'price': level,
                    'type': 'psychological',
                    'strength': 0.7,
                    'volume': 0,
                    'liquidity_score': 0.6
                })

            # 3. 识别前期高低点
            swing_points = self._find_swing_points(recent_df)
            for point in swing_points:
                zones.append({
                    'price': point['price'],
                    'type': point['type'],
                    'strength': 0.8,
                    'volume': point['volume'],
                    'liquidity_score': 0.7
                })

            # 去重和排序
            zones = self._consolidate_zones(zones)

        except Exception as e:
            self.logger.error(f"识别流动性区域错误: {e}")

        return zones

    def _detect_stop_hunt_zones(self, df: pd.DataFrame, liquidity_zones: List[Dict]) -> List[Dict]:
        """
        检测止损猎杀区域 - 庄家可能攻击的位置
        """
        hunt_zones = []

        try:
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['close'].std()

            for zone in liquidity_zones:
                # 计算距离当前价格的位置
                distance = abs(zone['price'] - current_price)
                distance_in_atr = distance / atr if atr > 0 else 0

                # 如果流动性区域在1-3个ATR范围内，可能成为猎杀目标
                if 0.5 <= distance_in_atr <= 3:
                    hunt_zone = {
                        'target_price': zone['price'],
                        'hunt_type': zone['type'],
                        'hunt_probability': self._calculate_hunt_probability(
                            zone, current_price, df
                        ),
                        'expected_reversal': zone['price'] * (1.002 if zone['type'] == 'support' else 0.998),
                        'risk_level': 'HIGH' if zone['liquidity_score'] > 0.7 else 'MEDIUM'
                    }
                    hunt_zones.append(hunt_zone)

            # 按猎杀概率排序
            hunt_zones.sort(key=lambda x: x['hunt_probability'], reverse=True)

        except Exception as e:
            self.logger.error(f"检测止损猎杀区域错误: {e}")

        return hunt_zones[:5]  # 返回前5个最可能的猎杀区域

    def _calculate_true_breakout_levels(self, df: pd.DataFrame,
                                        liquidity_zones: List[Dict],
                                        stop_hunt_zones: List[Dict]) -> List[Dict]:
        """
        计算真实突破位 - 避开假突破陷阱
        """
        breakout_levels = []

        try:
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['close'].std()

            # 对每个流动性区域，计算真实突破位
            for zone in liquidity_zones:
                if zone['type'] == 'resistance':
                    # 真实突破位 = 阻力位 + 安全缓冲（避免假突破）
                    safety_buffer = atr * 0.5  # 0.5 ATR的缓冲
                    true_breakout = zone['price'] + safety_buffer

                    breakout_levels.append({
                        'level': true_breakout,
                        'direction': 'LONG',
                        'original_resistance': zone['price'],
                        'confidence': self._calculate_breakout_confidence(zone, df),
                        'stop_loss': zone['price'] - atr * 0.3,  # 止损设在阻力下方
                        'take_profit': true_breakout + atr * 2  # 目标2 ATR
                    })

                elif zone['type'] == 'support':
                    # 真实突破位 = 支撑位 - 安全缓冲
                    safety_buffer = atr * 0.5
                    true_breakout = zone['price'] - safety_buffer

                    breakout_levels.append({
                        'level': true_breakout,
                        'direction': 'SHORT',
                        'original_support': zone['price'],
                        'confidence': self._calculate_breakout_confidence(zone, df),
                        'stop_loss': zone['price'] + atr * 0.3,
                        'take_profit': true_breakout - atr * 2
                    })

            # 过滤掉低置信度的突破位
            breakout_levels = [b for b in breakout_levels if b['confidence'] > 0.6]

        except Exception as e:
            self.logger.error(f"计算真实突破位错误: {e}")

        return breakout_levels

    def _generate_liquidity_based_entries(self, df: pd.DataFrame,
                                          liquidity_zones: List[Dict],
                                          breakout_levels: List[Dict]) -> List[Dict]:
        """
        基于流动性分析生成入场建议
        """
        entries = []
        current_price = df['close'].iloc[-1]

        try:
            # 策略1: 止损猎杀反转入场
            for zone in liquidity_zones[:3]:  # 只看最近的3个区域
                if zone['type'] == 'support' and current_price > zone['price']:
                    distance_percent = (current_price - zone['price']) / current_price

                    if 0.001 <= distance_percent <= 0.005:  # 价格接近支撑位
                        entries.append({
                            'strategy': 'STOP_HUNT_REVERSAL',
                            'entry_price': zone['price'] * 1.001,  # 略高于支撑位入场
                            'direction': 'LONG',
                            'confidence': zone['strength'],
                            'stop_loss': zone['price'] * 0.995,
                            'take_profit': current_price * 1.02,
                            'reasoning': f"止损猎杀反转机会 - 支撑位 {zone['price']:.4f}"
                        })

            # 策略2: 真实突破追踪
            for breakout in breakout_levels[:2]:  # 最多2个突破机会
                if breakout['direction'] == 'LONG' and current_price < breakout['level']:
                    distance_to_breakout = (breakout['level'] - current_price) / current_price

                    if distance_to_breakout <= 0.003:  # 接近突破位
                        entries.append({
                            'strategy': 'TRUE_BREAKOUT',
                            'entry_price': breakout['level'],
                            'direction': 'LONG',
                            'confidence': breakout['confidence'],
                            'stop_loss': breakout['stop_loss'],
                            'take_profit': breakout['take_profit'],
                            'reasoning': f"真实突破机会 - 突破位 {breakout['level']:.4f}"
                        })

        except Exception as e:
            self.logger.error(f"生成入场建议错误: {e}")

        return entries

    def _calculate_liquidity_score(self, price_level: float, df: pd.DataFrame, volume: float) -> float:
        """计算流动性评分"""
        try:
            # 基于成交量和价格触及次数
            touch_count = self._count_price_touches(df, price_level)
            volume_percentile = (volume - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

            score = (touch_count * 0.3 + volume_percentile * 0.7)
            return min(score, 1.0)
        except:
            return 0.5

    def _count_price_touches(self, df: pd.DataFrame, price_level: float, tolerance: float = 0.002) -> int:
        """计算价格触及次数"""
        count = 0
        for _, row in df.iterrows():
            if abs(row['high'] - price_level) / price_level <= tolerance or \
                    abs(row['low'] - price_level) / price_level <= tolerance:
                count += 1
        return count

    def _find_psychological_levels(self, current_price: float) -> List[float]:
        """找出心理价位（整数关口）"""
        levels = []

        # 根据价格范围确定步长
        if current_price < 1:
            step = 0.01
        elif current_price < 10:
            step = 0.1
        elif current_price < 100:
            step = 1
        else:
            step = 10

        # 找出附近的整数关口
        base = (current_price // step) * step
        for i in range(-2, 3):
            level = base + i * step
            if 0.95 * current_price <= level <= 1.05 * current_price:
                levels.append(level)

        return levels

    def _find_swing_points(self, df: pd.DataFrame, window: int = 10) -> List[Dict]:
        """找出摆动高低点"""
        points = []

        if len(df) < window * 2:
            return points

        for i in range(window, len(df) - window):
            # 检查高点
            if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
                points.append({
                    'price': df['high'].iloc[i],
                    'type': 'swing_high',
                    'volume': df['volume'].iloc[i]
                })

            # 检查低点
            if df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
                points.append({
                    'price': df['low'].iloc[i],
                    'type': 'swing_low',
                    'volume': df['volume'].iloc[i]
                })

        return points

    def _consolidate_zones(self, zones: List[Dict]) -> List[Dict]:
        """合并相近的区域"""
        if not zones:
            return zones

        # 按价格排序
        zones.sort(key=lambda x: x['price'])

        consolidated = []
        current_zone = zones[0]

        for zone in zones[1:]:
            # 如果价格相近（0.5%以内），合并
            if abs(zone['price'] - current_zone['price']) / current_zone['price'] < 0.005:
                # 取更强的信号
                if zone['strength'] > current_zone['strength']:
                    current_zone = zone
            else:
                consolidated.append(current_zone)
                current_zone = zone

        consolidated.append(current_zone)
        return consolidated

    def _calculate_hunt_probability(self, zone: Dict, current_price: float, df: pd.DataFrame) -> float:
        """计算止损猎杀概率"""
        probability = 0.5  # 基础概率

        try:
            # 流动性越高，越容易被猎杀
            probability += zone['liquidity_score'] * 0.2

            # 距离越近，越容易被猎杀
            distance = abs(zone['price'] - current_price) / current_price
            if distance < 0.01:
                probability += 0.2
            elif distance < 0.02:
                probability += 0.1

            # 如果是心理价位，增加概率
            if zone['type'] == 'psychological':
                probability += 0.1

            return min(probability, 0.95)
        except:
            return 0.5

    def _calculate_breakout_confidence(self, zone: Dict, df: pd.DataFrame) -> float:
        """计算突破置信度"""
        confidence = 0.5

        try:
            # 基于区域强度
            confidence += zone['strength'] * 0.3

            # 基于成交量
            if zone['volume'] > df['volume'].mean():
                confidence += 0.2

            return min(confidence, 1.0)
        except:
            return 0.5

    def _assess_fake_breakout_risk(self, df: pd.DataFrame, order_book: Dict = None) -> float:
        """评估假突破风险"""
        risk = 0.3  # 基础风险

        try:
            # 检查最近是否有假突破
            recent_fakeouts = self._detect_recent_fakeouts(df)
            risk += len(recent_fakeouts) * 0.1

            # 如果订单簿不平衡，增加风险
            if order_book:
                imbalance = self._calculate_order_book_imbalance(order_book)
                if abs(imbalance) > 0.3:
                    risk += 0.2

            return min(risk, 0.9)
        except:
            return 0.5

    def _detect_recent_fakeouts(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """检测最近的假突破"""
        fakeouts = []

        if len(df) < lookback:
            return fakeouts

        # 简化的假突破检测逻辑
        for i in range(len(df) - lookback, len(df) - 2):
            # 检查是否突破后快速回落
            if df['high'].iloc[i] > df['high'].iloc[i - 1] and \
                    df['close'].iloc[i + 1] < df['open'].iloc[i]:
                fakeouts.append({'index': i, 'type': 'bull_trap'})

            if df['low'].iloc[i] < df['low'].iloc[i - 1] and \
                    df['close'].iloc[i + 1] > df['open'].iloc[i]:
                fakeouts.append({'index': i, 'type': 'bear_trap'})

        return fakeouts

    def _calculate_order_book_imbalance(self, order_book: Dict) -> float:
        """计算订单簿失衡度"""
        try:
            bid_volume = sum(order_book.get('bid_sizes', [])[:5])
            ask_volume = sum(order_book.get('ask_sizes', [])[:5])

            if bid_volume + ask_volume > 0:
                return (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return 0
        except:
            return 0

    def _print_liquidity_analysis(self, analysis: Dict):
        """打印流动性分析结果"""
        print_colored("\n📊 流动性分析结果:", Colors.CYAN)

        # 流动性区域
        if analysis['liquidity_zones']:
            print_colored("  💧 流动性区域:", Colors.INFO)
            for zone in analysis['liquidity_zones'][:3]:
                print_colored(f"    • {zone['type'].upper()} @ {zone['price']:.4f} "
                              f"(强度: {zone['strength']:.2f})", Colors.GRAY)

        # 止损猎杀区域
        if analysis['stop_hunt_zones']:
            print_colored("  🎯 潜在猎杀区域:", Colors.WARNING)
            for hunt in analysis['stop_hunt_zones'][:2]:
                print_colored(f"    • {hunt['target_price']:.4f} "
                              f"(概率: {hunt['hunt_probability']:.1%})", Colors.YELLOW)

        # 真实突破位
        if analysis['true_breakout_levels']:
            print_colored("  🚀 真实突破位:", Colors.SUCCESS)
            for breakout in analysis['true_breakout_levels'][:2]:
                print_colored(f"    • {breakout['direction']} @ {breakout['level']:.4f} "
                              f"(置信度: {breakout['confidence']:.1%})", Colors.GREEN)

        # 入场建议
        if analysis['entry_recommendations']:
            print_colored("  📍 入场建议:", Colors.CYAN)
            for entry in analysis['entry_recommendations'][:2]:
                print_colored(f"    • {entry['strategy']}: {entry['direction']} @ {entry['entry_price']:.4f}",
                              Colors.BLUE)
                print_colored(f"      理由: {entry['reasoning']}", Colors.GRAY)


class EnhancedOrderBookAnalyzer:
    """增强的订单簿分析器 - 降低噪音，提高稳定性"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.order_book_history = deque(maxlen=window_size)
        self.logger = logging.getLogger('OrderBookAnalyzer')

    def add_snapshot(self, order_book: Dict):
        """添加订单簿快照"""
        if order_book:
            self.order_book_history.append({
                'timestamp': datetime.now(),
                'data': order_book
            })

    def get_smoothed_analysis(self) -> Dict:
        """
        获取平滑后的订单簿分析
        使用移动平均降低噪音
        """
        if len(self.order_book_history) < 3:
            return {'buy_sell_ratio': 1.0, 'confidence': 0}

        ratios = []
        imbalances = []

        for snapshot in self.order_book_history:
            ob = snapshot['data']
            bid_volume = sum(ob.get('bid_sizes', [])[:10])
            ask_volume = sum(ob.get('ask_sizes', [])[:10])

            if bid_volume + ask_volume > 0:
                ratio = bid_volume / (bid_volume + ask_volume)
                ratios.append(ratio)

                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                imbalances.append(imbalance)

        # 使用加权移动平均（最新数据权重更高）
        weights = np.linspace(0.5, 1.0, len(ratios))
        weights = weights / weights.sum()

        weighted_ratio = np.average(ratios, weights=weights)
        weighted_imbalance = np.average(imbalances, weights=weights)

        # 计算稳定性（标准差越小越稳定）
        stability = 1 - min(np.std(ratios), 0.5) / 0.5

        return {
            'buy_sell_ratio': weighted_ratio,
            'imbalance': weighted_imbalance,
            'confidence': stability,
            'trend': self._detect_trend(ratios),
            'raw_ratios': ratios  # 用于调试
        }

    def _detect_trend(self, ratios: List[float]) -> str:
        """检测买卖比例趋势"""
        if len(ratios) < 3:
            return 'NEUTRAL'

        # 简单线性回归
        x = np.arange(len(ratios))
        slope = np.polyfit(x, ratios, 1)[0]

        if slope > 0.02:
            return 'BULLISH'
        elif slope < -0.02:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


class IntegratedTradingDecisionSystem:
    """
    整合的交易决策系统
    结合流动性猎杀、多层指标和智能止损
    """

    def __init__(self):
        self.liquidity_analyzer = LiquidityBasedGameTheory()
        self.order_book_analyzer = EnhancedOrderBookAnalyzer()
        self.logger = logging.getLogger('TradingDecision')

    def make_comprehensive_decision(self, market_data: Dict) -> Dict:
        """
        做出综合交易决策
        """
        decision = {
            'action': 'HOLD',
            'confidence': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0,
            'reasoning': [],
            'trade_plan': None
        }

        try:
            df = market_data.get('kline_data')
            if df is None or len(df) < 50:
                return decision

            print_colored("\n🤖 执行综合交易决策分析...", Colors.CYAN)

            # 1. 流动性分析
            liquidity_analysis = self.liquidity_analyzer.analyze_liquidity_landscape(
                df, market_data.get('order_book')
            )

            # 2. 订单簿分析（平滑版本）
            if market_data.get('order_book'):
                self.order_book_analyzer.add_snapshot(market_data['order_book'])
            order_book_analysis = self.order_book_analyzer.get_smoothed_analysis()

            # 3. 技术指标分析（多层）
            technical_analysis = self._layered_technical_analysis(df)

            # 4. 综合决策
            decision = self._synthesize_decision(
                liquidity_analysis,
                order_book_analysis,
                technical_analysis,
                df
            )

            # 5. 如果有交易信号，创建详细计划
            if decision['action'] != 'HOLD':
                decision['trade_plan'] = self._create_detailed_trade_plan(
                    decision, liquidity_analysis, df
                )

            self._print_decision(decision)

        except Exception as e:
            self.logger.error(f"综合决策错误: {e}")
            decision['reasoning'].append(f"分析错误: {str(e)}")

        return decision

    def _layered_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """多层技术分析"""
        analysis = {
            'trend': {'direction': 'NEUTRAL', 'strength': 0},
            'momentum': {'status': 'NEUTRAL', 'strength': 0},
            'entry_timing': {'ready': False, 'score': 0}
        }

        try:
            # 趋势层（慢速指标）
            if all(col in df.columns for col in ['EMA20', 'EMA50', 'ADX']):
                ema20 = df['EMA20'].iloc[-1]
                ema50 = df['EMA50'].iloc[-1]
                price = df['close'].iloc[-1]
                adx = df['ADX'].iloc[-1] if not pd.isna(df['ADX'].iloc[-1]) else 0

                if price > ema20 > ema50 and adx > 25:
                    analysis['trend'] = {'direction': 'UP', 'strength': min(adx / 50, 1.0)}
                elif price < ema20 < ema50 and adx > 25:
                    analysis['trend'] = {'direction': 'DOWN', 'strength': min(adx / 50, 1.0)}

            # 动量层（中速指标）
            if all(col in df.columns for col in ['RSI', 'MACD']):
                rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
                macd = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0

                if rsi > 50 and macd > 0:
                    analysis['momentum'] = {'status': 'BULLISH', 'strength': (rsi - 50) / 50}
                elif rsi < 50 and macd < 0:
                    analysis['momentum'] = {'status': 'BEARISH', 'strength': (50 - rsi) / 50}

            # 入场时机层（快速指标）
            if 'Williams_%R' in df.columns:
                williams = df['Williams_%R'].iloc[-1] if not pd.isna(df['Williams_%R'].iloc[-1]) else -50

                if -30 < williams < -20:  # 超买区域附近
                    analysis['entry_timing'] = {'ready': True, 'score': 0.8}
                elif -80 < williams < -70:  # 超卖区域附近
                    analysis['entry_timing'] = {'ready': True, 'score': 0.8}

        except Exception as e:
            self.logger.error(f"技术分析错误: {e}")

        return analysis

    def _synthesize_decision(self, liquidity: Dict, order_book: Dict,
                             technical: Dict, df: pd.DataFrame) -> Dict:
        """综合所有分析做出决策"""
        decision = {
            'action': 'HOLD',
            'confidence': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0,
            'reasoning': []
        }

        try:
            current_price = df['close'].iloc[-1]
            signals = []

            # 1. 检查流动性入场机会
            if liquidity.get('entry_recommendations'):
                best_entry = liquidity['entry_recommendations'][0]
                signals.append({
                    'type': 'liquidity',
                    'action': best_entry['direction'],
                    'confidence': best_entry['confidence'],
                    'entry': best_entry['entry_price'],
                    'stop': best_entry['stop_loss'],
                    'target': best_entry['take_profit'],
                    'reason': best_entry['reasoning']
                })

            # 2. 检查技术面信号
            if technical['trend']['direction'] != 'NEUTRAL' and \
                    technical['momentum']['status'] != 'NEUTRAL' and \
                    technical['entry_timing']['ready']:

                # 趋势和动量一致
                if (technical['trend']['direction'] == 'UP' and
                        technical['momentum']['status'] == 'BULLISH'):

                    signals.append({
                        'type': 'technical',
                        'action': 'LONG',
                        'confidence': technical['trend']['strength'] * 0.7,
                        'entry': current_price,
                        'stop': current_price * 0.98,
                        'target': current_price * 1.03,
                        'reason': '技术面多头信号确认'
                    })

                elif (technical['trend']['direction'] == 'DOWN' and
                      technical['momentum']['status'] == 'BEARISH'):

                    signals.append({
                        'type': 'technical',
                        'action': 'SHORT',
                        'confidence': technical['trend']['strength'] * 0.7,
                        'entry': current_price,
                        'stop': current_price * 1.02,
                        'target': current_price * 0.97,
                        'reason': '技术面空头信号确认'
                    })

            # 3. 检查订单簿信号（权重较低）
            if order_book.get('confidence', 0) > 0.7:
                if order_book['imbalance'] > 0.2 and order_book['trend'] == 'BULLISH':
                    signals.append({
                        'type': 'orderbook',
                        'action': 'LONG',
                        'confidence': order_book['confidence'] * 0.5,  # 降低权重
                        'entry': current_price,
                        'stop': current_price * 0.99,
                        'target': current_price * 1.02,
                        'reason': '订单簿显示买压强劲'
                    })

            # 4. 选择最佳信号
            if signals:
                # 按置信度排序
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                best_signal = signals[0]

                # 只有置信度足够高才交易
                if best_signal['confidence'] > 0.6:
                    decision['action'] = best_signal['action']
                    decision['confidence'] = best_signal['confidence']
                    decision['entry_price'] = best_signal['entry']
                    decision['stop_loss'] = best_signal['stop']
                    decision['take_profit'] = best_signal['target']
                    decision['position_size'] = min(best_signal['confidence'] * 0.3, 0.2)  # 最大20%仓位
                    decision['reasoning'].append(best_signal['reason'])

                    # 添加其他支持信号
                    for signal in signals[1:3]:
                        if signal['confidence'] > 0.5:
                            decision['reasoning'].append(f"额外支持: {signal['reason']}")
                else:
                    decision['reasoning'].append('信号强度不足，继续等待')
            else:
                decision['reasoning'].append('无有效交易信号')

        except Exception as e:
            self.logger.error(f"综合决策错误: {e}")
            decision['reasoning'].append(f"决策错误: {str(e)}")

        return decision

    def _create_detailed_trade_plan(self, decision: Dict,
                                    liquidity: Dict, df: pd.DataFrame) -> Dict:
        """创建详细的交易计划"""
        plan = {
            'entry_strategy': {
                'method': 'LIMIT',
                'primary_entry': decision['entry_price'],
                'scale_in_levels': [],
                'max_slippage': 0.002
            },
            'position_management': {
                'initial_size': decision['position_size'],
                'max_size': decision['position_size'] * 2,
                'scaling_plan': 'PYRAMID'  # 金字塔加仓
            },
            'risk_management': {
                'stop_loss': decision['stop_loss'],
                'trailing_stop': True,
                'trail_distance': 0.005,
                'max_loss': 0.02  # 最大损失2%
            },
            'profit_taking': {
                'target_1': decision['take_profit'] * 0.7,  # 第一目标
                'target_2': decision['take_profit'],  # 第二目标
                'target_3': decision['take_profit'] * 1.5,  # 第三目标
                'partial_exits': [0.3, 0.3, 0.4]  # 分批止盈比例
            },
            'time_management': {
                'max_holding_period': '24h',
                'review_interval': '1h',
                'force_exit_time': None
            },
            'contingency': {
                'if_stopped_out': 'WAIT_30MIN',
                'if_profit_target_hit': 'LOOK_FOR_CONTINUATION',
                'if_sideways': 'EXIT_AFTER_4H'
            }
        }

        # 根据流动性分析调整计划
        if liquidity.get('stop_hunt_zones'):
            hunt_zone = liquidity['stop_hunt_zones'][0]
            plan['risk_management']['stop_hunt_protection'] = {
                'danger_zone': hunt_zone['target_price'],
                'adjusted_stop': hunt_zone['expected_reversal']
            }

        return plan

    def _print_decision(self, decision: Dict):
        """打印交易决策"""
        print_colored("\n📊 交易决策:", Colors.CYAN)

        if decision['action'] == 'HOLD':
            print_colored(f"  ⏸️ 决策: 观望", Colors.YELLOW)
        else:
            color = Colors.GREEN if decision['action'] in ['BUY', 'LONG'] else Colors.RED
            print_colored(f"  {'🔼' if decision['action'] in ['BUY', 'LONG'] else '🔽'} "
                          f"决策: {decision['action']}", color)
            print_colored(f"  💯 置信度: {decision['confidence']:.1%}", Colors.INFO)
            print_colored(f"  💰 仓位: {decision['position_size']:.1%}", Colors.INFO)
            print_colored(f"  📍 入场: {decision['entry_price']:.4f}", Colors.INFO)
            print_colored(f"  🛑 止损: {decision['stop_loss']:.4f}", Colors.WARNING)
            print_colored(f"  🎯 止盈: {decision['take_profit']:.4f}", Colors.SUCCESS)

        if decision['reasoning']:
            print_colored("  📝 理由:", Colors.GRAY)
            for reason in decision['reasoning'][:3]:
                print_colored(f"    • {reason}", Colors.GRAY)

        if decision.get('trade_plan'):
            print_colored("  📋 交易计划已生成", Colors.BLUE)