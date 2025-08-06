"""
增强版市场微观结构博弈论分析系统
真正捕捉庄家意图，结合现货和合约数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from logger_utils import Colors, print_colored


class EnhancedGameTheoryAnalyzer:
    """
    增强版博弈论分析器
    核心功能：
    1. 订单簿深度分析（识别冰山单、支撑阻力）
    2. 现货大单追踪（影响合约价格）
    3. 资金流向分析
    4. 庄家行为模式识别
    5. 技术指标融合
    """

    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('EnhancedGameTheoryAnalyzer')

        # 分析参数
        self.params = {
            'iceberg_threshold': 0.3,      # 冰山单检测阈值
            'whale_order_threshold': 50000, # 大单阈值（USDT）
            'order_book_depth': 20,        # 订单簿深度
            'spot_futures_correlation': 0.8 # 现货期货相关性阈值
        }

        print_colored("✅ 增强版博弈论分析器初始化完成", Colors.GREEN)

    def analyze_market_intent(self, symbol: str, df: pd.DataFrame, depth_data: Dict) -> Dict[str, Any]:
        """
        执行增强的博弈论分析 - 完整替换版本
        """
        print_colored(f"\n🔍 深度分析 {symbol} 市场结构...", Colors.CYAN + Colors.BOLD)

        try:
            # 1. 分析订单簿结构
            order_book_analysis = self._analyze_order_book_structure(depth_data)

            # 2. 追踪大单流向
            spot_flow_analysis = self._track_spot_whale_flow(depth_data)

            # 3. 分析资金费率和持仓
            funding_analysis = self._analyze_funding_and_positions(symbol)

            # 4. 获取技术指标
            technical_context = self._get_technical_context(df)

            # 5. 综合判断市场意图
            market_intent = self._determine_market_intent(
                order_book_analysis,
                spot_flow_analysis,
                funding_analysis,
                technical_context,
                symbol
            )

            return market_intent

        except Exception as e:
            self.logger.error(f"博弈论分析失败 {symbol}: {e}")
            return {
                'manipulation_detected': False,
                'whale_intent': 'NEUTRAL',
                'confidence': 0.5,
                'recommendation': 'HOLD',
                'signals': []
            }

    def _analyze_order_book_structure(self, depth_data: Dict) -> Dict:
        """分析订单簿结构 - 增强版"""
        bids = depth_data.get('bids', [])
        asks = depth_data.get('asks', [])

        if not bids or not asks:
            return {'bid_ask_ratio': 1.0, 'imbalance': 0, 'depth_quality': 0.5}

        # 计算买卖总量
        total_bid_volume = sum(float(bid[1]) for bid in bids[:20])
        total_ask_volume = sum(float(ask[1]) for ask in asks[:20])

        # 计算比率
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0
        imbalance = ((total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) * 100) if (
                                                                                                                         total_bid_volume + total_ask_volume) > 0 else 0

        # 评估订单簿深度质量
        order_count = len(bids) + len(asks)
        if order_count > 30:
            depth_quality = 1.0
        elif order_count > 20:
            depth_quality = 0.8
        elif order_count > 10:
            depth_quality = 0.6
        else:
            depth_quality = 0.4

        print_colored("  📊 分析订单簿结构...", Colors.INFO)
        print_colored(f"    💹 订单簿洞察:", Colors.INFO)
        print_colored(f"      • 买卖压力比: {bid_ask_ratio:.2f}", Colors.INFO)
        print_colored(f"      • 买单量: {len(bids)}", Colors.INFO)
        print_colored(f"      • 卖单量: {len(asks)}", Colors.INFO)
        print_colored(f"      • 订单簿失衡度: {imbalance:.2f}%", Colors.INFO)

        return {
            'bid_ask_ratio': bid_ask_ratio,
            'imbalance': imbalance,
            'bid_count': len(bids),
            'ask_count': len(asks),
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'depth_quality': depth_quality
        }

    def _analyze_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        深度分析订单簿，识别关键特征 - 修复版
        """
        try:
            # 获取订单簿
            order_book = self.client.futures_order_book(symbol=symbol, limit=1000)

            if not order_book:
                return None

            bids = [(float(price), float(qty)) for price, qty in order_book.get('bids', [])]
            asks = [(float(price), float(qty)) for price, qty in order_book.get('asks', [])]

            if not bids or not asks:
                return None

            current_price = (bids[0][0] + asks[0][0]) / 2

            # 分析买卖压力（使用前20档）
            bid_volume = sum(qty for _, qty in bids[:20])
            ask_volume = sum(qty for _, qty in asks[:20])
            pressure_ratio = bid_volume / ask_volume if ask_volume > 0 else 0

            # 检测冰山单（使用新的方法）
            iceberg_orders = self._detect_iceberg_orders(bids, asks)

            # 识别支撑阻力墙
            support_walls = self._find_order_walls(bids, 'support')
            resistance_walls = self._find_order_walls(asks, 'resistance')

            # 计算订单簿失衡度
            imbalance = self._calculate_order_book_imbalance(bids, asks)

            # 分析订单分布
            bid_distribution = self._analyze_order_distribution(bids)
            ask_distribution = self._analyze_order_distribution(asks)

            # 构建分析结果 - 确保包含所有需要的字段
            analysis = {
                'current_price': current_price,
                'pressure_ratio': pressure_ratio,
                'bid_volume': bid_volume,  # 添加这个
                'ask_volume': ask_volume,  # 添加这个
                'bid_volume_20': bid_volume,  # 保持兼容性
                'ask_volume_20': ask_volume,  # 保持兼容性
                'imbalance': imbalance,
                'iceberg_orders': iceberg_orders,
                'support_walls': support_walls,
                'resistance_walls': resistance_walls,
                'bid_distribution': bid_distribution,
                'ask_distribution': ask_distribution,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'mid_price': current_price
            }

            return analysis

        except Exception as e:
            self.logger.error(f"订单簿分析失败: {e}")
            print_colored(f"  ❌ 订单簿分析错误: {str(e)}", Colors.ERROR)
            return None

    def _detect_iceberg_orders_fixed(self, bids: List[Tuple[float, float]],
                                     asks: List[Tuple[float, float]]) -> List[Dict]:
        """检测冰山单 - 修复版（降低误报）"""
        iceberg_orders = []

        # 分析买单
        for i in range(min(10, len(bids))):  # 只检查前10档
            price, qty = bids[i]

            # 检查是否为整数倍的订单量（可能是冰山单）
            if qty > 1000 and qty % 1000 < 10:  # 接近1000的整数倍
                # 检查附近是否有类似订单
                similar_orders = 0
                for j in range(max(0, i - 2), min(len(bids), i + 3)):
                    if j != i and abs(bids[j][1] - qty) / qty < 0.1:  # 相差不到10%
                        similar_orders += 1

                if similar_orders >= 2:  # 至少有2个类似订单才认为是冰山单
                    iceberg_orders.append({
                        'side': 'buy',
                        'price': price,
                        'visible_qty': qty,
                        'estimated_total': qty * 3  # 保守估计总量
                    })

        # 分析卖单（类似逻辑）
        for i in range(min(10, len(asks))):
            price, qty = asks[i]

            if qty > 1000 and qty % 1000 < 10:
                similar_orders = 0
                for j in range(max(0, i - 2), min(len(asks), i + 3)):
                    if j != i and abs(asks[j][1] - qty) / qty < 0.1:
                        similar_orders += 1

                if similar_orders >= 2:
                    iceberg_orders.append({
                        'side': 'sell',
                        'price': price,
                        'visible_qty': qty,
                        'estimated_total': qty * 3
                    })

        return iceberg_orders

    def _identify_support_resistance(self, bids: List[Tuple[float, float]],
                                     asks: List[Tuple[float, float]]) -> Dict:
        """识别支撑和阻力位"""
        support_levels = []
        resistance_levels = []

        # 找出大额买单作为支撑
        if bids:
            avg_bid_size = sum(qty for _, qty in bids) / len(bids)
            for price, qty in bids[:10]:
                if qty > avg_bid_size * 2:  # 大于平均值2倍
                    support_levels.append({
                        'price': price,
                        'strength': qty / avg_bid_size
                    })

        # 找出大额卖单作为阻力
        if asks:
            avg_ask_size = sum(qty for _, qty in asks) / len(asks)
            for price, qty in asks[:10]:
                if qty > avg_ask_size * 2:
                    resistance_levels.append({
                        'price': price,
                        'strength': qty / avg_ask_size
                    })

        return {
            'support': support_levels[:3],  # 最强的3个支撑位
            'resistance': resistance_levels[:3]  # 最强的3个阻力位
        }

    def _detect_iceberg_orders(self, bids: List[Tuple[float, float]],
                               asks: List[Tuple[float, float]]) -> Dict[str, List[Dict]]:
        """
        检测冰山单（隐藏的大额订单）- 改进版

        改进点：
        1. 提高最小订单量门槛
        2. 缩小检查范围
        3. 更严格的相似度判断
        4. 检查算法交易特征
        5. 引入置信度评分
        """
        iceberg_orders = {'buy': [], 'sell': []}

        # === 检测买单中的冰山单 ===
        # 只检查前10档，避免检测太深的订单
        for i in range(min(10, len(bids))):
            price, qty = bids[i]

            # 1. 过滤小额订单（根据您的市场调整此值）
            if qty < 1000:  # 小于1000的订单直接跳过
                continue

            # 2. 检查是否有算法交易特征（整数倍）
            is_round_number = False
            # 检查是否接近1000、500、100的整数倍
            for base in [1000, 500, 100]:
                if qty % base < base * 0.01:  # 误差在1%以内
                    is_round_number = True
                    break

            # 3. 在较小范围内查找相似订单
            similar_orders = []
            total_similar_qty = qty  # 包含当前订单

            # 只检查前后2个价位（共5个价位）
            for j in range(max(0, i - 2), min(len(bids), i + 3)):
                if i == j:  # 跳过自己
                    continue

                other_price, other_qty = bids[j]

                # 相似度判断（更严格）
                qty_diff_ratio = abs(other_qty - qty) / qty

                # 条件1：数量非常接近（5%以内）
                if qty_diff_ratio < 0.05:
                    similar_orders.append({
                        'index': j,
                        'price': other_price,
                        'qty': other_qty,
                        'diff_ratio': qty_diff_ratio
                    })
                    total_similar_qty += other_qty
                # 条件2：倍数关系（2倍或0.5倍）
                elif (abs(other_qty - qty * 2) < qty * 0.05 or
                      abs(other_qty - qty * 0.5) < qty * 0.05):
                    similar_orders.append({
                        'index': j,
                        'price': other_price,
                        'qty': other_qty,
                        'diff_ratio': qty_diff_ratio,
                        'is_multiple': True
                    })
                    total_similar_qty += other_qty

            # 4. 判断是否为冰山单
            # 需要至少3个相似订单（共4个相关订单）
            if len(similar_orders) >= 3:
                # 计算置信度
                confidence = 0.0

                # 基础置信度（根据相似订单数量）
                confidence += min(len(similar_orders) * 0.15, 0.6)

                # 如果是整数倍，增加置信度
                if is_round_number:
                    confidence += 0.2

                # 如果相似订单非常接近（都在3%以内），增加置信度
                if all(order['diff_ratio'] < 0.03 for order in similar_orders):
                    confidence += 0.2

                # 如果总量很大，增加置信度
                if total_similar_qty > qty * 3:
                    confidence += 0.1

                # 限制最大置信度
                confidence = min(confidence, 0.95)

                # 只记录高置信度的冰山单
                if confidence >= 0.5:
                    iceberg_orders['buy'].append({
                        'price': price,
                        'visible_qty': qty,
                        'estimated_total': total_similar_qty,
                        'similar_orders_count': len(similar_orders),
                        'confidence': round(confidence, 2),
                        'pattern': 'algorithmic' if is_round_number else 'manual',
                        'price_range': [
                            bids[max(0, i - 2)][0],  # 最高价
                            bids[min(len(bids) - 1, i + 2)][0]  # 最低价
                        ] if len(bids) > i + 2 else [price, price]
                    })

        # === 检测卖单中的冰山单（逻辑相同）===
        for i in range(min(10, len(asks))):
            price, qty = asks[i]

            if qty < 1000:
                continue

            is_round_number = False
            for base in [1000, 500, 100]:
                if qty % base < base * 0.01:
                    is_round_number = True
                    break

            similar_orders = []
            total_similar_qty = qty

            for j in range(max(0, i - 2), min(len(asks), i + 3)):
                if i == j:
                    continue

                other_price, other_qty = asks[j]
                qty_diff_ratio = abs(other_qty - qty) / qty

                if qty_diff_ratio < 0.05:
                    similar_orders.append({
                        'index': j,
                        'price': other_price,
                        'qty': other_qty,
                        'diff_ratio': qty_diff_ratio
                    })
                    total_similar_qty += other_qty
                elif (abs(other_qty - qty * 2) < qty * 0.05 or
                      abs(other_qty - qty * 0.5) < qty * 0.05):
                    similar_orders.append({
                        'index': j,
                        'price': other_price,
                        'qty': other_qty,
                        'diff_ratio': qty_diff_ratio,
                        'is_multiple': True
                    })
                    total_similar_qty += other_qty

            if len(similar_orders) >= 3:
                confidence = 0.0
                confidence += min(len(similar_orders) * 0.15, 0.6)

                if is_round_number:
                    confidence += 0.2

                if all(order['diff_ratio'] < 0.03 for order in similar_orders):
                    confidence += 0.2

                if total_similar_qty > qty * 3:
                    confidence += 0.1

                confidence = min(confidence, 0.95)

                if confidence >= 0.5:
                    iceberg_orders['sell'].append({
                        'price': price,
                        'visible_qty': qty,
                        'estimated_total': total_similar_qty,
                        'similar_orders_count': len(similar_orders),
                        'confidence': round(confidence, 2),
                        'pattern': 'algorithmic' if is_round_number else 'manual',
                        'price_range': [
                            asks[max(0, i - 2)][0],  # 最低价
                            asks[min(len(asks) - 1, i + 2)][0]  # 最高价
                        ] if len(asks) > i + 2 else [price, price]
                    })

        # 按置信度排序，只返回最相关的
        iceberg_orders['buy'].sort(key=lambda x: x['confidence'], reverse=True)
        iceberg_orders['sell'].sort(key=lambda x: x['confidence'], reverse=True)

        # 可选：限制返回数量
        max_icebergs_per_side = 5  # 每边最多返回5个冰山单
        iceberg_orders['buy'] = iceberg_orders['buy'][:max_icebergs_per_side]
        iceberg_orders['sell'] = iceberg_orders['sell'][:max_icebergs_per_side]

        return iceberg_orders

    def _get_technical_context(self, df: pd.DataFrame) -> Dict:
        """获取技术指标上下文"""
        if df.empty:
            return {'rsi': 50, 'bb_position': 0.5, 'volume_ratio': 1.0}

        latest = df.iloc[-1]

        # 获取RSI
        rsi = latest.get('RSI', 50)

        # 获取布林带位置（修复硬编码问题）
        if 'bb_position' in df.columns:
            bb_position = df['bb_position'].iloc[-1]
        else:
            # 计算布林带位置
            close = latest['close']
            bb_upper = latest.get('bb_upper', close)
            bb_lower = latest.get('bb_lower', close)
            if bb_upper != bb_lower:
                bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5

        # 计算成交量比率
        if 'volume' in df.columns and len(df) >= 20:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # 获取ADX
        adx = latest.get('ADX', 25)

        # 判断技术面状态
        if rsi > 70:
            rsi_state = "OVERBOUGHT"
        elif rsi < 30:
            rsi_state = "OVERSOLD"
        elif rsi > 60:
            rsi_state = "BULLISH"
        elif rsi < 40:
            rsi_state = "BEARISH"
        else:
            rsi_state = "NEUTRAL"

        print_colored("  📈 计算技术指标共振...", Colors.INFO)
        print_colored(f"    📈 技术指标:", Colors.INFO)
        print_colored(f"      • RSI(14): {rsi:.1f} ({rsi_state})", Colors.INFO)
        print_colored(f"      • 布林带位置: {bb_position * 100:.1f}%", Colors.INFO)
        print_colored(f"      • 成交量比率: {volume_ratio:.2f}x", Colors.INFO)
        print_colored(f"      • ADX: {adx:.1f}", Colors.INFO)

        return {
            'rsi': rsi,
            'bb_position': bb_position,
            'volume_ratio': volume_ratio,
            'adx': adx,
            'rsi_state': rsi_state
        }

    def _calculate_order_book_score(self, bid_ask_ratio: float, imbalance: float) -> float:
        """计算订单簿得分"""
        if bid_ask_ratio > 2.5:
            base_score = 8
        elif bid_ask_ratio > 1.5:
            base_score = 5
        elif bid_ask_ratio > 1.2:
            base_score = 2
        elif bid_ask_ratio > 0.8:
            base_score = 0
        elif bid_ask_ratio > 0.5:
            base_score = -2
        elif bid_ask_ratio > 0.3:
            base_score = -5
        else:
            base_score = -8

        # 失衡度调整
        imbalance_adj = min(2, abs(imbalance) / 50)
        if imbalance > 0:
            return min(10, base_score + imbalance_adj)
        else:
            return max(-10, base_score - imbalance_adj)

    def _calculate_funding_score(self, funding_rate: float, ls_ratio: float) -> float:
        """计算资金费率得分"""
        # 资金费率得分
        if abs(funding_rate) < 0.0001:
            fr_score = 0
        elif funding_rate > 0.0003:
            fr_score = -3  # 多头过热，可能回调
        elif funding_rate < -0.0003:
            fr_score = 3  # 空头过热，可能反弹
        else:
            fr_score = -funding_rate * 10000  # 线性映射

        # 多空比得分
        if ls_ratio > 1.5:
            ls_score = -2
        elif ls_ratio < 0.7:
            ls_score = 2
        else:
            ls_score = 0

        return fr_score + ls_score


    def _determine_market_intent(self, order_book: Dict, spot_flow: Dict,
                                 funding: Dict, technical: Dict, symbol: str) -> Dict:
        """
        综合判断市场意图 - 增强版本
        """
        # 获取各项指标
        bid_ask_ratio = order_book['bid_ask_ratio']
        imbalance = order_book['imbalance']
        funding_rate = funding['funding_rate']
        rsi = technical['rsi']
        bb_position = technical['bb_position']
        volume_ratio = technical['volume_ratio']
        adx = technical['adx']

        # 计算各维度得分
        order_score = self._calculate_order_book_score(bid_ask_ratio, imbalance)
        funding_score = self._calculate_funding_score(funding_rate, funding['long_short_ratio'])
        technical_score = self._calculate_technical_score(rsi, bb_position, volume_ratio)

        # 计算综合置信度（加权平均）
        weights = {'order_book': 0.4, 'funding': 0.3, 'technical': 0.3}

        base_confidence = (
                weights['order_book'] * abs(order_score) / 10 +
                weights['funding'] * abs(funding_score) / 10 +
                weights['technical'] * abs(technical_score) / 10
        )

        # 一致性检查
        signals_aligned = (order_score > 0 and technical_score > 0) or (order_score < 0 and technical_score < 0)
        if signals_aligned:
            confidence_multiplier = 1.2
        else:
            confidence_multiplier = 0.8

        # 深度质量调整
        confidence_multiplier *= order_book['depth_quality']

        # 最终置信度（限制在0.3-0.85之间）
        final_confidence = min(0.85, max(0.3, base_confidence * confidence_multiplier))

        # 确定市场阶段和意图
        total_score = order_score + funding_score + technical_score

        # 更丰富的市场状态判断
        if bid_ask_ratio > 2.0 and rsi < 35 and volume_ratio > 1.5:
            market_phase = "恐慌性底部"
            whale_intent = "激进吸筹"
            manipulation_type = "SHAKE_OUT"
        elif bid_ask_ratio > 1.5 and volume_ratio > 1.3:
            market_phase = "积极建仓"
            whale_intent = "吸筹建仓"
            manipulation_type = "ACCUMULATION"
        elif bid_ask_ratio > 1.2 and volume_ratio < 0.8:
            market_phase = "隐秘吸筹"
            whale_intent = "温和吸筹"
            manipulation_type = "STEALTH_BUYING"
        elif bid_ask_ratio < 0.5 and rsi > 65 and volume_ratio > 1.5:
            market_phase = "狂热顶部"
            whale_intent = "激进派发"
            manipulation_type = "DISTRIBUTION"
        elif bid_ask_ratio < 0.8 and volume_ratio > 1.3:
            market_phase = "派发出货"
            whale_intent = "派发出货"
            manipulation_type = "DUMP"
        elif bid_ask_ratio < 0.8 and volume_ratio < 0.8:
            market_phase = "隐秘出货"
            whale_intent = "温和派发"
            manipulation_type = "STEALTH_SELLING"
        elif 0.8 <= bid_ask_ratio <= 1.2:
            if volume_ratio < 0.7:
                market_phase = "低迷震荡"
                whale_intent = "观望等待"
                manipulation_type = "NONE"
            else:
                market_phase = "活跃震荡"
                whale_intent = "区间操作"
                manipulation_type = "RANGE_BOUND"
        else:
            market_phase = "不确定"
            whale_intent = "中性"
            manipulation_type = "UNKNOWN"

        # 应用信号平滑
        smoothed_intent, smoothed_confidence = self._smooth_whale_intent(
            symbol, whale_intent, final_confidence
        )

        # 生成信号列表
        signals = []
        if bid_ask_ratio > 1.5:
            signals.append(f"买压强劲 ({bid_ask_ratio:.2f})")
        elif bid_ask_ratio < 0.67:
            signals.append(f"卖压强劲 ({bid_ask_ratio:.2f})")

        if abs(funding_rate) > 0.0003:
            signals.append(f"资金费率异常 ({funding_rate:.4%})")

        if volume_ratio > 1.5:
            signals.append("成交量放大")
        elif volume_ratio < 0.5:
            signals.append("成交量萎缩")

        if rsi < 30:
            signals.append("RSI超卖")
        elif rsi > 70:
            signals.append("RSI超买")

        # 确定交易建议
        if smoothed_intent in ["激进吸筹", "吸筹建仓"] and smoothed_confidence > 0.6:
            recommendation = "BUY"
            action = "建议买入 🟢"
        elif smoothed_intent in ["激进派发", "派发出货"] and smoothed_confidence > 0.6:
            recommendation = "SELL"
            action = "建议卖出 🔴"
        elif smoothed_intent in ["温和吸筹", "温和派发"] and smoothed_confidence > 0.7:
            recommendation = "WAIT"
            action = "建议观望 ⏸️"
        else:
            recommendation = "HOLD"
            action = "维持现状 ⏹️"

        # 添加庄家信号
        if manipulation_type != "NONE":
            if "吸筹" in smoothed_intent:
                signals.append("🟢 庄家吸筹信号")
            elif "派发" in smoothed_intent:
                signals.append("🔴 庄家派发信号")

        print_colored("  🧠 综合判断市场意图...", Colors.INFO)
        print_colored(f"    🎯 综合判断:", Colors.INFO)
        print_colored(f"      • 市场阶段: {market_phase}", Colors.INFO)
        print_colored(f"      • 庄家意图: {smoothed_intent}", Colors.INFO)
        print_colored(f"      • 置信度: {smoothed_confidence * 100:.1f}%", Colors.INFO)
        print_colored(f"      • 操纵类型: {manipulation_type}", Colors.INFO)
        print_colored(f"      • 交易建议: {action}", Colors.INFO)
        if signals:
            print_colored(f"      • 关键信号:", Colors.INFO)
            for signal in signals:
                print_colored(f"        - {signal}", Colors.INFO)

        return {
            'manipulation_detected': manipulation_type != "NONE",
            'market_phase': market_phase,
            'whale_intent': smoothed_intent,
            'confidence': smoothed_confidence,
            'manipulation_type': manipulation_type,
            'recommendation': recommendation,
            'total_score': total_score,
            'signals': signals
        }

    def _calculate_technical_score(self, rsi: float, bb_position: float, volume_ratio: float) -> float:
        """计算技术指标得分"""
        # RSI得分
        if rsi < 30:
            rsi_score = 3
        elif rsi < 40:
            rsi_score = 1
        elif rsi > 70:
            rsi_score = -3
        elif rsi > 60:
            rsi_score = -1
        else:
            rsi_score = 0

        # 布林带位置得分
        if bb_position < 0.2:
            bb_score = 2
        elif bb_position < 0.3:
            bb_score = 1
        elif bb_position > 0.8:
            bb_score = -2
        elif bb_position > 0.7:
            bb_score = -1
        else:
            bb_score = 0

        # 成交量得分
        if volume_ratio > 2.0:
            volume_score = 2
        elif volume_ratio > 1.5:
            volume_score = 1
        elif volume_ratio < 0.5:
            volume_score = -1
        else:
            volume_score = 0

        return rsi_score + bb_score + volume_score

    def _analyze_funding_and_positions(self, symbol: str) -> Dict:
        """分析资金费率和持仓 - 增强版"""
        try:
            # 获取资金费率
            funding_info = self.client.futures_funding_rate(symbol=symbol, limit=1)
            funding_rate = float(funding_info[0]['fundingRate']) if funding_info else 0

            # 获取持仓信息
            open_interest = self.client.futures_open_interest(symbol=symbol)
            current_oi = float(open_interest['openInterest'])

            # 获取多空比
            try:
                long_short_ratio = self.client.futures_global_longshort_ratio(symbol=symbol, period='5m', limit=1)
                ls_ratio = float(long_short_ratio[0]['longShortRatio']) if long_short_ratio else 1.0
            except:
                ls_ratio = 1.0

            # 计算持仓变化（这里简化处理）
            oi_change = 0  # 实际应该与历史数据比较

            # 判断资金面情绪
            if abs(funding_rate) < 0.0001:
                funding_sentiment = "中性"
                sentiment_emoji = "➖"
            elif funding_rate > 0.0003:
                funding_sentiment = "多头过热"
                sentiment_emoji = "🔥"
            elif funding_rate < -0.0003:
                funding_sentiment = "空头过热"
                sentiment_emoji = "❄️"
            else:
                funding_sentiment = "市场情绪温和"
                sentiment_emoji = "🌡️"

            print_colored("  💰 分析资金费率和持仓...", Colors.INFO)
            print_colored(f"    💰 资金面分析:", Colors.INFO)
            print_colored(f"      • 资金费率: {funding_rate:.4%} ({'LONG' if funding_rate > 0 else 'SHORT'})",
                          Colors.INFO)
            print_colored(f"      • 持仓变化(1h): {oi_change:.1f}%", Colors.INFO)
            print_colored(f"      • 当前持仓: {current_oi:,.0f}", Colors.INFO)
            print_colored(f"      • 多空比: {ls_ratio:.2f}", Colors.INFO)
            print_colored(f"      • 市场情绪: {funding_sentiment} {sentiment_emoji}", Colors.INFO)

            return {
                'funding_rate': funding_rate,
                'oi_change_1h': oi_change,
                'long_short_ratio': ls_ratio,
                'funding_sentiment': funding_sentiment,
                'open_interest': current_oi
            }

        except Exception as e:
            self.logger.error(f"获取资金费率失败: {e}")
            return {
                'funding_rate': 0,
                'oi_change_1h': 0,
                'long_short_ratio': 1.0,
                'funding_sentiment': '未知'
            }

    def _find_order_walls(self, orders: List[Tuple[float, float]],
                         wall_type: str) -> List[Dict[str, Any]]:
        """
        识别订单墙（大额挂单）
        """
        if not orders:
            return []

        # 计算平均订单量
        avg_qty = sum(qty for _, qty in orders[:50]) / min(50, len(orders))

        walls = []
        for price, qty in orders[:20]:  # 只看前20档
            if qty > avg_qty * 5:  # 超过平均值5倍视为墙
                walls.append({
                    'price': price,
                    'quantity': qty,
                    'strength': qty / avg_qty,
                    'type': wall_type
                })

        # 按强度排序
        walls.sort(key=lambda x: x['strength'], reverse=True)
        return walls[:3]  # 返回最强的3个墙

    def _analyze_spot_whale_trades(self, spot_symbol: str) -> Dict[str, Any]:
        """
        分析现货市场的大单交易
        """
        try:
            # 获取最近的成交
            trades = self.client.get_recent_trades(symbol=spot_symbol, limit=1000)

            # 转换为DataFrame便于分析
            df = pd.DataFrame(trades)
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            df['quoteQty'] = df['quoteQty'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')

            # 识别大单
            whale_threshold = self.params['whale_order_threshold']
            df['is_whale'] = df['quoteQty'] > whale_threshold

            # 统计大单买卖
            whale_trades = df[df['is_whale']]

            if len(whale_trades) == 0:
                return {
                    'whale_buy_volume': 0,
                    'whale_sell_volume': 0,
                    'whale_net_flow': 0,
                    'whale_trades_count': 0
                }

            # 判断买卖方向（这里简化处理，实际需要更复杂的逻辑）
            # 使用 .loc 来避免 SettingWithCopyWarning
            whale_trades.loc[:, 'is_buy'] = whale_trades['isBuyerMaker'] == False

            whale_buy_volume = whale_trades[whale_trades['is_buy']]['quoteQty'].sum()
            whale_sell_volume = whale_trades[~whale_trades['is_buy']]['quoteQty'].sum()

            # 计算最近的大单趋势
            recent_whales = whale_trades.tail(10)
            recent_buy_count = len(recent_whales[recent_whales['is_buy']])
            recent_sell_count = len(recent_whales) - recent_buy_count

            analysis = {
                'whale_buy_volume': whale_buy_volume,
                'whale_sell_volume': whale_sell_volume,
                'whale_net_flow': whale_buy_volume - whale_sell_volume,
                'whale_trades_count': len(whale_trades),
                'total_trades_count': len(df),
                'whale_ratio': len(whale_trades) / len(df),
                'recent_whale_trend': 'BUY' if recent_buy_count > recent_sell_count else 'SELL',
                'largest_trades': whale_trades.nlargest(5, 'quoteQty')[['price', 'qty', 'quoteQty', 'is_buy']].to_dict('records')
            }

            return analysis

        except Exception as e:
            self.logger.error(f"现货大单分析失败: {e}")
            return None

    def _analyze_funding_and_oi(self, symbol: str) -> Dict[str, Any]:
        """
        分析资金费率和持仓量变化
        """
        try:
            # 获取资金费率
            funding_rate = self.client.futures_funding_rate(symbol=symbol, limit=1)
            current_funding = float(funding_rate[0]['fundingRate']) if funding_rate else 0

            # 获取持仓量
            oi_stats = self.client.futures_open_interest(symbol=symbol)
            current_oi = float(oi_stats['openInterest'])

            # 获取历史数据对比
            hist_oi = self.client.futures_open_interest_hist(
                symbol=symbol,
                period='5m',
                limit=12  # 1小时数据
            )

            if hist_oi:
                oi_1h_ago = float(hist_oi[0]['sumOpenInterest'])
                oi_change = (current_oi - oi_1h_ago) / oi_1h_ago if oi_1h_ago > 0 else 0
            else:
                oi_change = 0

            # 获取多空比（如果可用）
            try:
                long_short_ratio = self.client.futures_top_longshort_position_ratio(
                    symbol=symbol,
                    period='5m',
                    limit=1
                )
                if long_short_ratio:
                    ls_ratio = float(long_short_ratio[0]['longShortRatio'])
                else:
                    ls_ratio = 1.0
            except:
                ls_ratio = 1.0

            analysis = {
                'funding_rate': current_funding,
                'funding_direction': 'LONG' if current_funding > 0 else 'SHORT',
                'open_interest': current_oi,
                'oi_change_1h': oi_change,
                'long_short_ratio': ls_ratio,
                'market_sentiment': self._interpret_funding_oi(current_funding, oi_change, ls_ratio)
            }

            return analysis

        except Exception as e:
            self.logger.error(f"资金费率分析失败: {e}")
            return None

    def _get_technical_confluence(self, symbol: str) -> Dict[str, Any]:
        """
        获取技术指标共振信号
        """
        try:
            # 获取K线数据
            klines = self.client.futures_klines(symbol=symbol, interval='15m', limit=100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                               'taker_buy_quote', 'ignore'])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # 计算基础技术指标
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']

            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # 成交量分析
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # 获取最新值
            latest = df.iloc[-1]

            # 判断技术信号
            signals = {
                'rsi': latest['rsi'],
                'rsi_signal': 'OVERBOUGHT' if latest['rsi'] > 70 else 'OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL',
                'macd_cross': 'BULLISH' if latest['histogram'] > 0 and df.iloc[-2]['histogram'] <= 0 else
                              'BEARISH' if latest['histogram'] < 0 and df.iloc[-2]['histogram'] >= 0 else 'NONE',
                'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'volume_surge': latest['volume_ratio'] > 2,
                'trend_strength': abs(latest['macd']) / latest['close'] * 100
            }

            # 计算综合技术评分
            tech_score = 0
            if signals['rsi_signal'] == 'OVERSOLD':
                tech_score += 1
            elif signals['rsi_signal'] == 'OVERBOUGHT':
                tech_score -= 1

            if signals['macd_cross'] == 'BULLISH':
                tech_score += 1
            elif signals['macd_cross'] == 'BEARISH':
                tech_score -= 1

            if signals['bb_position'] < 0.2:
                tech_score += 0.5
            elif signals['bb_position'] > 0.8:
                tech_score -= 0.5

            if signals['volume_surge']:
                tech_score = tech_score * 1.5  # 成交量确认

            signals['technical_score'] = tech_score
            signals['current_price'] = latest['close']

            return signals

        except Exception as e:
            self.logger.error(f"技术指标分析失败: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _smooth_whale_intent(self, symbol: str, new_intent: str, new_confidence: float) -> tuple:
        """
        平滑庄家意图信号，避免频繁切换
        """
        # 初始化历史记录
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
            self.last_whale_intent[symbol] = new_intent
            self.intent_change_count[symbol] = 0

        # 记录新信号
        self.signal_history[symbol].append({
            'intent': new_intent,
            'confidence': new_confidence,
            'time': datetime.now()
        })

        # 保持窗口大小
        if len(self.signal_history[symbol]) > self.signal_smoothing_window:
            self.signal_history[symbol].pop(0)

        # 如果历史不足，返回新信号
        if len(self.signal_history[symbol]) < self.signal_smoothing_window:
            return new_intent, new_confidence

        # 统计最近的意图
        recent_intents = [s['intent'] for s in self.signal_history[symbol]]
        intent_counts = {}
        for intent in recent_intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # 找出主导意图
        dominant_intent = max(intent_counts.items(), key=lambda x: x[1])[0]
        dominant_count = intent_counts[dominant_intent]

        # 计算平均置信度
        avg_confidence = np.mean([s['confidence'] for s in self.signal_history[symbol]])

        # 如果新意图与当前不同
        if new_intent != self.last_whale_intent[symbol]:
            # 需要足够的确认才改变
            if dominant_intent == new_intent and dominant_count >= 2:
                self.last_whale_intent[symbol] = new_intent
                self.intent_change_count[symbol] = 0
                return new_intent, avg_confidence
            else:
                # 保持原意图，但降低置信度
                self.intent_change_count[symbol] += 1
                confidence_penalty = 0.9 ** self.intent_change_count[symbol]
                return self.last_whale_intent[symbol], avg_confidence * confidence_penalty
        else:
            # 意图相同，重置计数
            self.intent_change_count[symbol] = 0
            return new_intent, avg_confidence

    def _determine_whale_intent(self, order_book: Dict, spot_flow: Dict,
                               funding: Dict, technical: Dict) -> Dict[str, Any]:
        """
        综合判断庄家意图
        """
        intent_scores = {
            'ACCUMULATION': 0,      # 吸筹
            'DISTRIBUTION': 0,      # 派发
            'MANIPULATION_UP': 0,   # 拉盘操纵
            'MANIPULATION_DOWN': 0, # 砸盘操纵
            'NEUTRAL': 0
        }

        confidence = 0.0
        signals = []

        # 1. 订单簿分析
        if order_book:
            # 买压强于卖压
            if order_book['pressure_ratio'] > 1.5:
                intent_scores['ACCUMULATION'] += 1
                signals.append(f"买压强劲 ({order_book['pressure_ratio']:.2f})")
            elif order_book['pressure_ratio'] < 0.7:
                intent_scores['DISTRIBUTION'] += 1
                signals.append(f"卖压强劲 ({order_book['pressure_ratio']:.2f})")

            # 冰山单分析
            if order_book['iceberg_orders']['buy']:
                intent_scores['ACCUMULATION'] += 1.5
                signals.append(f"发现买方冰山单 ({len(order_book['iceberg_orders']['buy'])}个)")
            if order_book['iceberg_orders']['sell']:
                intent_scores['DISTRIBUTION'] += 1.5
                signals.append(f"发现卖方冰山单 ({len(order_book['iceberg_orders']['sell'])}个)")

            # 订单墙分析
            if order_book['support_walls']:
                strongest_support = order_book['support_walls'][0]
                if strongest_support['strength'] > 10:
                    intent_scores['MANIPULATION_UP'] += 1
                    signals.append(f"强支撑墙 @ ${strongest_support['price']:.4f}")

            if order_book['resistance_walls']:
                strongest_resistance = order_book['resistance_walls'][0]
                if strongest_resistance['strength'] > 10:
                    intent_scores['MANIPULATION_DOWN'] += 1
                    signals.append(f"强阻力墙 @ ${strongest_resistance['price']:.4f}")

        # 2. 现货大单分析
        if spot_flow and spot_flow['whale_trades_count'] > 0:
            net_flow = spot_flow['whale_net_flow']
            if net_flow > 100000:  # 净流入超过10万USDT
                intent_scores['ACCUMULATION'] += 2
                signals.append(f"现货大单净流入 ${net_flow:,.0f}")
            elif net_flow < -100000:
                intent_scores['DISTRIBUTION'] += 2
                signals.append(f"现货大单净流出 ${abs(net_flow):,.0f}")

            # 最近趋势
            if spot_flow['recent_whale_trend'] == 'BUY':
                intent_scores['ACCUMULATION'] += 0.5
            else:
                intent_scores['DISTRIBUTION'] += 0.5

        # 3. 资金费率和持仓分析
        if funding:
            # 资金费率分析
            if abs(funding['funding_rate']) > 0.001:  # 0.1%
                if funding['funding_rate'] > 0:
                    intent_scores['MANIPULATION_UP'] += 0.5
                    signals.append(f"高正资金费率 ({funding['funding_rate']:.4%})")
                else:
                    intent_scores['MANIPULATION_DOWN'] += 0.5
                    signals.append(f"高负资金费率 ({funding['funding_rate']:.4%})")

            # 持仓量变化
            oi_change = funding['oi_change_1h']
            if abs(oi_change) > 0.05:  # 5%变化
                if oi_change > 0:
                    intent_scores['ACCUMULATION'] += 1
                    signals.append(f"持仓量增加 {oi_change:.1%}")
                else:
                    intent_scores['DISTRIBUTION'] += 1
                    signals.append(f"持仓量减少 {abs(oi_change):.1%}")

        # 4. 技术指标验证
        if technical:
            tech_score = technical['technical_score']
            if tech_score > 1:
                intent_scores['ACCUMULATION'] += tech_score * 0.5
                signals.append("技术指标看多")
            elif tech_score < -1:
                intent_scores['DISTRIBUTION'] += abs(tech_score) * 0.5
                signals.append("技术指标看空")

            # RSI极值
            if technical['rsi_signal'] == 'OVERSOLD':
                intent_scores['MANIPULATION_DOWN'] += 0.5
                signals.append(f"RSI超卖 ({technical['rsi']:.1f})")
            elif technical['rsi_signal'] == 'OVERBOUGHT':
                intent_scores['MANIPULATION_UP'] += 0.5
                signals.append(f"RSI超买 ({technical['rsi']:.1f})")

        # 确定最终意图
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        whale_intent = max_intent[0]

        # 计算置信度
        total_score = sum(intent_scores.values())
        if total_score > 0:
            confidence = max_intent[1] / total_score
            # 考虑次高分数，如果太接近则降低置信度
            sorted_scores = sorted(intent_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                score_diff = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                confidence *= (0.5 + score_diff * 0.5)

        return {
            'whale_intent': whale_intent,
            'confidence': confidence,
            'intent_scores': intent_scores,
            'signals': signals
        }

    def _interpret_funding_oi(self, funding_rate: float, oi_change: float, ls_ratio: float) -> str:
        """解释资金费率和持仓变化的含义"""
        if funding_rate > 0.001 and oi_change > 0.05:
            return "BULLISH_MOMENTUM"  # 多头动能强劲
        elif funding_rate < -0.001 and oi_change > 0.05:
            return "SHORT_SQUEEZE_SETUP"  # 可能的空头挤压
        elif funding_rate > 0.001 and oi_change < -0.05:
            return "LONG_LIQUIDATION"  # 多头平仓
        elif funding_rate < -0.001 and oi_change < -0.05:
            return "SHORT_COVERING"  # 空头回补
        else:
            return "NEUTRAL"

    def _calculate_order_book_imbalance(self, bids: List[Tuple[float, float]],
                                       asks: List[Tuple[float, float]]) -> float:
        """计算订单簿失衡度"""
        if not bids or not asks:
            return 0.0

        # 计算不同深度的失衡度
        depths = [5, 10, 20]
        imbalances = []

        for depth in depths:
            bid_sum = sum(qty for _, qty in bids[:depth])
            ask_sum = sum(qty for _, qty in asks[:depth])

            if bid_sum + ask_sum > 0:
                imbalance = (bid_sum - ask_sum) / (bid_sum + ask_sum)
                imbalances.append(imbalance)

        # 加权平均，近端权重更高
        weights = [0.5, 0.3, 0.2]
        weighted_imbalance = sum(w * i for w, i in zip(weights, imbalances))

        return weighted_imbalance

    def _analyze_order_distribution(self, orders: List[Tuple[float, float]]) -> Dict[str, float]:
        """分析订单分布特征"""
        if not orders:
            return {}

        quantities = [qty for _, qty in orders[:50]]

        return {
            'avg_size': np.mean(quantities),
            'median_size': np.median(quantities),
            'std_dev': np.std(quantities),
            'skewness': self._calculate_skewness(quantities),
            'concentration': max(quantities) / sum(quantities) if sum(quantities) > 0 else 0
        }

    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        return np.mean(((data - mean) / std) ** 3)

    # ========== 日志输出方法 ==========

    def _log_order_book_insights(self, analysis: Dict[str, Any]):
        """详细记录订单簿分析结果"""
        print_colored("    💹 订单簿洞察:", Colors.CYAN)

        # 买卖压力
        pressure = analysis['pressure_ratio']
        pressure_color = Colors.GREEN if pressure > 1.2 else Colors.RED if pressure < 0.8 else Colors.YELLOW
        print_colored(f"      • 买卖压力比: {pressure:.2f}", pressure_color)
        print_colored(f"      • 买单量: {analysis['bid_volume_20']:,.0f}", Colors.INFO)
        print_colored(f"      • 卖单量: {analysis['ask_volume_20']:,.0f}", Colors.INFO)

        # 订单簿失衡
        imbalance = analysis['imbalance']
        imb_color = Colors.GREEN if imbalance > 0.1 else Colors.RED if imbalance < -0.1 else Colors.YELLOW
        print_colored(f"      • 订单簿失衡度: {imbalance:.2%}", imb_color)

        # 冰山单
        if analysis['iceberg_orders']['buy'] or analysis['iceberg_orders']['sell']:
            print_colored("      • 🧊 检测到冰山单:", Colors.WARNING)
            for iceberg in analysis['iceberg_orders']['buy'][:2]:
                print_colored(f"        - 买方 @ ${iceberg['price']:.4f} "
                            f"(可见: {iceberg['visible_qty']:,.0f}, "
                            f"预估总量: {iceberg['estimated_total']:,.0f})", Colors.GREEN)
            for iceberg in analysis['iceberg_orders']['sell'][:2]:
                print_colored(f"        - 卖方 @ ${iceberg['price']:.4f} "
                            f"(可见: {iceberg['visible_qty']:,.0f}, "
                            f"预估总量: {iceberg['estimated_total']:,.0f})", Colors.RED)

        # 订单墙
        if analysis['support_walls'] or analysis['resistance_walls']:
            print_colored("      • 🧱 订单墙:", Colors.WARNING)
            for wall in analysis['support_walls'][:1]:
                print_colored(f"        - 支撑墙 @ ${wall['price']:.4f} "
                            f"(数量: {wall['quantity']:,.0f}, 强度: {wall['strength']:.1f}x)", Colors.GREEN)
            for wall in analysis['resistance_walls'][:1]:
                print_colored(f"        - 阻力墙 @ ${wall['price']:.4f} "
                            f"(数量: {wall['quantity']:,.0f}, 强度: {wall['strength']:.1f}x)", Colors.RED)

    def _log_spot_flow_insights(self, analysis: Dict[str, Any]):
        """详细记录现货大单分析结果"""
        if not analysis or analysis.get('whale_trades_count', 0) == 0:
            print_colored("    🐋 现货大单: 无显著活动", Colors.GRAY)
            return

        print_colored("    🐋 现货大单分析:", Colors.CYAN)

        # 净流向
        net_flow = analysis['whale_net_flow']
        flow_color = Colors.GREEN if net_flow > 0 else Colors.RED
        print_colored(f"      • 净流向: {flow_color}${abs(net_flow):,.0f}{Colors.RESET}", Colors.INFO)
        print_colored(f"      • 买入量: ${analysis['whale_buy_volume']:,.0f}", Colors.GREEN)
        print_colored(f"      • 卖出量: ${analysis['whale_sell_volume']:,.0f}", Colors.RED)
        print_colored(f"      • 大单数量: {analysis['whale_trades_count']} "
                     f"({analysis['whale_ratio']:.1%})", Colors.INFO)

        # 最大的几笔交易
        if 'largest_trades' in analysis and analysis['largest_trades']:
            print_colored("      • 最大交易:", Colors.INFO)
            for trade in analysis['largest_trades'][:3]:
                side_color = Colors.GREEN if trade['is_buy'] else Colors.RED
                print_colored(f"        - {side_color}{'买入' if trade['is_buy'] else '卖出'}{Colors.RESET} "
                            f"{trade['qty']:.2f} @ ${trade['price']:.4f} "
                            f"(${trade['quoteQty']:,.0f})", Colors.INFO)

    def _log_funding_insights(self, analysis: Dict[str, Any]):
        """详细记录资金费率分析结果"""
        if not analysis:
            return

        print_colored("    💰 资金面分析:", Colors.CYAN)

        # 资金费率
        funding = analysis['funding_rate']
        funding_color = Colors.RED if abs(funding) > 0.001 else Colors.YELLOW if abs(funding) > 0.0005 else Colors.GREEN
        print_colored(f"      • 资金费率: {funding:.4%} ({analysis['funding_direction']})", funding_color)

        # 持仓量变化
        oi_change = analysis['oi_change_1h']
        oi_color = Colors.GREEN if abs(oi_change) > 0.05 else Colors.YELLOW if abs(oi_change) > 0.02 else Colors.GRAY
        print_colored(f"      • 持仓变化(1h): {oi_change:+.1%}", oi_color)
        print_colored(f"      • 当前持仓: {analysis['open_interest']:,.0f}", Colors.INFO)

        # 多空比
        ls_ratio = analysis['long_short_ratio']
        ls_color = Colors.GREEN if ls_ratio > 1.2 else Colors.RED if ls_ratio < 0.8 else Colors.YELLOW
        print_colored(f"      • 多空比: {ls_ratio:.2f}", ls_color)

        # 市场情绪解读
        sentiment = analysis['market_sentiment']
        sentiment_map = {
            'BULLISH_MOMENTUM': ('多头势头强劲 🚀', Colors.GREEN),
            'SHORT_SQUEEZE_SETUP': ('潜在轧空机会 ⚡', Colors.YELLOW),
            'LONG_LIQUIDATION': ('多头清算中 📉', Colors.RED),
            'SHORT_COVERING': ('空头回补中 📈', Colors.GREEN),
            'NEUTRAL': ('市场情绪中性 ➖', Colors.GRAY)
        }
        sent_text, sent_color = sentiment_map.get(sentiment, ('未知', Colors.GRAY))
        print_colored(f"      • 市场情绪: {sent_text}", sent_color)

    def _log_technical_insights(self, analysis: Dict[str, Any]):
        """详细记录技术指标分析结果"""
        if not analysis:
            return

        print_colored("    📈 技术指标:", Colors.CYAN)

        # RSI
        rsi = analysis['rsi']
        rsi_signal = analysis['rsi_signal']
        rsi_color = Colors.RED if rsi > 70 else Colors.GREEN if rsi < 30 else Colors.YELLOW
        print_colored(f"      • RSI(14): {rsi:.1f} ({rsi_signal})", rsi_color)

        # MACD
        macd_cross = analysis['macd_cross']
        if macd_cross != 'NONE':
            cross_color = Colors.GREEN if macd_cross == 'BULLISH' else Colors.RED
            print_colored(f"      • MACD: {macd_cross} CROSS", cross_color)

        # 布林带位置
        bb_pos = analysis['bb_position']
        bb_color = Colors.RED if bb_pos > 0.9 else Colors.GREEN if bb_pos < 0.1 else Colors.YELLOW
        print_colored(f"      • 布林带位置: {bb_pos:.1%}", bb_color)

        # 成交量
        if analysis['volume_surge']:
            print_colored(f"      • ⚡ 成交量激增 (比率: {analysis.get('volume_ratio', 0):.1f}x)", Colors.WARNING)

        # 技术评分
        tech_score = analysis['technical_score']
        score_color = Colors.GREEN if tech_score > 1 else Colors.RED if tech_score < -1 else Colors.YELLOW
        print_colored(f"      • 技术评分: {tech_score:.1f}", score_color)

    def _log_final_verdict(self, analysis: Dict[str, Any]):
        """输出最终判断结果"""
        print_colored("\n    🎯 综合判断:", Colors.CYAN + Colors.BOLD)

        # 庄家意图
        intent = analysis['whale_intent']
        confidence = analysis['confidence']

        intent_map = {
            'ACCUMULATION': ('吸筹建仓', Colors.GREEN),
            'DISTRIBUTION': ('派发出货', Colors.RED),
            'MANIPULATION_UP': ('拉升操纵', Colors.YELLOW),
            'MANIPULATION_DOWN': ('打压操纵', Colors.YELLOW),
            'NEUTRAL': ('意图不明', Colors.GRAY)
        }

        intent_text, intent_color = intent_map.get(intent, ('未知', Colors.GRAY))
        print_colored(f"      • 庄家意图: {intent_text}", intent_color + Colors.BOLD)
        print_colored(f"      • 置信度: {confidence:.1%}", Colors.INFO)

        # 交易建议
        recommendation = analysis['recommendation']
        rec_map = {
            'BUY': ('建议买入 🟢', Colors.GREEN),
            'SELL': ('建议卖出 🔴', Colors.RED),
            'BUY_CAUTIOUS': ('谨慎做多 ⚠️', Colors.YELLOW),
            'SELL_CAUTIOUS': ('谨慎做空 ⚠️', Colors.YELLOW),
            'HOLD': ('观望等待 ⏸️', Colors.GRAY)
        }

        rec_text, rec_color = rec_map.get(recommendation, ('观望', Colors.GRAY))
        print_colored(f"      • 交易建议: {rec_text}", rec_color + Colors.BOLD)

        # 关键信号
        if analysis.get('signals'):
            print_colored("      • 关键信号:", Colors.INFO)
            for signal in analysis['signals'][:5]:  # 最多显示5个
                print_colored(f"        - {signal}", Colors.INFO)

        # 风险提示
        if analysis.get('risk_factors'):
            print_colored("      • ⚠️ 风险因素:", Colors.WARNING)
            for risk in analysis['risk_factors'][:3]:
                print_colored(f"        - {risk}", Colors.WARNING)