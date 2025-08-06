# =====================================================
# 止损猎杀与流动性分析系统
# 核心理念：识别市场流动性聚集点，预测价格运动目标
# =====================================================

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored


class LiquidityHunterSystem:
    """
    流动性猎手系统 - 识别止损和爆仓密集区
    """

    def __init__(self, client, logger=None):
        self.client = client
        self.logger = logger

        # 不同类型交易者的止损习惯
        self.trader_profiles = {
            'scalper': {
                'stop_loss_pct': [0.5, 1.0, 1.5],  # 短线交易者
                'leverage': [20, 50, 75],
                'holding_time': '15m',
                'volume_weight': 0.2
            },
            'day_trader': {
                'stop_loss_pct': [2.0, 3.0, 4.0],  # 日内交易者
                'leverage': [10, 20, 30],
                'holding_time': '4h',
                'volume_weight': 0.5
            },
            'swing_trader': {
                'stop_loss_pct': [5.0, 7.0, 10.0],  # 波段交易者
                'leverage': [3, 5, 10],
                'holding_time': '1d',
                'volume_weight': 0.3
            }
        }

        print_colored("✅ 流动性猎手系统初始化完成", Colors.GREEN)

    def calculate_liquidation_levels(self, symbol: str, current_price: float) -> Dict[str, List[Dict]]:
        """
        计算各个价位的潜在爆仓和止损订单
        """
        try:
            # 获取持仓数据
            position_data = self.get_position_distribution(symbol)
            long_short_ratio = position_data.get('long_short_ratio', 1.0)
            open_interest = position_data.get('open_interest', 0)

            liquidation_levels = {
                'long_liquidations': [],
                'short_liquidations': [],
                'stop_loss_clusters': [],
                'total_liquidity_map': {}
            }

            # 计算多头爆仓位置
            for trader_type, profile in self.trader_profiles.items():
                for leverage in profile['leverage']:
                    for stop_pct in profile['stop_loss_pct']:
                        # 多头爆仓价格 = 入场价 * (1 - 1/杠杆)
                        liquidation_price = current_price * (1 - 1 / leverage)
                        stop_loss_price = current_price * (1 - stop_pct / 100)

                        # 估算该位置的订单量
                        volume_estimate = self.estimate_volume_at_level(
                            open_interest,
                            long_short_ratio,
                            profile['volume_weight'],
                            'long'
                        )

                        liquidation_levels['long_liquidations'].append({
                            'price': liquidation_price,
                            'type': f'{trader_type}_liquidation',
                            'leverage': leverage,
                            'volume': volume_estimate,
                            'distance_pct': (liquidation_price - current_price) / current_price * 100
                        })

                        liquidation_levels['stop_loss_clusters'].append({
                            'price': stop_loss_price,
                            'type': f'{trader_type}_stop_loss',
                            'stop_pct': stop_pct,
                            'volume': volume_estimate * 0.8,  # 止损单通常比爆仓少
                            'distance_pct': (stop_loss_price - current_price) / current_price * 100
                        })

            # 计算空头爆仓位置
            for trader_type, profile in self.trader_profiles.items():
                for leverage in profile['leverage']:
                    for stop_pct in profile['stop_loss_pct']:
                        # 空头爆仓价格 = 入场价 * (1 + 1/杠杆)
                        liquidation_price = current_price * (1 + 1 / leverage)
                        stop_loss_price = current_price * (1 + stop_pct / 100)

                        volume_estimate = self.estimate_volume_at_level(
                            open_interest,
                            long_short_ratio,
                            profile['volume_weight'],
                            'short'
                        )

                        liquidation_levels['short_liquidations'].append({
                            'price': liquidation_price,
                            'type': f'{trader_type}_liquidation',
                            'leverage': leverage,
                            'volume': volume_estimate,
                            'distance_pct': (liquidation_price - current_price) / current_price * 100
                        })

                        liquidation_levels['stop_loss_clusters'].append({
                            'price': stop_loss_price,
                            'type': f'{trader_type}_stop_loss',
                            'stop_pct': stop_pct,
                            'volume': volume_estimate * 0.8,
                            'distance_pct': (stop_loss_price - current_price) / current_price * 100
                        })

            # 聚合相近价位的流动性
            liquidation_levels = self.aggregate_liquidity_clusters(liquidation_levels, current_price)

            # 识别主要流动性池
            liquidation_levels['major_targets'] = self.identify_major_targets(liquidation_levels, current_price)

            return liquidation_levels

        except Exception as e:
            self.logger.error(f"计算爆仓位置失败: {e}")
            return {}

    def get_position_distribution(self, symbol: str) -> Dict:
        """获取持仓分布数据"""
        try:
            # 获取多空比
            long_short_ratio = self.client.futures_global_longshort_ratio(
                symbol=symbol,
                period='5m',
                limit=1
            )[0]

            # 获取持仓量
            open_interest = self.client.futures_open_interest(symbol=symbol)

            # 获取大户持仓（如果可用）
            top_trader_ratio = self.client.futures_top_longshort_ratio(
                symbol=symbol,
                period='5m',
                limit=1
            )[0]

            return {
                'long_short_ratio': float(long_short_ratio['longShortRatio']),
                'open_interest': float(open_interest['openInterest']),
                'top_trader_ratio': float(top_trader_ratio['longShortRatio']),
                'timestamp': long_short_ratio['timestamp']
            }
        except Exception as e:
            self.logger.warning(f"获取持仓数据失败: {e}")
            return {
                'long_short_ratio': 1.0,
                'open_interest': 0,
                'top_trader_ratio': 1.0
            }

    def estimate_volume_at_level(self, open_interest: float, ls_ratio: float,
                                 weight: float, side: str) -> float:
        """估算特定价位的订单量"""
        if side == 'long':
            position_share = ls_ratio / (ls_ratio + 1)
        else:
            position_share = 1 / (ls_ratio + 1)

        return open_interest * position_share * weight

    def aggregate_liquidity_clusters(self, levels: Dict, current_price: float,
                                     cluster_threshold: float = 0.005) -> Dict:
        """聚合相近价位的流动性"""
        # 创建价格到流动性的映射
        liquidity_map = defaultdict(float)

        # 合并所有流动性来源
        all_levels = []
        for long_liq in levels['long_liquidations']:
            all_levels.append(long_liq)
        for short_liq in levels['short_liquidations']:
            all_levels.append(short_liq)
        for stop in levels['stop_loss_clusters']:
            all_levels.append(stop)

        # 按价格排序
        all_levels.sort(key=lambda x: x['price'])

        # 聚合相近价位
        clusters = []
        current_cluster = None

        for level in all_levels:
            if current_cluster is None:
                current_cluster = {
                    'price_min': level['price'],
                    'price_max': level['price'],
                    'price_center': level['price'],
                    'total_volume': level['volume'],
                    'components': [level]
                }
            else:
                # 检查是否在聚类范围内
                price_diff = abs(level['price'] - current_cluster['price_center']) / current_cluster['price_center']

                if price_diff <= cluster_threshold:
                    # 添加到当前聚类
                    current_cluster['price_max'] = max(current_cluster['price_max'], level['price'])
                    current_cluster['price_min'] = min(current_cluster['price_min'], level['price'])
                    current_cluster['total_volume'] += level['volume']
                    current_cluster['components'].append(level)
                    # 重新计算中心价格（加权平均）
                    total_weighted = sum(l['price'] * l['volume'] for l in current_cluster['components'])
                    total_volume = sum(l['volume'] for l in current_cluster['components'])
                    current_cluster['price_center'] = total_weighted / total_volume if total_volume > 0 else \
                    current_cluster['price_center']
                else:
                    # 开始新聚类
                    clusters.append(current_cluster)
                    current_cluster = {
                        'price_min': level['price'],
                        'price_max': level['price'],
                        'price_center': level['price'],
                        'total_volume': level['volume'],
                        'components': [level]
                    }

        if current_cluster:
            clusters.append(current_cluster)

        # 计算每个聚类的强度
        for cluster in clusters:
            cluster['strength'] = cluster['total_volume'] / max(c['total_volume'] for c in clusters)
            cluster['distance_pct'] = (cluster['price_center'] - current_price) / current_price * 100
            cluster['side'] = 'below' if cluster['price_center'] < current_price else 'above'

        levels['liquidity_clusters'] = clusters
        return levels

    def identify_major_targets(self, levels: Dict, current_price: float,
                               min_strength: float = 0.3) -> List[Dict]:
        """识别主要的价格目标"""
        major_targets = []

        # 从聚类中选择强度足够的目标
        for cluster in levels.get('liquidity_clusters', []):
            if cluster['strength'] >= min_strength:
                # 计算吸引力分数
                distance = abs(cluster['distance_pct'])
                volume_score = cluster['strength']

                # 距离越近，吸引力越大
                distance_factor = 1 / (1 + distance * 0.1)

                attraction_score = volume_score * distance_factor

                major_targets.append({
                    'price': cluster['price_center'],
                    'range': [cluster['price_min'], cluster['price_max']],
                    'volume': cluster['total_volume'],
                    'strength': cluster['strength'],
                    'distance_pct': cluster['distance_pct'],
                    'attraction_score': attraction_score,
                    'side': cluster['side'],
                    'description': self.describe_target(cluster)
                })

        # 按吸引力排序
        major_targets.sort(key=lambda x: x['attraction_score'], reverse=True)

        return major_targets[:5]  # 返回前5个目标

    def describe_target(self, cluster: Dict) -> str:
        """描述流动性目标"""
        components = cluster['components']

        # 统计组成
        liquidations = sum(1 for c in components if 'liquidation' in c['type'])
        stop_losses = sum(1 for c in components if 'stop_loss' in c['type'])

        desc_parts = []
        if liquidations > 0:
            desc_parts.append(f"{liquidations}个爆仓位")
        if stop_losses > 0:
            desc_parts.append(f"{stop_losses}个止损位")

        return ' + '.join(desc_parts)

    def analyze_volume_momentum(self, df: pd.DataFrame, liquidity_levels: Dict) -> Dict:
        """
        分析成交量与价格动能的关系
        """
        analysis = {
            'volume_trend': 'neutral',
            'momentum_strength': 0,
            'liquidity_attraction': {},
            'probable_direction': 'neutral',
            'confidence': 0
        }

        try:
            # 计算成交量趋势
            volume_sma = df['volume'].rolling(20).mean()
            recent_volume_ratio = df['volume'].iloc[-5:].mean() / volume_sma.iloc[-1]

            if recent_volume_ratio > 1.5:
                analysis['volume_trend'] = 'expanding'
            elif recent_volume_ratio < 0.7:
                analysis['volume_trend'] = 'contracting'
            else:
                analysis['volume_trend'] = 'normal'

            # 计算价格动能
            momentum = df['close'].pct_change(5).iloc[-1]
            analysis['momentum_strength'] = momentum

            # 分析流动性吸引力
            current_price = df['close'].iloc[-1]
            above_targets = [t for t in liquidity_levels.get('major_targets', []) if t['side'] == 'above']
            below_targets = [t for t in liquidity_levels.get('major_targets', []) if t['side'] == 'below']

            # 计算上下方吸引力
            above_attraction = sum(t['attraction_score'] for t in above_targets[:2]) if above_targets else 0
            below_attraction = sum(t['attraction_score'] for t in below_targets[:2]) if below_targets else 0

            analysis['liquidity_attraction'] = {
                'above': above_attraction,
                'below': below_attraction,
                'ratio': above_attraction / below_attraction if below_attraction > 0 else 0
            }

            # 判断可能的方向
            if analysis['volume_trend'] == 'expanding':
                # 放量情况下，价格倾向于向流动性更多的方向移动
                if above_attraction > below_attraction * 1.2:
                    analysis['probable_direction'] = 'up'
                    analysis['confidence'] = min(0.8, above_attraction / below_attraction * 0.3)
                elif below_attraction > above_attraction * 1.2:
                    analysis['probable_direction'] = 'down'
                    analysis['confidence'] = min(0.8, below_attraction / above_attraction * 0.3)
            elif analysis['volume_trend'] == 'contracting':
                # 缩量整理，可能在积累能量
                analysis['probable_direction'] = 'consolidation'
                analysis['confidence'] = 0.6

            # 结合动能确认
            if momentum > 0.01 and analysis['probable_direction'] == 'up':
                analysis['confidence'] *= 1.2
            elif momentum < -0.01 and analysis['probable_direction'] == 'down':
                analysis['confidence'] *= 1.2

            analysis['confidence'] = min(0.9, analysis['confidence'])

        except Exception as e:
            self.logger.error(f"分析成交量动能失败: {e}")

        return analysis

    def apply_auction_theory(self, liquidity_levels: Dict, volume_analysis: Dict,
                             order_book: Dict) -> Dict:
        """
        应用拍卖理论分析
        包括赢家诅咒等概念
        """
        auction_analysis = {
            'auction_state': 'balanced',
            'winner_curse_risk': 0,
            'optimal_entry': {},
            'market_inefficiency': 0
        }

        try:
            # 分析订单簿不平衡
            bid_volume = sum(order_book.get('bids', {}).values())
            ask_volume = sum(order_book.get('asks', {}).values())

            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

            # 判断拍卖状态
            if abs(imbalance) > 0.3:
                auction_analysis['auction_state'] = 'imbalanced'
                if imbalance > 0:
                    auction_analysis['auction_bias'] = 'buyer_dominated'
                else:
                    auction_analysis['auction_bias'] = 'seller_dominated'

            # 计算赢家诅咒风险
            # 如果市场单边且追高/抄底，风险高
            if volume_analysis['probable_direction'] == 'up' and imbalance > 0.5:
                auction_analysis['winner_curse_risk'] = 0.8  # 买方竞争激烈，可能买贵
            elif volume_analysis['probable_direction'] == 'down' and imbalance < -0.5:
                auction_analysis['winner_curse_risk'] = 0.8  # 卖方竞争激烈，可能卖便宜

            # 寻找最优入场点
            major_targets = liquidity_levels.get('major_targets', [])
            if major_targets:
                nearest_target = major_targets[0]

                # 在流动性目标附近入场，利用止损单的推动
                if nearest_target['side'] == 'above':
                    # 目标在上方，可在略低于目标处入场
                    auction_analysis['optimal_entry'] = {
                        'price': nearest_target['price'] * 0.998,
                        'side': 'BUY',
                        'target': nearest_target['price'],
                        'stop_loss': nearest_target['price'] * 0.99,
                        'reason': '利用上方止损推动'
                    }
                else:
                    # 目标在下方，可在略高于目标处入场
                    auction_analysis['optimal_entry'] = {
                        'price': nearest_target['price'] * 1.002,
                        'side': 'SELL',
                        'target': nearest_target['price'],
                        'stop_loss': nearest_target['price'] * 1.01,
                        'reason': '利用下方止损推动'
                    }

            # 评估市场效率
            # 流动性分布越不均匀，市场越无效，机会越大
            liquidity_variance = np.var([t['strength'] for t in major_targets]) if major_targets else 0
            auction_analysis['market_inefficiency'] = min(1.0, liquidity_variance * 2)

        except Exception as e:
            self.logger.error(f"拍卖理论分析失败: {e}")

        return auction_analysis

    def generate_trading_signal(self, symbol: str) -> Dict:
        """
        综合生成交易信号
        """
        try:
            # 获取市场数据
            df = self.get_market_data(symbol)
            current_price = df['close'].iloc[-1]

            # 获取订单簿
            order_book = self.get_order_book(symbol)

            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"🎯 {symbol} 流动性分析", Colors.CYAN + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            # 1. 计算流动性分布
            print_colored("\n📊 计算止损和爆仓分布...", Colors.CYAN)
            liquidity_levels = self.calculate_liquidation_levels(symbol, current_price)

            # 打印主要目标
            print_colored("\n🎯 主要流动性目标:", Colors.YELLOW)
            for i, target in enumerate(liquidity_levels.get('major_targets', [])[:3], 1):
                direction = "↑" if target['side'] == 'above' else "↓"
                print_colored(
                    f"   {i}. {direction} ${target['price']:.4f} "
                    f"({target['distance_pct']:+.2f}%) "
                    f"强度:{target['strength']:.2f} "
                    f"{target['description']}",
                    Colors.INFO
                )

            # 2. 分析成交量动能
            print_colored("\n📈 分析成交量与动能...", Colors.CYAN)
            volume_analysis = self.analyze_volume_momentum(df, liquidity_levels)

            print_colored(f"   • 成交量状态: {volume_analysis['volume_trend']}", Colors.INFO)
            print_colored(f"   • 价格动能: {volume_analysis['momentum_strength']:.4f}", Colors.INFO)
            print_colored(f"   • 上方吸引力: {volume_analysis['liquidity_attraction']['above']:.2f}", Colors.INFO)
            print_colored(f"   • 下方吸引力: {volume_analysis['liquidity_attraction']['below']:.2f}", Colors.INFO)

            # 3. 应用拍卖理论
            print_colored("\n🔨 应用拍卖理论...", Colors.CYAN)
            auction_analysis = self.apply_auction_theory(liquidity_levels, volume_analysis, order_book)

            print_colored(f"   • 拍卖状态: {auction_analysis['auction_state']}", Colors.INFO)
            print_colored(f"   • 赢家诅咒风险: {auction_analysis['winner_curse_risk']:.1%}", Colors.INFO)
            print_colored(f"   • 市场无效性: {auction_analysis['market_inefficiency']:.1%}", Colors.INFO)

            # 4. 生成最终信号
            signal = self.compile_final_signal(
                liquidity_levels,
                volume_analysis,
                auction_analysis,
                current_price
            )

            # 打印交易建议
            if signal['action'] != 'HOLD':
                print_colored(f"\n💡 交易建议: {signal['action']}",
                              Colors.GREEN if signal['action'] == 'BUY' else Colors.RED)
                print_colored(f"   • 入场价: ${signal['entry_price']:.4f}", Colors.INFO)
                print_colored(f"   • 目标价: ${signal['target_price']:.4f}", Colors.INFO)
                print_colored(f"   • 止损价: ${signal['stop_loss']:.4f}", Colors.INFO)
                print_colored(f"   • 置信度: {signal['confidence']:.1%}", Colors.INFO)
                print_colored(f"   • 理由: {signal['reason']}", Colors.INFO)
            else:
                print_colored(f"\n⏸️ 建议观望", Colors.YELLOW)
                print_colored(f"   • 理由: {signal['reason']}", Colors.INFO)

            return signal

        except Exception as e:
            self.logger.error(f"生成交易信号失败: {e}")
            return {'action': 'HOLD', 'reason': '分析失败'}

    def compile_final_signal(self, liquidity: Dict, volume: Dict,
                             auction: Dict, current_price: float) -> Dict:
        """编译最终交易信号"""
        signal = {
            'action': 'HOLD',
            'confidence': 0,
            'entry_price': current_price,
            'target_price': current_price,
            'stop_loss': current_price,
            'reason': ''
        }

        # 检查是否有清晰的方向
        if volume['probable_direction'] in ['up', 'down'] and volume['confidence'] > 0.5:

            # 检查拍卖理论支持
            if auction.get('optimal_entry'):
                optimal = auction['optimal_entry']

                # 调整置信度
                confidence = volume['confidence']

                # 如果赢家诅咒风险高，降低置信度
                confidence *= (1 - auction['winner_curse_risk'] * 0.5)

                # 如果市场无效性高，增加置信度
                confidence *= (1 + auction['market_inefficiency'] * 0.3)

                # 确保置信度在合理范围
                confidence = max(0.3, min(0.9, confidence))

                if confidence > 0.5:
                    signal.update({
                        'action': optimal['side'],
                        'confidence': confidence,
                        'entry_price': optimal['price'],
                        'target_price': optimal['target'],
                        'stop_loss': optimal['stop_loss'],
                        'reason': optimal['reason']
                    })

        # 如果没有清晰信号，检查是否在震荡
        if signal['action'] == 'HOLD' and volume['volume_trend'] == 'contracting':
            signal['reason'] = '成交量萎缩，等待方向选择'

        return signal

    def get_market_data(self, symbol: str, interval: str = '5m', limit: int = 200) -> pd.DataFrame:
        """获取市场数据"""
        klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """获取订单簿数据"""
        try:
            book = self.client.futures_order_book(symbol=symbol, limit=limit)

            bids = {}
            asks = {}

            for bid in book['bids']:
                price = float(bid[0])
                volume = float(bid[1])
                bids[price] = volume

            for ask in book['asks']:
                price = float(ask[0])
                volume = float(ask[1])
                asks[price] = volume

            return {'bids': bids, 'asks': asks}

        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return {'bids': {}, 'asks': {}}