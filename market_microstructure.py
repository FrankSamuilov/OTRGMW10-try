"""
市场微观结构分析模块
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

from logger_utils import Colors, print_colored


class OrderFlowToxicityAnalyzer:
    """订单流毒性分析器 - 识别知情交易者的痕迹"""

    def __init__(self):
        self.toxicity_threshold = 0.7
        self.lookback_window = 100
        self.logger = logging.getLogger('OrderFlowToxicity')

    def calculate_vpin(self, df, bucket_size=50):
        """计算VPIN（Volume-synchronized Probability of Informed Trading）"""

        vpin_result = {
            'vpin': 0,
            'toxicity_level': 'UNKNOWN',
            'informed_trading_probability': 0,
            'recommendation': '无法计算VPIN'
        }

        if df is None or len(df) < bucket_size:
            return vpin_result

        try:
            # 将成交量分成等量的桶
            total_volume = df['volume'].sum()
            if total_volume == 0:
                return vpin_result

            bucket_volume = total_volume / bucket_size

            buckets = []
            current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total': 0}

            for idx in range(len(df)):
                # 使用tick rule判断买卖方向
                if idx > 0:
                    if df.iloc[idx]['close'] > df.iloc[idx - 1]['close']:
                        current_bucket['buy_volume'] += df.iloc[idx]['volume']
                    else:
                        current_bucket['sell_volume'] += df.iloc[idx]['volume']

                current_bucket['total'] += df.iloc[idx]['volume']

                # 当桶满时，计算该桶的不平衡度
                if current_bucket['total'] >= bucket_volume:
                    imbalance = abs(current_bucket['buy_volume'] - current_bucket['sell_volume'])
                    if current_bucket['total'] > 0:
                        imbalance_ratio = imbalance / current_bucket['total']
                    else:
                        imbalance_ratio = 0

                    buckets.append({
                        'imbalance': imbalance,
                        'total_volume': current_bucket['total'],
                        'imbalance_ratio': imbalance_ratio
                    })
                    current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total': 0}

            # 计算VPIN
            if len(buckets) >= 50:
                recent_buckets = buckets[-50:]
                vpin = np.mean([b['imbalance_ratio'] for b in recent_buckets])

                vpin_result = {
                    'vpin': vpin,
                    'toxicity_level': 'HIGH' if vpin > 0.3 else 'MEDIUM' if vpin > 0.2 else 'LOW',
                    'informed_trading_probability': min(1.0, vpin * 3),
                    'recommendation': self._get_vpin_recommendation(vpin)
                }

        except Exception as e:
            self.logger.error(f"VPIN计算错误: {e}")

        return vpin_result

    def _get_vpin_recommendation(self, vpin):
        """基于VPIN值给出建议"""
        if vpin > 0.35:
            return "极高毒性，可能有内幕交易，避免交易"
        elif vpin > 0.25:
            return "高毒性，庄家可能在行动，谨慎跟随"
        elif vpin > 0.15:
            return "中等毒性，正常市场，可以交易"
        else:
            return "低毒性，市场平静，适合建仓"

    def analyze_trade_informativeness(self, trades):
        """分析交易的信息含量"""

        informativeness = {
            'price_discovery_contribution': 0,
            'trade_size_distribution': {},
            'informed_trade_probability': 0,
            'large_trade_impact': 0
        }

        if not trades or len(trades) < 20:
            return informativeness

        try:
            # 分析交易大小分布
            sizes = [float(t.get('qty', 0)) for t in trades]
            prices = [float(t.get('price', 0)) for t in trades]

            if sizes and prices:
                # 计算大中小单的比例
                size_percentiles = np.percentile(sizes, [25, 75])
                small_threshold = size_percentiles[0]
                large_threshold = size_percentiles[1]

                small_trades = sum(1 for s in sizes if s <= small_threshold)
                medium_trades = sum(1 for s in sizes if small_threshold < s <= large_threshold)
                large_trades = sum(1 for s in sizes if s > large_threshold)

                total_trades = len(sizes)
                informativeness['trade_size_distribution'] = {
                    'small': small_trades / total_trades,
                    'medium': medium_trades / total_trades,
                    'large': large_trades / total_trades
                }

                # 计算大单的价格影响
                large_trade_indices = [i for i, s in enumerate(sizes) if s > large_threshold]
                if len(large_trade_indices) >= 2:
                    price_impacts = []
                    for i in range(1, len(large_trade_indices)):
                        idx = large_trade_indices[i]
                        prev_idx = large_trade_indices[i - 1]
                        if idx < len(prices) and prev_idx < len(prices):
                            impact = abs(prices[idx] - prices[prev_idx]) / prices[prev_idx]
                            price_impacts.append(impact)

                    if price_impacts:
                        informativeness['large_trade_impact'] = np.mean(price_impacts)

                # 估计知情交易概率
                if informativeness['trade_size_distribution']['large'] > 0.4:
                    informativeness['informed_trade_probability'] = 0.7
                elif informativeness['trade_size_distribution']['large'] > 0.3:
                    informativeness['informed_trade_probability'] = 0.5
                else:
                    informativeness['informed_trade_probability'] = 0.3

        except Exception as e:
            self.logger.error(f"交易信息含量分析错误: {e}")

        return informativeness


class SmartMoneyTracker:
    """聪明钱追踪器 - 识别庄家资金流向"""

    def __init__(self):
        self.logger = logging.getLogger('SmartMoney')

    def track_smart_money_flow(self, df, order_book_snapshots):
        """追踪聪明钱的流向"""

        flow_analysis = {
            'net_flow': 0,
            'smart_money_direction': 'NEUTRAL',
            'accumulation_distribution_line': [],
            'money_flow_index': 0,
            'large_order_imbalance': 0,
            'conviction_level': 'LOW'
        }

        if df is None or df.empty:
            return flow_analysis

        try:
            # 1. 计算修正的资金流向指标
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']

            # 区分大单和小单
            volume_mean = df['volume'].mean()
            large_threshold = volume_mean * 2  # 2倍平均成交量为大单

            # 计算大单资金流
            for i in range(len(df)):
                if df['volume'].iloc[i] > large_threshold:
                    # 大单资金流
                    if df['close'].iloc[i] > df['open'].iloc[i]:
                        flow_analysis['net_flow'] += money_flow.iloc[i]
                    else:
                        flow_analysis['net_flow'] -= money_flow.iloc[i]

            # 2. 计算资金流强度指标 (MFI)
            if len(df) >= 14:
                mfi = self.calculate_mfi(df, period=14)
                flow_analysis['money_flow_index'] = mfi

            # 3. 分析订单簿中的异常大单
            if order_book_snapshots:
                imbalance = self.analyze_order_book_imbalance(order_book_snapshots)
                flow_analysis['large_order_imbalance'] = imbalance

            # 4. 判断聪明钱方向
            if flow_analysis['net_flow'] > 0:
                if flow_analysis['money_flow_index'] > 70:
                    flow_analysis['smart_money_direction'] = 'STRONG_ACCUMULATING'
                    flow_analysis['conviction_level'] = 'HIGH'
                else:
                    flow_analysis['smart_money_direction'] = 'ACCUMULATING'
                    flow_analysis['conviction_level'] = 'MEDIUM'
            elif flow_analysis['net_flow'] < 0:
                if flow_analysis['money_flow_index'] < 30:
                    flow_analysis['smart_money_direction'] = 'STRONG_DISTRIBUTING'
                    flow_analysis['conviction_level'] = 'HIGH'
                else:
                    flow_analysis['smart_money_direction'] = 'DISTRIBUTING'
                    flow_analysis['conviction_level'] = 'MEDIUM'

            # 5. 计算累积/派发线
            ad_line = self.calculate_accumulation_distribution(df)
            flow_analysis['accumulation_distribution_line'] = ad_line

        except Exception as e:
            self.logger.error(f"聪明钱追踪错误: {e}")

        return flow_analysis

    def calculate_mfi(self, df, period=14):
        """计算资金流强度指标"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']

            # 计算正负资金流
            positive_flow = []
            negative_flow = []

            for i in range(1, len(df)):
                if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                    positive_flow.append(money_flow.iloc[i])
                    negative_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(money_flow.iloc[i])

            # 计算MFI
            positive_mf = pd.Series(positive_flow).rolling(period).sum()
            negative_mf = pd.Series(negative_flow).rolling(period).sum()

            mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

            return mfi.iloc[-1] if not mfi.empty else 50

        except Exception as e:
            self.logger.error(f"MFI计算错误: {e}")
            return 50

    def calculate_accumulation_distribution(self, df):
        """计算累积/派发线"""
        try:
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
            ad = (clv * df['volume']).cumsum()
            return ad.tolist()
        except:
            return []

    def analyze_order_book_imbalance(self, order_book_snapshots):
        """分析订单簿不平衡"""
        if not order_book_snapshots:
            return 0

        try:
            imbalances = []

            for snapshot in order_book_snapshots[-10:]:  # 最近10个快照
                if snapshot and snapshot.get('bid_sizes') and snapshot.get('ask_sizes'):
                    # 只看前5档的大单
                    bid_total = sum(snapshot['bid_sizes'][:5])
                    ask_total = sum(snapshot['ask_sizes'][:5])

                    if bid_total + ask_total > 0:
                        imbalance = (bid_total - ask_total) / (bid_total + ask_total)
                        imbalances.append(imbalance)

            return np.mean(imbalances) if imbalances else 0

        except Exception as e:
            self.logger.error(f"订单簿不平衡分析错误: {e}")
            return 0


class MicrostructureArbitrage:
    """微观结构套利检测 - 利用庄家操作产生的价格扭曲"""

    def __init__(self):
        self.logger = logging.getLogger('MicrostructureArb')

    def detect_arbitrage_opportunities(self, df, order_book, funding_rate=None):
        """检测各种微观结构套利机会"""

        opportunities = []

        try:
            # 1. 订单簿不平衡套利
            book_arb = self.detect_order_book_arbitrage(order_book)
            if book_arb['opportunity']:
                opportunities.append(book_arb)

            # 2. 时间优先权套利（抢先交易）
            latency_arb = self.detect_latency_arbitrage(df, order_book)
            if latency_arb['opportunity']:
                opportunities.append(latency_arb)

            # 3. 资金费率套利（针对永续合约）
            if funding_rate is not None:
                funding_arb = self.detect_funding_arbitrage(df, funding_rate)
                if funding_arb['opportunity']:
                    opportunities.append(funding_arb)

            # 4. 价格偏离套利
            deviation_arb = self.detect_price_deviation_arbitrage(df)
            if deviation_arb['opportunity']:
                opportunities.append(deviation_arb)

        except Exception as e:
            self.logger.error(f"套利机会检测错误: {e}")

        return {
            'opportunities': opportunities,
            'best_opportunity': max(opportunities,
                                    key=lambda x: x.get('expected_profit', 0)) if opportunities else None,
            'total_expected_profit': sum([opp.get('expected_profit', 0) for opp in opportunities])
        }

    def detect_order_book_arbitrage(self, order_book):
        """检测订单簿套利机会"""

        if not order_book or not order_book.get('bid_prices') or not order_book.get('ask_prices'):
            return {'opportunity': False}

        try:
            # 检查买一卖一价差
            if len(order_book['bid_prices']) > 0 and len(order_book['ask_prices']) > 0:
                spread = order_book['ask_prices'][0] - order_book['bid_prices'][0]
                mid_price = (order_book['ask_prices'][0] + order_book['bid_prices'][0]) / 2
                spread_pct = spread / mid_price if mid_price > 0 else 0

                # 如果价差异常大，可能存在套利机会
                if spread_pct > 0.002:  # 0.2%以上的价差
                    # 检查订单簿深度
                    bid_depth = sum(order_book['bid_sizes'][:5]) if len(order_book['bid_sizes']) >= 5 else sum(
                        order_book['bid_sizes'])
                    ask_depth = sum(order_book['ask_sizes'][:5]) if len(order_book['ask_sizes']) >= 5 else sum(
                        order_book['ask_sizes'])

                    if bid_depth > ask_depth * 2:  # 买盘远大于卖盘
                        return {
                            'opportunity': True,
                            'type': 'ORDER_BOOK_IMBALANCE',
                            'action': 'BUY_THEN_SELL',
                            'entry_price': order_book['ask_prices'][0],
                            'target_price': order_book['ask_prices'][0] * (1 + spread_pct * 0.7),
                            'expected_profit': spread_pct * 0.5,  # 保守估计
                            'risk': 'MEDIUM',
                            'reason': '买盘深度远大于卖盘，价格可能上推'
                        }
                    elif ask_depth > bid_depth * 2:  # 卖盘远大于买盘
                        return {
                            'opportunity': True,
                            'type': 'ORDER_BOOK_IMBALANCE',
                            'action': 'SELL_THEN_BUY',
                            'entry_price': order_book['bid_prices'][0],
                            'target_price': order_book['bid_prices'][0] * (1 - spread_pct * 0.7),
                            'expected_profit': spread_pct * 0.5,
                            'risk': 'MEDIUM',
                            'reason': '卖盘深度远大于买盘，价格可能下压'
                        }

        except Exception as e:
            self.logger.error(f"订单簿套利检测错误: {e}")

        return {'opportunity': False}

    def detect_latency_arbitrage(self, df, order_book):
        """检测延迟套利机会"""

        if df is None or len(df) < 10 or not order_book:
            return {'opportunity': False}

        try:
            # 检查价格动量
            recent_returns = df['close'].pct_change().tail(5)
            momentum = recent_returns.mean()

            # 如果有强烈动量但订单簿还未反应
            if abs(momentum) > 0.001:  # 0.1%的动量
                current_price = df['close'].iloc[-1]

                if order_book.get('ask_prices') and len(order_book['ask_prices']) > 0:
                    ask_price = order_book['ask_prices'][0]

                    if momentum > 0 and ask_price < current_price * 1.001:
                        # 上涨动量但卖价还未调整
                        return {
                            'opportunity': True,
                            'type': 'LATENCY_ARBITRAGE',
                            'action': 'QUICK_BUY',
                            'entry_price': ask_price,
                            'expected_profit': momentum * 0.5,
                            'risk': 'LOW',
                            'reason': '价格动量未反映在订单簿'
                        }

        except Exception as e:
            self.logger.error(f"延迟套利检测错误: {e}")

        return {'opportunity': False}

    def detect_funding_arbitrage(self, df, funding_rate):
        """检测资金费率套利"""

        if df is None or funding_rate is None:
            return {'opportunity': False}

        try:
            # 如果资金费率异常高或低
            if abs(funding_rate) > 0.001:  # 0.1%
                current_price = df['close'].iloc[-1]

                if funding_rate > 0.001:
                    # 多头支付费率，可以做空套利
                    return {
                        'opportunity': True,
                        'type': 'FUNDING_ARBITRAGE',
                        'action': 'SHORT_PERP_LONG_SPOT',
                        'funding_rate': funding_rate,
                        'expected_profit': funding_rate * 3,  # 假设持有3个费率周期
                        'risk': 'LOW',
                        'reason': f'资金费率{funding_rate:.4f}过高，空头套利'
                    }
                elif funding_rate < -0.001:
                    # 空头支付费率，可以做多套利
                    return {
                        'opportunity': True,
                        'type': 'FUNDING_ARBITRAGE',
                        'action': 'LONG_PERP_SHORT_SPOT',
                        'funding_rate': funding_rate,
                        'expected_profit': abs(funding_rate) * 3,
                        'risk': 'LOW',
                        'reason': f'资金费率{funding_rate:.4f}过低，多头套利'
                    }

        except Exception as e:
            self.logger.error(f"资金费率套利检测错误: {e}")

        return {'opportunity': False}

    def detect_price_deviation_arbitrage(self, df):
        """检测价格偏离套利"""

        if df is None or len(df) < 20:
            return {'opportunity': False}

        try:
            # 计算布林带
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()

            if not sma.empty and not std.empty:
                current_price = df['close'].iloc[-1]
                middle_band = sma.iloc[-1]
                upper_band = middle_band + 2 * std.iloc[-1]
                lower_band = middle_band - 2 * std.iloc[-1]

                # 价格偏离检测
                if current_price > upper_band:
                    deviation_pct = (current_price - upper_band) / upper_band
                    if deviation_pct > 0.01:  # 偏离1%以上
                        return {
                            'opportunity': True,
                            'type': 'MEAN_REVERSION',
                            'action': 'SELL',
                            'entry_price': current_price,
                            'target_price': middle_band,
                            'expected_profit': deviation_pct * 0.7,
                            'risk': 'MEDIUM',
                            'reason': '价格严重偏离上轨，均值回归机会'
                        }
                elif current_price < lower_band:
                    deviation_pct = (lower_band - current_price) / current_price
                    if deviation_pct > 0.01:
                        return {
                            'opportunity': True,
                            'type': 'MEAN_REVERSION',
                            'action': 'BUY',
                            'entry_price': current_price,
                            'target_price': middle_band,
                            'expected_profit': deviation_pct * 0.7,
                            'risk': 'MEDIUM',
                            'reason': '价格严重偏离下轨，均值回归机会'
                        }

        except Exception as e:
            self.logger.error(f"价格偏离套利检测错误: {e}")

        return {'opportunity': False}