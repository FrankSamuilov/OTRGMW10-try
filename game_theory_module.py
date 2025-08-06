"""
博弈论分析模块 - 包含所有博弈相关的分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import time
import asyncio
from auction_market_theory import AuctionMarketAnalyzer, AuctionGameTheoryIntegration
from logger_utils import Colors, print_colored
from indicators_module import calculate_optimized_indicators
from smc_module import detect_fvg, detect_order_blocks

"""
拍卖市场理论模块 - 替代SMC概念
基于价格发现、价值区域和市场均衡的分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from logger_utils import Colors, print_colored


class AuctionMarketAnalyzer:
    """拍卖市场理论分析器"""

    def __init__(self):
        self.logger = logging.getLogger('AuctionMarket')

    def analyze_market_structure(self, df: pd.DataFrame, order_book: Dict = None) -> Dict[str, Any]:
        """
        分析市场结构 - 基于拍卖理论

        核心概念：
        1. 价值区域 (Value Area) - 70%成交量发生的价格区间
        2. POC (Point of Control) - 成交量最大的价格
        3. 市场平衡/不平衡 - 价格接受或拒绝某个区域
        """

        analysis = {
            'market_state': 'UNKNOWN',
            'value_areas': [],
            'poc': None,
            'balance_areas': [],
            'imbalance_zones': [],
            'auction_type': None,
            'strength': 0
        }

        if df is None or len(df) < 20:
            return analysis

        try:
            # 1. 计算价值区域
            value_areas = self.calculate_value_areas(df)
            analysis['value_areas'] = value_areas

            # 2. 找出POC（成交量最大的价格）
            poc = self.find_point_of_control(df)
            analysis['poc'] = poc

            # 3. 识别平衡/不平衡区域
            balance_analysis = self.identify_balance_imbalance(df)
            analysis['balance_areas'] = balance_analysis['balance']
            analysis['imbalance_zones'] = balance_analysis['imbalance']

            # 4. 判断拍卖类型
            auction_type = self.determine_auction_type(df, value_areas)
            analysis['auction_type'] = auction_type

            # 5. 计算市场强度
            analysis['strength'] = self.calculate_market_strength(df, analysis)

            # 6. 确定市场状态
            analysis['market_state'] = self.determine_market_state(analysis)

        except Exception as e:
            self.logger.error(f"拍卖市场分析错误: {e}")

        return analysis

    def calculate_value_areas(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict[str, float]]:
        """
        计算价值区域 - 70%成交量发生的价格范围
        """
        value_areas = []

        try:
            # 使用最近N根K线
            recent_df = df.tail(lookback)

            # 创建价格-成交量分布
            price_volume_dist = {}

            for idx, row in recent_df.iterrows():
                # 将每根K线的成交量分配到高低价之间
                price_range = np.linspace(row['low'], row['high'], 10)
                volume_per_level = row['volume'] / len(price_range)

                for price in price_range:
                    price_key = round(price, 2)
                    if price_key not in price_volume_dist:
                        price_volume_dist[price_key] = 0
                    price_volume_dist[price_key] += volume_per_level

            # 排序并计算累积成交量
            sorted_prices = sorted(price_volume_dist.items(), key=lambda x: x[1], reverse=True)
            total_volume = sum([v for _, v in sorted_prices])

            # 找出70%成交量的价格区间
            cumulative_volume = 0
            value_area_prices = []

            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)

                if cumulative_volume >= total_volume * 0.7:
                    break

            if value_area_prices:
                value_area = {
                    'high': max(value_area_prices),
                    'low': min(value_area_prices),
                    'poc': sorted_prices[0][0],  # 成交量最大的价格
                    'volume_pct': 0.7,
                    'strength': len(value_area_prices) / len(sorted_prices)
                }
                value_areas.append(value_area)

        except Exception as e:
            self.logger.error(f"计算价值区域错误: {e}")

        return value_areas

    def find_point_of_control(self, df: pd.DataFrame) -> Optional[float]:
        """找出POC - 成交量最大的价格点"""
        try:
            # 简化方法：使用成交量加权平均价格
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            return round(vwap, 2)
        except:
            return None

    def identify_balance_imbalance(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        识别平衡和不平衡区域
        平衡：价格在一个区间内来回交易
        不平衡：价格快速穿过某个区域（类似FVG）
        """
        result = {
            'balance': [],
            'imbalance': []
        }

        try:
            # 检测平衡区域（横盘整理）
            for i in range(20, len(df)):
                window = df.iloc[i - 20:i]
                high_range = window['high'].max() - window['high'].min()
                low_range = window['low'].max() - window['low'].min()
                avg_range = (high_range + low_range) / 2

                # 如果20根K线的波动很小，认为是平衡区域
                if avg_range < window['ATR'].mean() * 2:
                    balance_zone = {
                        'start_idx': i - 20,
                        'end_idx': i,
                        'high': window['high'].max(),
                        'low': window['low'].min(),
                        'center': (window['high'].max() + window['low'].min()) / 2,
                        'strength': 1 - (avg_range / window['ATR'].mean())
                    }
                    result['balance'].append(balance_zone)

            # 检测不平衡区域（快速移动）
            for i in range(2, len(df) - 1):
                # 检查是否有价格快速移动
                move = abs(df['close'].iloc[i] - df['close'].iloc[i - 1])
                avg_move = df['ATR'].iloc[i]

                if move > avg_move * 2:  # 移动超过2倍ATR
                    imbalance = {
                        'idx': i,
                        'type': 'UP' if df['close'].iloc[i] > df['close'].iloc[i - 1] else 'DOWN',
                        'start_price': df['close'].iloc[i - 1],
                        'end_price': df['close'].iloc[i],
                        'magnitude': move / avg_move,
                        'filled': False
                    }

                    # 检查是否被回补
                    for j in range(i + 1, min(i + 20, len(df))):
                        if imbalance['type'] == 'UP':
                            if df['low'].iloc[j] <= imbalance['start_price']:
                                imbalance['filled'] = True
                                break
                        else:
                            if df['high'].iloc[j] >= imbalance['start_price']:
                                imbalance['filled'] = True
                                break

                    result['imbalance'].append(imbalance)

        except Exception as e:
            self.logger.error(f"识别平衡/不平衡错误: {e}")

        return result

    def determine_auction_type(self, df: pd.DataFrame, value_areas: List[Dict]) -> str:
        """
        判断拍卖类型
        - D型：平衡市场，价格在价值区域内
        - P型：上升趋势
        - b型：下降趋势
        - 趋势日：单向移动
        - 区间日：在范围内震荡
        """
        try:
            current_price = df['close'].iloc[-1]

            # 检查最近的价格动作
            recent_move = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]

            # 检查是否在价值区域内
            if value_areas:
                va = value_areas[0]
                if va['low'] <= current_price <= va['high']:
                    return 'D_SHAPE'  # 平衡市场

            # 根据移动幅度判断
            if abs(recent_move) > 0.02:  # 2%以上的移动
                return 'TREND_DAY' if recent_move > 0 else 'TREND_DAY_DOWN'
            else:
                return 'RANGE_DAY'

        except:
            return 'UNKNOWN'

    def calculate_market_strength(self, df: pd.DataFrame, analysis: Dict) -> float:
        """计算市场强度（0-1）"""
        strength = 0.5  # 基础强度

        try:
            # 1. 成交量因素
            recent_vol = df['volume'].iloc[-5:].mean()
            avg_vol = df['volume'].iloc[-20:].mean()
            if recent_vol > avg_vol * 1.5:
                strength += 0.2

            # 2. 不平衡区域未填补的数量
            unfilled_imbalances = sum(1 for i in analysis['imbalance_zones'] if not i['filled'])
            if unfilled_imbalances > 0:
                strength += 0.1 * min(unfilled_imbalances, 3)

            # 3. 趋势强度
            if analysis['auction_type'] in ['TREND_DAY', 'TREND_DAY_DOWN']:
                strength += 0.2

            # 确保在0-1范围内
            strength = max(0, min(1, strength))

        except:
            pass

        return strength

    def determine_market_state(self, analysis: Dict) -> str:
        """确定市场状态"""
        auction_type = analysis.get('auction_type', 'UNKNOWN')
        strength = analysis.get('strength', 0.5)

        if auction_type == 'D_SHAPE':
            return 'BALANCED'
        elif auction_type in ['TREND_DAY', 'TREND_DAY_DOWN']:
            return 'TRENDING'
        elif auction_type == 'RANGE_DAY':
            return 'RANGING'
        elif strength > 0.7:
            return 'BREAKOUT_POTENTIAL'
        else:
            return 'NEUTRAL'

    def get_trading_bias(self, analysis: Dict, current_price: float) -> Dict[str, Any]:
        """
        基于拍卖理论获取交易偏向
        """
        bias = {
            'direction': 'NEUTRAL',
            'strength': 0,
            'reason': '',
            'target': None,
            'stop': None
        }

        try:
            # 1. 基于POC的偏向
            if analysis['poc']:
                if current_price < analysis['poc'] * 0.995:  # 价格低于POC
                    bias['direction'] = 'LONG'
                    bias['reason'] = '价格低于POC，可能回归'
                    bias['target'] = analysis['poc']
                elif current_price > analysis['poc'] * 1.005:  # 价格高于POC
                    bias['direction'] = 'SHORT'
                    bias['reason'] = '价格高于POC，可能回归'
                    bias['target'] = analysis['poc']

            # 2. 基于不平衡区域
            unfilled_imbalances = [i for i in analysis['imbalance_zones'] if not i['filled']]
            if unfilled_imbalances:
                latest_imbalance = unfilled_imbalances[-1]
                if latest_imbalance['type'] == 'UP':
                    bias['direction'] = 'LONG'
                    bias['reason'] = '上方不平衡未填补'
                else:
                    bias['direction'] = 'SHORT'
                    bias['reason'] = '下方不平衡未填补'

            # 3. 基于市场状态
            if analysis['market_state'] == 'TRENDING':
                if analysis['auction_type'] == 'TREND_DAY':
                    bias['direction'] = 'LONG'
                    bias['strength'] = 0.8
                elif analysis['auction_type'] == 'TREND_DAY_DOWN':
                    bias['direction'] = 'SHORT'
                    bias['strength'] = 0.8

            # 计算强度
            bias['strength'] = min(analysis['strength'], 0.9)

        except Exception as e:
            self.logger.error(f"获取交易偏向错误: {e}")

        return bias


class AuctionGameTheoryIntegration:
    """拍卖理论与博弈论的整合"""

    def __init__(self):
        self.auction_analyzer = AuctionMarketAnalyzer()
        self.logger = logging.getLogger('AuctionGameTheory')

    def analyze_with_game_theory(self, df: pd.DataFrame, market_data: Dict) -> Dict[str, Any]:
        """
        结合拍卖理论和博弈论进行分析
        """
        integrated_analysis = {
            'auction_analysis': {},
            'game_theory_insights': {},
            'combined_signal': {},
            'confidence': 0
        }

        try:
            # 1. 拍卖市场分析
            auction_analysis = self.auction_analyzer.analyze_market_structure(
                df,
                market_data.get('order_book')
            )
            integrated_analysis['auction_analysis'] = auction_analysis

            # 2. 博弈论视角
            game_insights = self.analyze_participant_behavior(
                auction_analysis,
                market_data
            )
            integrated_analysis['game_theory_insights'] = game_insights

            # 3. 综合信号
            combined = self.generate_combined_signal(
                auction_analysis,
                game_insights,
                df
            )
            integrated_analysis['combined_signal'] = combined
            integrated_analysis['confidence'] = combined.get('confidence', 0)

        except Exception as e:
            self.logger.error(f"整合分析错误: {e}")

        return integrated_analysis

    def analyze_participant_behavior(self, auction_analysis: Dict, market_data: Dict) -> Dict:
        """
        分析市场参与者行为
        """
        behavior = {
            'buyer_strength': 0.5,
            'seller_strength': 0.5,
            'dominant_side': 'NEUTRAL',
            'manipulation_signs': False,
            'smart_money_direction': 'NEUTRAL'
        }

        try:
            # 1. 从订单簿分析买卖力量
            if market_data.get('order_book'):
                order_book = market_data['order_book']

                # 计算买卖压力
                bid_volume = sum(order_book.get('bid_sizes', [])[:10])
                ask_volume = sum(order_book.get('ask_sizes', [])[:10])

                if bid_volume + ask_volume > 0:
                    behavior['buyer_strength'] = bid_volume / (bid_volume + ask_volume)
                    behavior['seller_strength'] = 1 - behavior['buyer_strength']

                    if behavior['buyer_strength'] > 0.6:
                        behavior['dominant_side'] = 'BUYERS'
                    elif behavior['seller_strength'] > 0.6:
                        behavior['dominant_side'] = 'SELLERS'

            # 2. 检测操纵迹象
            if auction_analysis.get('imbalance_zones'):
                # 如果有多个未填补的不平衡，可能存在操纵
                unfilled_count = sum(1 for i in auction_analysis['imbalance_zones'] if not i['filled'])
                if unfilled_count > 3:
                    behavior['manipulation_signs'] = True

            # 3. 判断聪明钱方向
            if market_data.get('long_short_ratio'):
                ls_ratio = market_data['long_short_ratio']
                top_trader_ratio = ls_ratio.get('top_traders', {}).get('ratio', 1)

                if top_trader_ratio > 1.5:
                    behavior['smart_money_direction'] = 'LONG'
                elif top_trader_ratio < 0.7:
                    behavior['smart_money_direction'] = 'SHORT'

        except Exception as e:
            self.logger.error(f"分析参与者行为错误: {e}")

        return behavior

    def generate_combined_signal(self, auction_analysis: Dict,
                                 game_insights: Dict, df: pd.DataFrame) -> Dict:
        """生成综合交易信号"""
        signal = {
            'action': 'HOLD',
            'confidence': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'reasoning': []
        }

        try:
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0

            # 获取拍卖理论的交易偏向
            auction_bias = self.auction_analyzer.get_trading_bias(auction_analysis, current_price)

            # 1. 强趋势信号
            if (auction_analysis['market_state'] == 'TRENDING' and
                    game_insights['dominant_side'] != 'NEUTRAL'):

                if (auction_bias['direction'] == 'LONG' and
                        game_insights['dominant_side'] == 'BUYERS'):
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('拍卖理论和买方主导一致')

                elif (auction_bias['direction'] == 'SHORT' and
                      game_insights['dominant_side'] == 'SELLERS'):
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('拍卖理论和卖方主导一致')

            # 2. 价值回归信号
            elif (auction_analysis['poc'] and
                  abs(current_price - auction_analysis['poc']) / current_price > 0.01):

                if current_price < auction_analysis['poc']:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.6
                    signal['reasoning'].append('价格低于POC，预期回归')
                else:
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.6
                    signal['reasoning'].append('价格高于POC，预期回归')

            # 3. 聪明钱跟随
            if (game_insights['smart_money_direction'] != 'NEUTRAL' and
                    not game_insights['manipulation_signs']):

                if game_insights['smart_money_direction'] == 'LONG':
                    if signal['action'] == 'HOLD':
                        signal['action'] = 'BUY'
                        signal['confidence'] = 0.5
                    elif signal['action'] == 'BUY':
                        signal['confidence'] = min(signal['confidence'] + 0.2, 0.9)
                    signal['reasoning'].append('聪明钱看多')

                elif game_insights['smart_money_direction'] == 'SHORT':
                    if signal['action'] == 'HOLD':
                        signal['action'] = 'SELL'
                        signal['confidence'] = 0.5
                    elif signal['action'] == 'SELL':
                        signal['confidence'] = min(signal['confidence'] + 0.2, 0.9)
                    signal['reasoning'].append('聪明钱看空')

            # 4. 设置入场和止损
            if signal['action'] != 'HOLD':
                signal['entry_price'] = current_price

                # 基于ATR的止损
                if signal['action'] == 'BUY':
                    signal['stop_loss'] = current_price - atr * 2
                    signal['take_profit'] = current_price + atr * 3
                else:  # SELL
                    signal['stop_loss'] = current_price + atr * 2
                    signal['take_profit'] = current_price - atr * 3

                # 如果有POC目标，使用POC
                if auction_bias.get('target'):
                    signal['take_profit'] = auction_bias['target']

            # 5. 风险检查
            if game_insights['manipulation_signs']:
                signal['confidence'] *= 0.7
                signal['reasoning'].append('检测到可能的操纵行为，降低置信度')

        except Exception as e:
            self.logger.error(f"生成综合信号错误: {e}")

        return signal

class MarketDataCollector:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('MarketDataCollector')

    def collect_full_market_data(self, symbol):
        """收集完整市场数据"""
        try:
            # 这是一个简化的实现
            market_data = {
                'symbol': symbol,
                'price_data': None,  # 这里应该获取价格数据
                'order_book': {},
                'long_short_ratio': {},
                'timestamp': datetime.now()
            }

            # 获取K线数据
            try:
                klines = self.client.futures_klines(symbol=symbol, interval='15m', limit=100)
                # 转换为DataFrame格式
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                   'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                                   'taker_buy_quote', 'ignore'])
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['open'] = df['open'].astype(float)
                df['volume'] = df['volume'].astype(float)

                market_data['price_data'] = df
            except:
                pass

            return market_data

        except Exception as e:
            self.logger.error(f"收集市场数据失败 {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def get_historical_data_safe(self, symbol):
        """安全获取历史数据"""
        try:
            from data_module import get_historical_data
            return get_historical_data(self.client, symbol)
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            return None

    def get_order_book_async(self, symbol):
        """异步获取订单簿"""
        try:
            order_book = self.client.futures_order_book(symbol=symbol, limit=20)
            return self.parse_order_book(order_book)
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return None

    def parse_order_book(self, order_book):
        """解析订单簿数据"""
        if not order_book:
            return None

        return {
            'bid_prices': [float(bid[0]) for bid in order_book.get('bids', [])],
            'bid_sizes': [float(bid[1]) for bid in order_book.get('bids', [])],
            'ask_prices': [float(ask[0]) for ask in order_book.get('asks', [])],
            'ask_sizes': [float(ask[1]) for ask in order_book.get('asks', [])],
            'timestamp': time.time()
        }

    def get_long_short_ratio(self, symbol):
        """获取多空比数据"""
        try:
            # 顶级交易员持仓比
            top_trader_ratio = self.client.futures_top_longshort_position_ratio(
                symbol=symbol,
                period='5m',
                limit=1
            )

            # 普通用户持仓比
            global_ratio = self.client.futures_global_longshort_ratio(
                symbol=symbol,
                period='5m',
                limit=1
            )

            # 大户持仓比
            taker_ratio = self.client.futures_takerlongshort_ratio(
                symbol=symbol,
                period='5m',
                limit=1
            )

            return {
                'top_traders': {
                    'long_ratio': float(top_trader_ratio[0]['longAccount']) if top_trader_ratio else 0.5,
                    'short_ratio': float(top_trader_ratio[0]['shortAccount']) if top_trader_ratio else 0.5,
                    'ratio': float(top_trader_ratio[0]['longShortRatio']) if top_trader_ratio else 1.0
                },
                'global': {
                    'long_ratio': float(global_ratio[0]['longAccount']) if global_ratio else 0.5,
                    'short_ratio': float(global_ratio[0]['shortAccount']) if global_ratio else 0.5,
                    'ratio': float(global_ratio[0]['longShortRatio']) if global_ratio else 1.0
                },
                'takers': {
                    'buy_vol_ratio': float(taker_ratio[0]['buySellRatio']) if taker_ratio else 1.0,
                    'sell_vol_ratio': 1.0,  # 计算得出
                    'ratio': float(taker_ratio[0]['buySellRatio']) if taker_ratio else 1.0
                }
            }
        except Exception as e:
            self.logger.error(f"获取多空比失败: {e}")
            # 返回默认值
            return {
                'top_traders': {'long_ratio': 0.5, 'short_ratio': 0.5, 'ratio': 1.0},
                'global': {'long_ratio': 0.5, 'short_ratio': 0.5, 'ratio': 1.0},
                'takers': {'buy_vol_ratio': 1.0, 'sell_vol_ratio': 1.0, 'ratio': 1.0}
            }

    def get_funding_rate_async(self, symbol):
        """异步获取资金费率"""
        try:
            funding = self.client.futures_funding_rate(symbol=symbol, limit=1)
            return float(funding[0]['fundingRate']) if funding else 0
        except Exception as e:
            self.logger.error(f"获取资金费率失败: {e}")
            return 0

    def get_open_interest_async(self, symbol):
        """异步获取持仓量"""
        try:
            open_interest = self.client.futures_open_interest(symbol=symbol)
            return float(open_interest['openInterest'])
        except Exception as e:
            self.logger.error(f"获取持仓量失败: {e}")
            return 0

    def get_recent_trades_async(self, symbol):
        """异步获取最近成交"""
        try:
            trades = self.client.futures_recent_trades(symbol=symbol, limit=100)
            return trades
        except Exception as e:
            self.logger.error(f"获取最近成交失败: {e}")
            return []

    def analyze_trade_flow(self, trades):
        """分析成交流"""
        if not trades:
            return {
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_count': 0,
                'sell_count': 0,
                'large_trades': []
            }

        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        large_trades = []

        # 计算平均交易量
        volumes = [float(t['qty']) for t in trades]
        avg_volume = np.mean(volumes) if volumes else 0
        large_threshold = avg_volume * 3  # 3倍平均值为大单

        for trade in trades:
            qty = float(trade['qty'])
            price = float(trade['price'])

            # 判断买卖方向
            if trade['isBuyerMaker']:
                sell_volume += qty
                sell_count += 1
            else:
                buy_volume += qty
                buy_count += 1

            # 记录大单
            if qty > large_threshold:
                large_trades.append({
                    'price': price,
                    'qty': qty,
                    'side': 'SELL' if trade['isBuyerMaker'] else 'BUY',
                    'time': trade['time']
                })

        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'large_trades': large_trades[:10]  # 只保留最近10笔大单
        }

    def get_liquidation_data(self, symbol):
        """获取爆仓数据（如果可用）"""
        # 这个需要特殊API或第三方数据源
        # 暂时返回模拟数据
        return {
            'long_liquidations': 0,
            'short_liquidations': 0,
            'total_liquidations': 0
        }


class SMCGameTheoryAnalyzer:
    """博弈分析器 - 使用拍卖理论替代SMC"""

    def __init__(self):
        self.logger = logging.getLogger('GameTheory')
        self.auction_analyzer = AuctionMarketAnalyzer()
        self.auction_game_integration = AuctionGameTheoryIntegration()

    def analyze_market_structure(self, data):
        """分析市场结构 - 使用拍卖理论"""

        analysis = {
            'auction_signals': {},
            'game_theory_signals': {},
            'long_short_dynamics': {},
            'manipulation_evidence': {},
            'trading_recommendation': {}
        }

        if not data or not data.get('kline_data') is not None:
            return analysis

        df = data['kline_data']

        # 1. 拍卖理论分析（替代SMC）
        auction_analysis = self.auction_analyzer.analyze_market_structure(df, data.get('order_book'))
        analysis['auction_signals'] = auction_analysis

        # 2. 整合博弈论
        integrated = self.auction_game_integration.analyze_with_game_theory(df, data)
        analysis['game_theory_signals'] = integrated['game_theory_insights']

        # 3. 多空比博弈分析
        ls_game = self.analyze_long_short_game(data.get('long_short_ratio', {}), data)
        analysis['long_short_dynamics'] = ls_game

        # 4. 操纵检测
        manipulation = self.detect_market_manipulation(data, integrated)
        analysis['manipulation_evidence'] = manipulation

        # 5. 综合建议
        recommendation = self.generate_trading_recommendation(analysis)
        analysis['trading_recommendation'] = recommendation

        return analysis

    def analyze_fvg_with_game_theory(self, df, market_data):
        """FVG分析结合博弈论"""

        # 使用现有的FVG检测
        try:
            fvg_data = detect_fvg(df)
        except:
            fvg_data = []

        if not market_data.get('long_short_ratio'):
            return fvg_data

        # 增强：结合多空比分析FVG的可靠性
        enhanced_fvgs = []

        for fvg in fvg_data:
            # 检查FVG形成时的多空比
            fvg_reliability = self.assess_fvg_reliability(
                fvg,
                market_data['long_short_ratio'],
                market_data['order_book']
            )

            fvg['game_theory_assessment'] = fvg_reliability

            # 如果是看涨FVG
            if fvg['direction'] == 'UP':
                # 检查是否与多头趋势一致
                if market_data['long_short_ratio']['top_traders']['ratio'] > 1.5:
                    fvg['reliability'] = 'HIGH'
                    fvg['reason'] = '顶级交易员支持看涨FVG'
                elif market_data['long_short_ratio']['global']['ratio'] < 0.8:
                    fvg['reliability'] = 'LOW'
                    fvg['reason'] = '散户过度看空，可能是诱多陷阱'
                else:
                    fvg['reliability'] = 'MEDIUM'
                    fvg['reason'] = '市场情绪中性'

            # 如果是看跌FVG
            elif fvg['direction'] == 'DOWN':
                if market_data['long_short_ratio']['top_traders']['ratio'] < 0.7:
                    fvg['reliability'] = 'HIGH'
                    fvg['reason'] = '顶级交易员支持看跌FVG'
                elif market_data['long_short_ratio']['global']['ratio'] > 1.2:
                    fvg['reliability'] = 'LOW'
                    fvg['reason'] = '散户过度看多，可能是诱空陷阱'
                else:
                    fvg['reliability'] = 'MEDIUM'
                    fvg['reason'] = '市场情绪中性'

            enhanced_fvgs.append(fvg)

        return enhanced_fvgs

    def assess_fvg_reliability(self, fvg, ls_ratio, order_book):
        """评估FVG的可靠性"""
        reliability_score = 0.5  # 基础分数

        # 1. 检查订单簿支持
        if order_book:
            bid_pressure = sum(order_book.get('bid_sizes', [])[:5])
            ask_pressure = sum(order_book.get('ask_sizes', [])[:5])

            if fvg['direction'] == 'UP' and bid_pressure > ask_pressure * 1.5:
                reliability_score += 0.2
            elif fvg['direction'] == 'DOWN' and ask_pressure > bid_pressure * 1.5:
                reliability_score += 0.2

        # 2. 检查多空比一致性
        if ls_ratio:
            top_ratio = ls_ratio['top_traders']['ratio']
            if (fvg['direction'] == 'UP' and top_ratio > 1.3) or \
                    (fvg['direction'] == 'DOWN' and top_ratio < 0.7):
                reliability_score += 0.3

        return {
            'score': reliability_score,
            'grade': 'HIGH' if reliability_score > 0.7 else 'MEDIUM' if reliability_score > 0.4 else 'LOW'
        }

    def analyze_long_short_game(self, ls_ratio, market_data):
        """多空比博弈分析"""

        if not ls_ratio:
            return {'status': 'NO_DATA'}

        game_analysis = {
            'market_sentiment': '',
            'smart_vs_retail': '',
            'manipulation_probability': 0.0,
            'game_state': '',
            'strategic_recommendation': ''
        }

        # 1. 分析顶级交易员vs散户的博弈
        top_ratio = ls_ratio['top_traders']['ratio']
        global_ratio = ls_ratio['global']['ratio']

        # 计算聪明钱与散户的分歧度
        divergence = abs(top_ratio - global_ratio) / max(top_ratio, global_ratio)

        if divergence > 0.3:  # 30%以上的分歧
            if top_ratio > global_ratio * 1.3:
                game_analysis['smart_vs_retail'] = 'SMART_BULLISH_RETAIL_BEARISH'
                game_analysis['game_state'] = '聪明钱在吸筹，散户在恐慌'
                game_analysis['manipulation_probability'] = 0.7
                game_analysis['strategic_recommendation'] = '跟随聪明钱做多'
            elif top_ratio < global_ratio * 0.7:
                game_analysis['smart_vs_retail'] = 'SMART_BEARISH_RETAIL_BULLISH'
                game_analysis['game_state'] = '聪明钱在派发，散户在接盘'
                game_analysis['manipulation_probability'] = 0.8
                game_analysis['strategic_recommendation'] = '跟随聪明钱做空'
        else:
            game_analysis['smart_vs_retail'] = 'ALIGNED'
            game_analysis['game_state'] = '市场共识'
            game_analysis['manipulation_probability'] = 0.2
            game_analysis['strategic_recommendation'] = '顺势而为'

        # 2. 极端多空比分析
        if global_ratio > 2.0:
            game_analysis['market_sentiment'] = 'EXTREME_BULLISH'
            if game_analysis['strategic_recommendation'] == '顺势而为':
                game_analysis['strategic_recommendation'] = '极度看多，注意反转风险'
        elif global_ratio < 0.5:
            game_analysis['market_sentiment'] = 'EXTREME_BEARISH'
            if game_analysis['strategic_recommendation'] == '顺势而为':
                game_analysis['strategic_recommendation'] = '极度看空，可能接近底部'
        else:
            game_analysis['market_sentiment'] = 'NEUTRAL'

        # 3. 结合资金费率的博弈
        funding = market_data.get('funding_rate', 0)
        if funding > 0.001 and global_ratio > 1.5:
            game_analysis['funding_pressure'] = '多头支付高额资金费率，可能被迫平仓'
        elif funding < -0.001 and global_ratio < 0.7:
            game_analysis['funding_pressure'] = '空头支付高额资金费率，可能反弹'

        return game_analysis

    def detect_order_blocks_with_volume(self, df, order_book):
        """结合成交量检测订单块"""
        try:
            # 使用现有的订单块检测
            order_blocks = detect_order_blocks(df)

            # 增强：结合订单簿信息
            if order_book and order_blocks:
                current_price = df['close'].iloc[-1]

                for block in order_blocks:
                    # 检查订单块附近的订单簿深度
                    block_price = block['price']

                    # 计算订单块附近的支撑/阻力强度
                    support_strength = 0
                    resistance_strength = 0

                    if order_book.get('bid_prices'):
                        for i, bid_price in enumerate(order_book['bid_prices'][:10]):
                            if abs(bid_price - block_price) / block_price < 0.01:  # 1%范围内
                                support_strength += order_book['bid_sizes'][i]

                    if order_book.get('ask_prices'):
                        for i, ask_price in enumerate(order_book['ask_prices'][:10]):
                            if abs(ask_price - block_price) / block_price < 0.01:
                                resistance_strength += order_book['ask_sizes'][i]

                    block['order_book_support'] = support_strength
                    block['order_book_resistance'] = resistance_strength

            return order_blocks

        except Exception as e:
            self.logger.error(f"订单块检测失败: {e}")
            return []

    def identify_liquidity_zones(self, df, market_data):
        """识别流动性区域"""
        zones = []

        if df is None or df.empty:
            return zones

        try:
            # 1. 识别前高前低
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            current_price = df['close'].iloc[-1]

            # 2. 计算关键心理价位
            psychological_levels = self.calculate_psychological_levels(current_price)

            # 3. 结合多空比识别止损区域
            if market_data.get('long_short_ratio'):
                ls_ratio = market_data['long_short_ratio']['global']['ratio']

                if ls_ratio > 1.5:  # 多头过多
                    # 多头止损区可能在下方
                    long_stop_zone = current_price * 0.97  # 3%下方
                    zones.append({
                        'type': 'LONG_STOP_ZONE',
                        'price': long_stop_zone,
                        'strength': ls_ratio,
                        'description': '多头止损密集区'
                    })

                elif ls_ratio < 0.7:  # 空头过多
                    # 空头止损区可能在上方
                    short_stop_zone = current_price * 1.03  # 3%上方
                    zones.append({
                        'type': 'SHORT_STOP_ZONE',
                        'price': short_stop_zone,
                        'strength': 1 / ls_ratio,
                        'description': '空头止损密集区'
                    })

            # 4. 添加前高前低作为流动性区域
            zones.append({
                'type': 'RESISTANCE',
                'price': recent_high,
                'strength': 1.0,
                'description': '近期高点阻力'
            })

            zones.append({
                'type': 'SUPPORT',
                'price': recent_low,
                'strength': 1.0,
                'description': '近期低点支撑'
            })

            return zones

        except Exception as e:
            self.logger.error(f"流动性区域识别失败: {e}")
            return zones

    def calculate_psychological_levels(self, price):
        """计算心理价位"""
        levels = []

        # 整千价位
        thousand_level = round(price / 1000) * 1000
        levels.append(thousand_level)

        # 整百价位
        hundred_level = round(price / 100) * 100
        levels.append(hundred_level)

        # 整十价位（对于小价格）
        if price < 100:
            ten_level = round(price / 10) * 10
            levels.append(ten_level)

        return [l for l in levels if 0.95 * price <= l <= 1.05 * price]  # 只保留±5%范围内的

    def detect_market_manipulation(self, data, integrated_analysis):
        """检测市场操纵 - 基于拍卖理论"""
        manipulation = {
            'probability': 0,
            'type': 'NONE',
            'evidence': []
        }

        try:
            # 从拍卖理论角度检测
            game_insights = integrated_analysis.get('game_theory_insights', {})

            if game_insights.get('manipulation_signs'):
                manipulation['probability'] = 0.7
                manipulation['evidence'].append('拍卖理论检测到异常')

            # 检查订单簿异常
            if data.get('order_book'):
                order_book = data['order_book']
                # 检查是否有异常大的订单
                if self._detect_spoofing(order_book):
                    manipulation['probability'] = max(manipulation['probability'], 0.8)
                    manipulation['type'] = 'SPOOFING'
                    manipulation['evidence'].append('订单簿存在幌骗')

        except Exception as e:
            self.logger.error(f"操纵检测错误: {e}")

        return manipulation

    def generate_trading_recommendation(self, analysis):
        """生成交易建议 - 基于拍卖理论"""
        recommendation = {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': []
        }

        try:
            # 获取拍卖理论信号
            auction_signals = analysis.get('auction_signals', {})

            # 使用整合的信号（如果可用）
            if 'combined_signal' in analysis.get('game_theory_signals', {}):
                combined = analysis['game_theory_signals']['combined_signal']
                recommendation['action'] = combined.get('action', 'HOLD')
                recommendation['confidence'] = combined.get('confidence', 0)
                recommendation['reasoning'] = combined.get('reasoning', [])

            # 额外的风险检查
            if analysis.get('manipulation_evidence', {}).get('probability', 0) > 0.7:
                recommendation['confidence'] *= 0.5
                recommendation['reasoning'].append('高操纵风险，降低置信度')

        except Exception as e:
            self.logger.error(f"生成建议错误: {e}")

        return recommendation

    def detect_bull_trap(self, data):
        """检测诱多陷阱"""
        result = {
            'detected': False,
            'confidence': 0.0,
            'evidence': []
        }

        if not data.get('kline_data') is not None:
            return result

        df = data['kline_data']
        ls_ratio = data.get('long_short_ratio', {})

        # 条件1：价格在高位，但聪明钱在卖
        if len(df) >= 20:
            price_position = (df['close'].iloc[-1] - df['low'].tail(20).min()) / \
                             (df['high'].tail(20).max() - df['low'].tail(20).min())

            if price_position > 0.8:  # 价格在近期高位
                if ls_ratio and ls_ratio['top_traders']['ratio'] < 0.8:  # 但顶级交易员看空
                    result['detected'] = True
                    result['confidence'] += 0.4
                    result['evidence'].append('价格高位但聪明钱看空')

        # 条件2：成交量递减的上涨
        if len(df) >= 5:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-5]) / df['volume'].iloc[-5]

            if price_change > 0.02 and volume_change < -0.2:  # 价涨量缩
                result['detected'] = True
                result['confidence'] += 0.3
                result['evidence'].append('价格上涨但成交量萎缩')

        return result

    def detect_bear_trap(self, data):
        """检测诱空陷阱"""
        result = {
            'detected': False,
            'confidence': 0.0,
            'evidence': []
        }

        if not data.get('kline_data') is not None:
            return result

        df = data['kline_data']
        ls_ratio = data.get('long_short_ratio', {})

        # 条件1：价格在低位，但聪明钱在买
        if len(df) >= 20:
            price_position = (df['close'].iloc[-1] - df['low'].tail(20).min()) / \
                             (df['high'].tail(20).max() - df['low'].tail(20).min())

            if price_position < 0.2:  # 价格在近期低位
                if ls_ratio and ls_ratio['top_traders']['ratio'] > 1.2:  # 但顶级交易员看多
                    result['detected'] = True
                    result['confidence'] += 0.4
                    result['evidence'].append('价格低位但聪明钱看多')

        # 条件2：恐慌性下跌后的快速反弹
        if len(df) >= 3:
            if df['close'].iloc[-3] > df['close'].iloc[-2] * 1.02 and \
                    df['close'].iloc[-1] > df['close'].iloc[-2] * 1.01:
                # V型反转
                result['detected'] = True
                result['confidence'] += 0.3
                result['evidence'].append('V型反转形态')

        return result

    # 在 game_theory_module.py 中修改
    def detect_order_blocks_with_volume(self, df, order_book):
        """简化版订单块检测 - 使用支撑阻力替代"""
        try:
            # 不再使用复杂的订单块，改用简单的支撑阻力
            support_resistance = []

            if df is not None and len(df) > 20:
                # 简单的支撑阻力：使用近期高低点
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                current_price = df['close'].iloc[-1]

                support_resistance.append({
                    'type': 'RESISTANCE',
                    'price': recent_high,
                    'strength': 1.0
                })

                support_resistance.append({
                    'type': 'SUPPORT',
                    'price': recent_low,
                    'strength': 1.0
                })

            return support_resistance

        except Exception as e:
            self.logger.error(f"支撑阻力检测失败: {e}")
            return []

    def detect_stop_hunting(self, data):
        """检测止损猎杀"""
        result = {
            'detected': False,
            'confidence': 0.0,
            'evidence': []
        }

        if not data.get('kline_data') is not None:
            return result

        df = data['kline_data']

        # 检测插针行为
        if len(df) >= 1:
            last_candle = df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            upper_wick = last_candle['high'] - max(last_candle['close'], last_candle['open'])
            lower_wick = min(last_candle['close'], last_candle['open']) - last_candle['low']

            # 上影线过长
            if upper_wick > body * 2:
                result['detected'] = True
                result['confidence'] += 0.5
                result['evidence'].append('长上影线，可能扫空头止损')

            # 下影线过长
            if lower_wick > body * 2:
                result['detected'] = True
                result['confidence'] += 0.5
                result['evidence'].append('长下影线，可能扫多头止损')

        return result

    def synthesize_signals(self, analysis, data):
        """综合所有信号生成最终建议"""
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'reasoning': []
        }

        if not data.get('kline_data') is not None:
            return recommendation

        current_price = data['kline_data']['close'].iloc[-1]

        # 1. 检查是否有操纵行为
        if analysis['manipulation_evidence']['detected']:
            manipulation_type = analysis['manipulation_evidence']['type']

            if manipulation_type == 'BULL_TRAP':
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = analysis['manipulation_evidence']['confidence']
                recommendation['reasoning'].append('检测到诱多陷阱，建议做空')
            elif manipulation_type == 'BEAR_TRAP':
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = analysis['manipulation_evidence']['confidence']
                recommendation['reasoning'].append('检测到诱空陷阱，建议做多')
            elif manipulation_type == 'STOP_HUNTING':
                recommendation['action'] = 'WAIT'
                recommendation['reasoning'].append('检测到止损猎杀，等待明朗')
                return recommendation

        # 2. 分析多空博弈
        ls_dynamics = analysis['long_short_dynamics']
        if ls_dynamics.get('manipulation_probability', 0) > 0.6:
            if 'SMART_BULLISH' in ls_dynamics.get('smart_vs_retail', ''):
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = max(recommendation['confidence'], 0.7)
                recommendation['reasoning'].append(ls_dynamics['strategic_recommendation'])
            elif 'SMART_BEARISH' in ls_dynamics.get('smart_vs_retail', ''):
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = max(recommendation['confidence'], 0.7)
                recommendation['reasoning'].append(ls_dynamics['strategic_recommendation'])

        # 3. 检查SMC信号
        # FVG信号
        fvg_signals = analysis['smc_signals'].get('fvg', [])
        for fvg in fvg_signals:
            if fvg.get('reliability') == 'HIGH' and not fvg.get('is_filled', True):
                if fvg['direction'] == 'UP' and recommendation['action'] != 'SELL':
                    recommendation['action'] = 'BUY'
                    recommendation['confidence'] = max(recommendation['confidence'], 0.8)
                    recommendation['reasoning'].append(f"高可靠性看涨FVG: {fvg.get('reason', '')}")
                elif fvg['direction'] == 'DOWN' and recommendation['action'] != 'BUY':
                    recommendation['action'] = 'SELL'
                    recommendation['confidence'] = max(recommendation['confidence'], 0.8)
                    recommendation['reasoning'].append(f"高可靠性看跌FVG: {fvg.get('reason', '')}")

        # 4. 计算具体交易参数
        if recommendation['action'] in ['BUY', 'SELL']:
            recommendation['entry_price'] = current_price

            if recommendation['action'] == 'BUY':
                recommendation['stop_loss'] = current_price * 0.98  # 2%止损
                recommendation['take_profit'] = current_price * 1.05  # 5%止盈
            else:
                recommendation['stop_loss'] = current_price * 1.02
                recommendation['take_profit'] = current_price * 0.95

            # 根据置信度计算仓位
            recommendation['position_size'] = min(recommendation['confidence'], 0.3)  # 最大30%仓位

        return recommendation


class IntegratedDecisionEngine:
    """综合所有分析的决策引擎"""

    def __init__(self):
        self.smc_analyzer = SMCGameTheoryAnalyzer()
        self.logger = logging.getLogger('DecisionEngine')

    def make_trading_decision(self, market_data):
        """基于完整分析做出交易决策"""

        decision = {
            'action': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'entry_strategy': {},
            'risk_management': {},
            'reasoning': []
        }

        try:
            # 1. 运行SMC + 博弈论分析
            analysis = self.smc_analyzer.analyze_market_structure(market_data)

            # 2. 获取综合建议
            recommendation = analysis.get('trading_recommendation', {})

            # 3. 转换为决策格式
            if recommendation.get('action') in ['BUY', 'SELL']:
                decision['action'] = recommendation['action']
                decision['confidence'] = recommendation.get('confidence', 0.5)
                decision['position_size'] = recommendation.get('position_size', 0.1)
                decision['reasoning'] = recommendation.get('reasoning', [])

                # 设置入场策略
                decision['entry_strategy'] = {
                    'type': 'MARKET',
                    'entry_price': recommendation.get('entry_price', 0),
                    'method': '市价入场'
                }

                # 设置风险管理
                decision['risk_management'] = {
                    'stop_loss': recommendation.get('stop_loss', 0),
                    'take_profit': recommendation.get('take_profit', 0),
                    'risk_level': 'HIGH' if decision['confidence'] < 0.6 else 'MEDIUM'
                }

            elif recommendation.get('action') == 'WAIT':
                decision['action'] = 'HOLD'
                decision['reasoning'] = recommendation.get('reasoning', ['等待更好的入场时机'])

        except Exception as e:
            self.logger.error(f"决策引擎错误: {e}")
            decision['reasoning'].append(f'分析错误: {str(e)}')

        return decision


class GameTheoryModule:
    """主博弈论模块 - 整合所有博弈分析功能"""

    def __init__(self):
        self.logger = logging.getLogger('GameTheoryModule')
        self.enabled = True

        # 初始化子组件
        try:
            self.data_collector = None  # 需要client才能初始化
            self.analyzer = SMCGameTheoryAnalyzer()
            self.decision_engine = IntegratedDecisionEngine()
            print_colored("✅ 博弈论模块初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 博弈论模块初始化部分失败: {e}", Colors.WARNING)
            self.enabled = False

    def set_client(self, client):
        """设置交易客户端"""
        if client:
            self.data_collector = MarketDataCollector(client)

    def analyze(self, market_data: Dict) -> Dict:
        """执行博弈论分析"""
        if not self.enabled:
            return {
                'recommendation': 'NEUTRAL',
                'confidence': 0.5,
                'analysis': '博弈论模块未启用'
            }

        try:
            # 使用分析器分析市场结构
            analysis = self.analyzer.analyze_market_structure(market_data)

            # 使用决策引擎生成交易决策
            decision = self.decision_engine.make_decision(analysis)

            return {
                'recommendation': analysis.get('trading_recommendation', {}).get('action', 'HOLD'),
                'confidence': analysis.get('trading_recommendation', {}).get('confidence', 0.5),
                'analysis': analysis,
                'decision': decision
            }
        except Exception as e:
            self.logger.error(f"博弈论分析错误: {e}")
            return {
                'recommendation': 'NEUTRAL',
                'confidence': 0.0,
                'analysis': f'分析错误: {str(e)}'
            }


class IntegratedDecisionEngine:
    """综合决策引擎"""

    def __init__(self):
        self.logger = logging.getLogger('DecisionEngine')

    def make_decision(self, analysis: Dict) -> Dict:
        """基于分析生成决策"""
        decision = {
            'action': 'HOLD',
            'confidence': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'reasoning': []
        }

        try:
            # 从分析中获取建议
            recommendation = analysis.get('trading_recommendation', {})

            if recommendation:
                decision['action'] = recommendation.get('action', 'HOLD')
                decision['confidence'] = recommendation.get('confidence', 0)
                decision['reasoning'] = recommendation.get('reasoning', [])

                # 添加具体的交易参数
                if decision['action'] != 'HOLD':
                    # 这里可以添加计算入场、止损、止盈的逻辑
                    pass

        except Exception as e:
            self.logger.error(f"决策生成错误: {e}")

        return decision