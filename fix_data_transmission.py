"""
fix_data_transmission.py
修复技术指标数据传递和显示问题
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from logger_utils import Colors, print_colored


class DataTransmissionFixer:
    """修复数据传递问题的工具类"""

    def __init__(self):
        self.column_mapping = {
            # 标准化列名映射
            'RSI': ['rsi', 'RSI', 'Rsi'],
            'ADX': ['adx', 'ADX', 'Adx'],
            'MACD': ['macd', 'MACD', 'Macd'],
            'ATR': ['atr', 'ATR', 'Atr'],
            'EMA20': ['ema20', 'EMA20', 'ema_20'],
            'EMA50': ['ema50', 'EMA50', 'ema_50'],
            'Williams_%R': ['williams_%r', 'Williams_%R', 'williams', 'Williams'],
            'BB_Upper': ['bb_upper', 'BB_Upper', 'BBands_Upper'],
            'BB_Lower': ['bb_lower', 'BB_Lower', 'BBands_Lower'],
            'BB_Middle': ['bb_middle', 'BB_Middle', 'BBands_Middle'],
        }

    def standardize_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化DataFrame的列名
        确保所有指标使用一致的命名
        """
        if df is None or df.empty:
            return df

        # 创建副本避免修改原始数据
        df_standardized = df.copy()

        # 标准化所有列名
        for standard_name, variations in self.column_mapping.items():
            for variant in variations:
                if variant in df_standardized.columns and standard_name not in df_standardized.columns:
                    df_standardized[standard_name] = df_standardized[variant]
                    if variant != standard_name:
                        print_colored(f"  ✅ 列名标准化: {variant} -> {standard_name}", Colors.INFO)
                    break

        return df_standardized

    def extract_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        安全地提取技术指标值
        处理NaN和缺失列的情况
        """
        indicators = {}

        if df is None or df.empty:
            return self._get_default_indicators()

        # 先标准化列名
        df = self.standardize_dataframe_columns(df)

        # 提取各个指标的最新值
        indicator_configs = [
            ('RSI', 50, 'momentum'),
            ('ADX', 25, 'trend'),
            ('MACD', 0, 'momentum'),
            ('MACD_signal', 0, 'momentum'),
            ('ATR', 0, 'volatility'),
            ('EMA20', None, 'trend'),
            ('EMA50', None, 'trend'),
            ('Williams_%R', -50, 'momentum'),
            ('BB_Upper', None, 'volatility'),
            ('BB_Lower', None, 'volatility'),
            ('BB_Middle', None, 'volatility'),
            ('volume', 0, 'volume'),
            ('OBV', 0, 'volume'),
            ('CCI', 0, 'momentum'),
            ('Stochastic_K', 50, 'momentum'),
            ('Stochastic_D', 50, 'momentum'),
        ]

        for indicator_name, default_value, category in indicator_configs:
            value = self._safe_extract_value(df, indicator_name, default_value)
            indicators[indicator_name] = {
                'value': value,
                'category': category,
                'is_valid': value != default_value and not pd.isna(value)
            }

        # 计算衍生指标
        indicators['bb_position'] = self._calculate_bb_position(df, indicators)
        indicators['volume_ratio'] = self._calculate_volume_ratio(df)
        indicators['trend_direction'] = self._determine_trend(indicators)
        indicators['momentum_status'] = self._determine_momentum(indicators)

        return indicators

    def _safe_extract_value(self, df: pd.DataFrame, column: str, default: Any) -> Any:
        """安全地提取DataFrame中的值"""
        try:
            if column in df.columns:
                value = df[column].iloc[-1]
                if pd.isna(value):
                    return default if default is not None else 0
                return float(value)
            return default if default is not None else 0
        except Exception as e:
            print_colored(f"  ⚠️ 提取 {column} 失败: {e}", Colors.WARNING)
            return default if default is not None else 0

    def _get_default_indicators(self) -> Dict[str, Any]:
        """返回默认指标值"""
        return {
            'RSI': {'value': 50, 'category': 'momentum', 'is_valid': False},
            'ADX': {'value': 25, 'category': 'trend', 'is_valid': False},
            'MACD': {'value': 0, 'category': 'momentum', 'is_valid': False},
            'trend_direction': 'NEUTRAL',
            'momentum_status': 'NEUTRAL',
            'bb_position': 50,
            'volume_ratio': 1.0
        }

    def _calculate_bb_position(self, df: pd.DataFrame, indicators: Dict) -> float:
        """计算价格在布林带中的位置"""
        try:
            if all(indicators[col]['is_valid'] for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                current_price = df['close'].iloc[-1]
                upper = indicators['BB_Upper']['value']
                lower = indicators['BB_Lower']['value']

                if upper > lower:
                    position = ((current_price - lower) / (upper - lower)) * 100
                    return max(0, min(100, position))
            return 50
        except:
            return 50

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """计算成交量比率"""
        try:
            if 'volume' in df.columns and len(df) >= 20:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]

                if avg_volume > 0 and not pd.isna(avg_volume):
                    return current_volume / avg_volume
            return 1.0
        except:
            return 1.0

    def _determine_trend(self, indicators: Dict) -> str:
        """确定趋势方向"""
        try:
            # 基于EMA判断
            if indicators['EMA20']['is_valid'] and indicators['EMA50']['is_valid']:
                if indicators['EMA20']['value'] > indicators['EMA50']['value']:
                    return 'UP'
                elif indicators['EMA20']['value'] < indicators['EMA50']['value']:
                    return 'DOWN'

            # 基于ADX判断
            if indicators['ADX']['is_valid'] and indicators['ADX']['value'] > 25:
                return 'TRENDING'

            return 'NEUTRAL'
        except:
            return 'NEUTRAL'

    def _determine_momentum(self, indicators: Dict) -> str:
        """确定动量状态"""
        try:
            rsi = indicators['RSI']['value'] if indicators['RSI']['is_valid'] else 50
            macd = indicators['MACD']['value'] if indicators['MACD']['is_valid'] else 0

            if rsi > 70 or macd > 0.001:
                return 'BULLISH'
            elif rsi < 30 or macd < -0.001:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'

    def format_indicators_for_display(self, indicators: Dict) -> str:
        """格式化指标用于显示"""
        display_lines = []

        # 主要指标
        if indicators.get('RSI', {}).get('is_valid'):
            rsi_value = indicators['RSI']['value']
            rsi_status = "超买" if rsi_value > 70 else "超卖" if rsi_value < 30 else "中性"
            display_lines.append(f"  ✓ RSI: {rsi_value:.2f} ({rsi_status})")

        if indicators.get('ADX', {}).get('is_valid'):
            adx_value = indicators['ADX']['value']
            trend_strength = "强" if adx_value > 40 else "中" if adx_value > 25 else "弱"
            display_lines.append(f"  ✓ ADX: {adx_value:.2f} (趋势{trend_strength})")

        # 布林带位置
        bb_pos = indicators.get('bb_position', 50)
        bb_status = "超买区" if bb_pos > 80 else "超卖区" if bb_pos < 20 else "中性区"
        display_lines.append(f"  ✓ 布林带位置: {bb_pos:.1f}% ({bb_status})")

        # 成交量
        vol_ratio = indicators.get('volume_ratio', 1.0)
        vol_status = "放量" if vol_ratio > 1.5 else "缩量" if vol_ratio < 0.7 else "正常"
        display_lines.append(f"  ✓ 成交量比率: {vol_ratio:.2f}x ({vol_status})")

        # 趋势和动量
        display_lines.append(f"  ✓ 趋势: {indicators.get('trend_direction', 'NEUTRAL')}")
        display_lines.append(f"  ✓ 动量: {indicators.get('momentum_status', 'NEUTRAL')}")

        return "\n".join(display_lines)


def fix_calculate_optimized_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复版本的 calculate_optimized_indicators
    确保所有指标正确计算并保存到DataFrame
    """
    if df is None or df.empty:
        return df

    try:
        # 确保基础列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print_colored("  ⚠️ 缺少必要的OHLCV列", Colors.WARNING)
            return df

        # 使用副本避免警告
        df = df.copy()

        # 1. 趋势指标
        df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

        # 2. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 3. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # 4. 布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # 5. ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # 6. ADX
        df['ADX'] = calculate_adx(df)

        # 7. Williams %R
        df['Williams_%R'] = calculate_williams_r(df)

        # 8. OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # 9. VWAP
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # 10. CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma) / (0.015 * mad + 1e-10)

        # 填充NaN值
        df = df.fillna(method='ffill').fillna(0)

        print_colored(f"  ✅ 计算完成 {len([col for col in df.columns if col not in required_columns])} 个技术指标",
                      Colors.SUCCESS)

        # 验证关键指标
        key_indicators = ['RSI', 'ADX', 'MACD', 'ATR', 'EMA20', 'EMA50']
        for indicator in key_indicators:
            if indicator in df.columns:
                last_value = df[indicator].iloc[-1]
                if not pd.isna(last_value):
                    print_colored(f"    • {indicator}: {last_value:.4f}", Colors.GRAY)

        return df

    except Exception as e:
        print_colored(f"  ❌ 指标计算错误: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()
        return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ADX指标"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']

        # 计算+DM和-DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # 计算TR
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=period).mean()

        # 计算+DI和-DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # 计算DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # 计算ADX
        adx = dx.rolling(window=period).mean()

        return adx

    except Exception as e:
        print_colored(f"    ⚠️ ADX计算失败: {e}", Colors.WARNING)
        return pd.Series([25] * len(df))


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算威廉指标"""
    try:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low + 1e-10))

        return williams_r

    except Exception as e:
        print_colored(f"    ⚠️ Williams %R计算失败: {e}", Colors.WARNING)
        return pd.Series([-50] * len(df))


class TechnicalAnalysisEnhancer:
    """增强的技术分析类 - 修复数据传递问题"""

    def __init__(self):
        self.fixer = DataTransmissionFixer()

    def perform_technical_analysis(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """
        执行技术分析并确保数据正确传递
        """
        print_colored(f"\n📈 执行技术分析 {symbol if symbol else ''}...", Colors.CYAN)

        analysis_result = {
            'indicators': {},
            'signals': {},
            'market_state': 'NEUTRAL',
            'recommendation': None
        }

        try:
            # 1. 计算所有指标
            df = fix_calculate_optimized_indicators(df)

            # 2. 提取指标值
            indicators = self.fixer.extract_technical_indicators(df)
            analysis_result['indicators'] = indicators

            # 3. 生成交易信号
            signals = self._generate_signals(indicators)
            analysis_result['signals'] = signals

            # 4. 确定市场状态
            market_state = self._determine_market_state(indicators, signals)
            analysis_result['market_state'] = market_state

            # 5. 生成建议
            recommendation = self._generate_recommendation(market_state, signals, indicators)
            analysis_result['recommendation'] = recommendation

            # 6. 打印结果
            print_colored("\n📊 技术分析完成:", Colors.SUCCESS)
            print(self.fixer.format_indicators_for_display(indicators))

            if recommendation:
                self._print_recommendation(recommendation)

        except Exception as e:
            print_colored(f"  ❌ 技术分析失败: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()

        return analysis_result

    def _generate_signals(self, indicators: Dict) -> Dict:
        """生成交易信号"""
        signals = {
            'rsi_signal': 'NEUTRAL',
            'macd_signal': 'NEUTRAL',
            'trend_signal': 'NEUTRAL',
            'volume_signal': 'NEUTRAL',
            'overall': 'NEUTRAL'
        }

        # RSI信号
        if indicators['RSI']['is_valid']:
            rsi = indicators['RSI']['value']
            if rsi > 70:
                signals['rsi_signal'] = 'OVERBOUGHT'
            elif rsi < 30:
                signals['rsi_signal'] = 'OVERSOLD'

        # MACD信号
        if indicators['MACD']['is_valid']:
            macd = indicators['MACD']['value']
            if macd > 0:
                signals['macd_signal'] = 'BULLISH'
            elif macd < 0:
                signals['macd_signal'] = 'BEARISH'

        # 趋势信号
        signals['trend_signal'] = indicators.get('trend_direction', 'NEUTRAL')

        # 成交量信号
        vol_ratio = indicators.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:
            signals['volume_signal'] = 'HIGH'
        elif vol_ratio < 0.5:
            signals['volume_signal'] = 'LOW'

        # 综合信号
        bullish_count = sum([
            signals['rsi_signal'] == 'OVERSOLD',
            signals['macd_signal'] == 'BULLISH',
            signals['trend_signal'] == 'UP',
            signals['volume_signal'] == 'HIGH'
        ])

        bearish_count = sum([
            signals['rsi_signal'] == 'OVERBOUGHT',
            signals['macd_signal'] == 'BEARISH',
            signals['trend_signal'] == 'DOWN',
            signals['volume_signal'] == 'LOW'
        ])

        if bullish_count >= 3:
            signals['overall'] = 'BULLISH'
        elif bearish_count >= 3:
            signals['overall'] = 'BEARISH'

        return signals

    def _determine_market_state(self, indicators: Dict, signals: Dict) -> str:
        """确定市场状态"""
        adx = indicators.get('ADX', {}).get('value', 25)

        if adx < 20:
            return 'RANGING'
        elif adx > 40:
            if signals['overall'] == 'BULLISH':
                return 'STRONG_UPTREND'
            elif signals['overall'] == 'BEARISH':
                return 'STRONG_DOWNTREND'
            else:
                return 'STRONG_TREND'
        else:
            if signals['overall'] == 'BULLISH':
                return 'UPTREND'
            elif signals['overall'] == 'BEARISH':
                return 'DOWNTREND'
            else:
                return 'NEUTRAL'

    def _generate_recommendation(self, market_state: str, signals: Dict, indicators: Dict) -> Dict:
        """生成交易建议"""
        recommendation = {
            'action': 'HOLD',
            'confidence': 0,
            'reasons': []
        }

        if market_state in ['STRONG_UPTREND', 'UPTREND'] and signals['overall'] == 'BULLISH':
            recommendation['action'] = 'BUY'
            recommendation['confidence'] = 0.8 if market_state == 'STRONG_UPTREND' else 0.6
            recommendation['reasons'] = ['上升趋势确认', '技术指标看涨']

        elif market_state in ['STRONG_DOWNTREND', 'DOWNTREND'] and signals['overall'] == 'BEARISH':
            recommendation['action'] = 'SELL'
            recommendation['confidence'] = 0.8 if market_state == 'STRONG_DOWNTREND' else 0.6
            recommendation['reasons'] = ['下降趋势确认', '技术指标看跌']

        elif market_state == 'RANGING':
            # 震荡市场策略
            rsi = indicators.get('RSI', {}).get('value', 50)
            if rsi < 30:
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = 0.5
                recommendation['reasons'] = ['震荡市场超卖']
            elif rsi > 70:
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = 0.5
                recommendation['reasons'] = ['震荡市场超买']

        return recommendation

    def _print_recommendation(self, recommendation: Dict):
        """打印交易建议"""
        if recommendation['action'] != 'HOLD':
            color = Colors.GREEN if recommendation['action'] == 'BUY' else Colors.RED
            print_colored(f"\n  📌 建议: {recommendation['action']} (置信度: {recommendation['confidence']:.1%})", color)
            for reason in recommendation['reasons']:
                print_colored(f"    • {reason}", Colors.GRAY)


# 使用示例
if __name__ == "__main__":
    # 测试数据传递修复
    print_colored("测试数据传递修复功能...", Colors.CYAN)

    # 创建测试DataFrame
    test_df = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100) * 1.1,
        'low': np.random.rand(100) * 0.9,
        'close': np.random.rand(100),
        'volume': np.random.rand(100) * 1000
    })

    # 测试修复功能
    fixer = DataTransmissionFixer()
    enhancer = TechnicalAnalysisEnhancer()

    # 执行分析
    result = enhancer.perform_technical_analysis(test_df, "TEST_SYMBOL")

    print_colored("\n✅ 测试完成!", Colors.SUCCESS)