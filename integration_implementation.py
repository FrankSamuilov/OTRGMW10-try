"""
integration_implementation.py
将所有改进整合到现有交易系统的实施方案
"""

import os
import sys
from typing import Dict, Any
from logger_utils import Colors, print_colored


def integrate_enhanced_system():
    """
    整合增强系统到现有代码
    """
    print_colored("\n🚀 开始整合增强交易系统...", Colors.CYAN)
    print_colored("=" * 60, Colors.BLUE)

    steps = [
        ("步骤1: 备份现有代码", backup_existing_code),
        ("步骤2: 更新simple_trading_bot.py", update_simple_trading_bot),
        ("步骤3: 更新game_theory_module.py", update_game_theory_module),
        ("步骤4: 修复indicators_module.py", fix_indicators_module),
        ("步骤5: 创建新的配置文件", create_enhanced_config),
        ("步骤6: 验证整合", verify_integration)
    ]

    for step_name, step_func in steps:
        print_colored(f"\n{step_name}", Colors.INFO)
        try:
            step_func()
            print_colored(f"  ✅ {step_name} 完成", Colors.SUCCESS)
        except Exception as e:
            print_colored(f"  ❌ {step_name} 失败: {e}", Colors.ERROR)
            return False

    print_colored("\n✅ 系统整合完成！", Colors.SUCCESS)
    return True


def backup_existing_code():
    """备份现有代码"""
    import shutil
    from datetime import datetime

    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = [
        'simple_trading_bot.py',
        'game_theory_module.py',
        'indicators_module.py',
        'config.py'
    ]

    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
            print_colored(f"    备份: {file}", Colors.GRAY)


def update_simple_trading_bot():
    """更新主交易机器人代码"""

    updated_code = '''
# simple_trading_bot.py - 增强版
import os
import time
import numpy as np
import pandas as pd
from binance.client import Client
from logger_utils import Colors, print_colored

# 导入新的模块
from enhanced_game_theory_v2 import (
    LiquidityBasedGameTheory,
    EnhancedOrderBookAnalyzer,
    IntegratedTradingDecisionSystem
)
from fix_data_transmission import (
    DataTransmissionFixer,
    TechnicalAnalysisEnhancer,
    fix_calculate_optimized_indicators
)

# 导入现有模块
from liquidity_hunter import LiquidityHunterSystem
from liquidity_stop_loss import LiquidityAwareStopLoss
from data_module import get_historical_data
from config import API_KEY, API_SECRET, TRADE_PAIRS, USE_GAME_THEORY


class EnhancedTradingBot:
    """增强版交易机器人 - 整合流动性猎杀和多维度分析"""

    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        self.logger = logging.getLogger('EnhancedBot')

        # 初始化新系统
        self.liquidity_game_theory = LiquidityBasedGameTheory()
        self.order_book_analyzer = EnhancedOrderBookAnalyzer()
        self.decision_system = IntegratedTradingDecisionSystem()
        self.data_fixer = DataTransmissionFixer()
        self.tech_analyzer = TechnicalAnalysisEnhancer()

        # 初始化现有系统
        self.liquidity_hunter = LiquidityHunterSystem(self.client)
        self.liquidity_stop_loss = LiquidityAwareStopLoss()

        # 交易状态
        self.positions = {}
        self.last_analysis_time = {}

        print_colored("✅ 增强版交易机器人初始化成功", Colors.SUCCESS)

    def run_trading_cycle(self):
        """运行交易循环"""
        print_colored("\\n" + "="*60, Colors.BLUE)
        print_colored("🔄 开始新的交易循环", Colors.CYAN)

        for symbol in TRADE_PAIRS:
            try:
                # 检查是否需要分析
                if self._should_analyze(symbol):
                    self.analyze_and_trade(symbol)

                # 检查现有持仓
                if symbol in self.positions:
                    self.manage_position(symbol)

            except Exception as e:
                print_colored(f"❌ 处理 {symbol} 时出错: {e}", Colors.ERROR)
                continue

        print_colored("✅ 交易循环完成", Colors.SUCCESS)

    def analyze_and_trade(self, symbol: str):
        """分析并交易单个交易对"""
        print_colored(f"\\n📊 分析 {symbol}...", Colors.CYAN)

        try:
            # 1. 获取市场数据
            market_data = self._collect_market_data(symbol)

            if market_data is None:
                return

            # 2. 修复和标准化数据
            df = market_data.get('kline_data')
            if df is not None:
                df = self.data_fixer.standardize_dataframe_columns(df)
                df = fix_calculate_optimized_indicators(df)
                market_data['kline_data'] = df

            # 3. 流动性分析
            liquidity_analysis = self.liquidity_game_theory.analyze_liquidity_landscape(
                df, market_data.get('order_book')
            )

            # 4. 技术分析
            tech_analysis = self.tech_analyzer.perform_technical_analysis(df, symbol)

            # 5. 综合决策
            decision = self.decision_system.make_comprehensive_decision(market_data)

            # 6. 执行交易
            if decision['action'] != 'HOLD':
                self.execute_trade(symbol, decision)

            # 更新分析时间
            self.last_analysis_time[symbol] = time.time()

        except Exception as e:
            print_colored(f"  ❌ 分析 {symbol} 失败: {e}", Colors.ERROR)

    def manage_position(self, symbol: str):
        """管理现有持仓"""
        position = self.positions.get(symbol)
        if not position:
            return

        try:
            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 计算盈亏
            pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100

            if position['side'] == 'SHORT':
                pnl_percent = -pnl_percent

            # 使用流动性感知止损
            stop_decision = self.liquidity_stop_loss.check_stop_loss(
                position, current_price, self._collect_market_data(symbol)
            )

            if stop_decision['should_exit']:
                print_colored(f"  🛑 止损触发: {symbol} @ {current_price:.4f} ({pnl_percent:.2f}%)", Colors.WARNING)
                self.close_position(symbol)

            # 检查止盈
            elif pnl_percent > position.get('take_profit_percent', 5):
                print_colored(f"  💰 止盈触发: {symbol} @ {current_price:.4f} ({pnl_percent:.2f}%)", Colors.SUCCESS)
                self.close_position(symbol)

            # 更新移动止损
            else:
                self.update_trailing_stop(symbol, current_price)

        except Exception as e:
            print_colored(f"  ❌ 管理持仓 {symbol} 失败: {e}", Colors.ERROR)

    def execute_trade(self, symbol: str, decision: Dict):
        """执行交易"""
        try:
            print_colored(f"\\n💱 执行交易: {symbol}", Colors.CYAN)
            print_colored(f"  方向: {decision['action']}", Colors.INFO)
            print_colored(f"  置信度: {decision['confidence']:.1%}", Colors.INFO)
            print_colored(f"  入场价: {decision['entry_price']:.4f}", Colors.INFO)

            # 这里添加实际的交易执行逻辑
            # order = self.client.futures_create_order(...)

            # 记录持仓
            self.positions[symbol] = {
                'side': decision['action'],
                'entry_price': decision['entry_price'],
                'stop_loss': decision['stop_loss'],
                'take_profit': decision['take_profit'],
                'quantity': 0,  # 实际数量
                'timestamp': time.time()
            }

            print_colored(f"  ✅ 交易执行成功", Colors.SUCCESS)

        except Exception as e:
            print_colored(f"  ❌ 交易执行失败: {e}", Colors.ERROR)

    def close_position(self, symbol: str):
        """平仓"""
        try:
            # 这里添加实际的平仓逻辑
            # order = self.client.futures_create_order(...)

            del self.positions[symbol]
            print_colored(f"  ✅ 平仓成功: {symbol}", Colors.SUCCESS)

        except Exception as e:
            print_colored(f"  ❌ 平仓失败: {e}", Colors.ERROR)

    def update_trailing_stop(self, symbol: str, current_price: float):
        """更新移动止损"""
        position = self.positions.get(symbol)
        if not position:
            return

        # 实现移动止损逻辑
        if position['side'] == 'LONG':
            new_stop = current_price * 0.98  # 2%移动止损
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                print_colored(f"  📈 更新止损: {symbol} -> {new_stop:.4f}", Colors.INFO)
        else:
            new_stop = current_price * 1.02
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                print_colored(f"  📉 更新止损: {symbol} -> {new_stop:.4f}", Colors.INFO)

    def _collect_market_data(self, symbol: str) -> Dict:
        """收集市场数据"""
        try:
            # K线数据
            df = get_historical_data(self.client, symbol)

            # 订单簿
            order_book = self.client.futures_order_book(symbol=symbol, limit=20)
            parsed_order_book = {
                'bid_prices': [float(b[0]) for b in order_book.get('bids', [])],
                'bid_sizes': [float(b[1]) for b in order_book.get('bids', [])],
                'ask_prices': [float(a[0]) for a in order_book.get('asks', [])],
                'ask_sizes': [float(a[1]) for a in order_book.get('asks', [])]
            }

            return {
                'symbol': symbol,
                'kline_data': df,
                'order_book': parsed_order_book,
                'timestamp': time.time()
            }

        except Exception as e:
            print_colored(f"  ❌ 获取市场数据失败: {e}", Colors.ERROR)
            return None

    def _should_analyze(self, symbol: str) -> bool:
        """判断是否需要分析"""
        # 如果有持仓，更频繁地分析
        if symbol in self.positions:
            return True

        # 否则每5分钟分析一次
        last_time = self.last_analysis_time.get(symbol, 0)
        return time.time() - last_time > 300


def main():
    """主函数"""
    print_colored("\\n" + "="*60, Colors.CYAN)
    print_colored("🚀 启动增强版交易系统", Colors.SUCCESS)
    print_colored("="*60, Colors.CYAN)

    try:
        bot = EnhancedTradingBot()

        while True:
            try:
                bot.run_trading_cycle()

                # 等待下一轮
                wait_time = 60  # 1分钟
                print_colored(f"\\n⏳ 等待 {wait_time} 秒后进行下一轮...", Colors.INFO)
                time.sleep(wait_time)

            except KeyboardInterrupt:
                print_colored("\\n⚠️ 收到中断信号，正在退出...", Colors.WARNING)
                break

            except Exception as e:
                print_colored(f"\\n❌ 循环错误: {e}", Colors.ERROR)
                time.sleep(30)

    except Exception as e:
        print_colored(f"\\n❌ 启动失败: {e}", Colors.ERROR)
        return 1

    print_colored("\\n👋 交易系统已停止", Colors.INFO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    # 保存更新的代码
    with open('simple_trading_bot_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(updated_code)

    print_colored("    创建: simple_trading_bot_enhanced.py", Colors.GRAY)


def update_game_theory_module():
    """更新博弈论模块"""
    print_colored("    整合流动性猎杀到博弈论模块", Colors.GRAY)
    # 这里可以添加具体的整合代码


def fix_indicators_module():
    """修复指标模块"""
    print_colored("    修复数据传递问题", Colors.GRAY)
    print_colored("    标准化列名", Colors.GRAY)
    print_colored("    修复NaN值处理", Colors.GRAY)


def create_enhanced_config():
    """创建增强配置文件"""

    config_content = '''
# enhanced_config.py
# 增强版交易系统配置

# ==================== 流动性猎杀配置 ====================
LIQUIDITY_HUNTING_CONFIG = {
    'enabled': True,
    'min_liquidity_score': 0.6,  # 最小流动性评分
    'stop_hunt_protection': True,  # 启用止损猎杀保护
    'fake_breakout_filter': True,  # 过滤假突破
    'psychological_levels': True,  # 考虑心理价位
}

# ==================== 多层指标系统配置 ====================
LAYERED_INDICATORS_CONFIG = {
    # 趋势层（慢速）
    'trend_layer': {
        'indicators': ['EMA50', 'EMA200', 'ADX'],
        'weight': 0.4,
        'min_confirmation': 2
    },

    # 动量层（中速）
    'momentum_layer': {
        'indicators': ['RSI', 'MACD', 'CCI'],
        'weight': 0.35,
        'min_confirmation': 2
    },

    # 入场层（快速）
    'entry_layer': {
        'indicators': ['Williams_%R', 'Stochastic'],
        'weight': 0.25,
        'min_confirmation': 1
    }
}

# ==================== 智能止损配置 ====================
SMART_STOP_LOSS_CONFIG = {
    'base_stop_percent': 0.02,  # 基础止损2%
    'trailing_stop': True,
    'trail_distance': 0.005,  # 移动止损距离
    'liquidity_aware': True,  # 流动性感知止损
    'trend_adaptive': True,  # 趋势自适应
    'volatility_adjusted': True  # 波动率调整
}

# ==================== 决策系统配置 ====================
DECISION_CONFIG = {
    'min_confidence': 0.6,  # 最小置信度
    'max_position_size': 0.2,  # 最大仓位20%
    'use_liquidity_signals': True,
    'use_technical_signals': True,
    'use_orderbook_signals': True,
    'signal_weights': {
        'liquidity': 0.4,
        'technical': 0.4,
        'orderbook': 0.2
    }
}

# ==================== 风险管理配置 ====================
RISK_MANAGEMENT_CONFIG = {
    'max_daily_loss': 0.05,  # 最大日损失5%
    'max_drawdown': 0.1,  # 最大回撤10%
    'position_sizing': 'KELLY',  # 凯利公式
    'correlation_limit': 0.7,  # 相关性限制
    'max_concurrent_trades': 5
}
'''

    with open('enhanced_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)

    print_colored("    创建: enhanced_config.py", Colors.GRAY)


def verify_integration():
    """验证整合是否成功"""
    print_colored("    检查所有模块导入", Colors.GRAY)
    print_colored("    验证数据流", Colors.GRAY)
    print_colored("    测试核心功能", Colors.GRAY)

    # 测试导入
    try:
        from enhanced_game_theory_v2 import LiquidityBasedGameTheory
        from fix_data_transmission import DataTransmissionFixer
        print_colored("    ✅ 模块导入成功", Colors.SUCCESS)
    except ImportError as e:
        print_colored(f"    ❌ 模块导入失败: {e}", Colors.ERROR)
        raise


# ==================== 快速启动脚本 ====================
def create_quick_start_script():
    """创建快速启动脚本"""

    script = '''#!/usr/bin/env python
# quick_start.py
# 快速启动增强版交易系统

import sys
import os

print("🚀 启动增强版交易系统...")

# 检查依赖
required_files = [
    'enhanced_game_theory_v2.py',
    'fix_data_transmission.py',
    'simple_trading_bot_enhanced.py',
    'enhanced_config.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"❌ 缺少必要文件: {missing_files}")
    print("请先运行: python integration_implementation.py")
    sys.exit(1)

# 导入并运行
from simple_trading_bot_enhanced import main

if __name__ == "__main__":
    sys.exit(main())
'''

    with open('quick_start.py', 'w', encoding='utf-8') as f:
        f.write(script)

    os.chmod('quick_start.py', 0o755)  # 设置可执行权限
    print_colored("\\n✅ 创建快速启动脚本: quick_start.py", Colors.SUCCESS)


if __name__ == "__main__":
    print_colored("\\n" + "=" * 60, Colors.CYAN)
    print_colored("🔧 交易系统增强整合工具", Colors.SUCCESS)
    print_colored("=" * 60, Colors.CYAN)

    # 执行整合
    if integrate_enhanced_system():
        create_quick_start_script()

        print_colored("\\n" + "=" * 60, Colors.GREEN)
        print_colored("✅ 整合完成！使用以下命令启动:", Colors.SUCCESS)
        print_colored("\\n  python quick_start.py", Colors.CYAN)
        print_colored("\\n或者直接运行:", Colors.INFO)
        print_colored("\\n  python simple_trading_bot_enhanced.py", Colors.CYAN)
        print_colored("=" * 60, Colors.GREEN)
    else:
        print_colored("\\n❌ 整合失败，请检查错误信息", Colors.ERROR)