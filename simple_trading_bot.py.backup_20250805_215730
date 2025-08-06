import os
import time
import math
from liquidity_hunter import LiquidityHunterSystem
from liquidity_stop_loss import LiquidityAwareStopLoss
import numpy as np
import pandas as pd
import datetime
import logging
from binance.client import Client
import importlib
import sys
import config
from advanced_pattern_recognition import AdvancedPatternRecognition
from market_auction_analyzer import MarketAuctionAnalyzer
from enhanced_scoring_system import EnhancedScoringSystem
from position_visualizer import PositionVisualizer
from data_module import get_historical_data
from indicators_module import calculate_optimized_indicators, get_smc_trend_and_duration, find_swing_points, \
    calculate_fibonacci_retracements
from position_module import load_positions, get_total_position_exposure, calculate_order_amount, \
    adjust_position_for_market_change
from logger_setup import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_module import get_max_leverage, get_precise_quantity, format_quantity
from quality_module import calculate_quality_score, detect_pattern_similarity, adjust_quality_for_similarity
from pivot_points_module import calculate_pivot_points, analyze_pivot_point_strategy
from advanced_indicators import calculate_smi, calculate_stochastic, calculate_parabolic_sar
from smc_enhanced_prediction import enhanced_smc_prediction, multi_timeframe_smc_prediction
from risk_management import adaptive_risk_management
from integration_module import calculate_enhanced_indicators, comprehensive_market_analysis, generate_trade_recommendation
from logger_utils import Colors, print_colored
import datetime
import time
from integration_module import calculate_enhanced_indicators, generate_trade_recommendation
from multi_timeframe_module import MultiTimeframeCoordinator
from config import (
    API_KEY,
    API_SECRET,
    TRADE_PAIRS,
    USE_GAME_THEORY,
    MIN_MARGIN_BALANCE,
    ORDER_AMOUNT_PERCENT,
    MAX_POSITIONS,
    MAX_DAILY_TRADES,
    TAKE_PROFIT_PERCENT,
    STOP_LOSS_PERCENT,
    SCAN_INTERVAL,
    MIN_SCORE,
    MAX_CONCURRENT_TRADES,
    GAME_THEORY_CONFIG,
    CONFIG
)
# 在现有导入之后添加
import nest_asyncio
nest_asyncio.apply()  # 允许在已有事件循环中运行异步代码
# 导入新的博弈论模块
from game_theory_module import (
    MarketDataCollector,
    SMCGameTheoryAnalyzer,
    IntegratedDecisionEngine
)
from auction_module import (
    AuctionTheoryFramework,
    AuctionOrderFlowAnalyzer
)
from market_microstructure import (
    OrderFlowToxicityAnalyzer,
    SmartMoneyTracker
)

from smart_trailing_stop import SmartTrailingStop
from atr_dynamic_stop import ATRDynamicStopLoss
from rvi_indicator import rvi_entry_filter, rvi_exit_signal

# 导入集成模块（这是最简单的方法，因为它整合了所有其他模块的功能）
from integration_module import (
    calculate_enhanced_indicators,
    comprehensive_market_analysis,
    generate_trade_recommendation
)
from enhanced_game_theory import EnhancedGameTheoryAnalyzer
import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys  # 这个必须有！
import time
import logging
import json
import nest_asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from risk_manager import RiskManager  # 假设您将上面的代码保存为risk_manager.py
# 设置基本变量（防止导入失败）
from game_theory_module import GameTheoryModule
VERSION = "2.0.0"
from performance_monitor import PerformanceMonitor
from auction_module import (
    AuctionTheoryFramework,
    AuctionOrderFlowAnalyzer,
    AuctionManipulationDetector
)
from trend_aware_indicators import TrendAwareRSI
from signal_stabilizer import SignalStabilizer
from dynamic_weight_manager import DynamicWeightManager
from enhanced_game_theory import EnhancedGameTheoryAnalyzer
from smart_trailing_stop import SmartTrailingStop
from atr_dynamic_stop import ATRDynamicStopLoss
from rvi_indicator import rvi_entry_filter, rvi_exit_signal, calculate_rvi
from volume_spike_detector import VolumeSpikDetector
from enhanced_scoring_system import EnhancedScoringSystem
from advanced_pattern_recognition import AdvancedPatternRecognition
from market_auction_analyzer import MarketAuctionAnalyzer
from enhanced_scoring_system import EnhancedScoringSystem

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored


def setup_logger(name='TradingBot', log_file='logs/trading_bot.log'):
    """设置日志记录器"""
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 清除现有处理器（避免重复）
    logger.handlers.clear()

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class SimpleTradingBot:
    def __init__(self, client=None, config=None):
        """初始化交易机器人 - 兼容两种调用方式"""

        # ==================== 1. 首先处理配置 ====================
        if config:
            self.config = config
        else:
            from config import CONFIG
            self.config = CONFIG

        # ==================== 2. 初始化日志 ====================
        self.logger = setup_logger('TradingBot', 'logs/trading_bot.log')
        self.logger.info("交易机器人启动", extra={"version": VERSION})

        # ==================== 3. 处理客户端 ====================
        if client:
            self.client = client
        else:
            self.client = Client(
                api_key=self.config['API_KEY'],
                api_secret=self.config['API_SECRET']
            )

        # ==================== 4. 测试连接 ====================
        self._test_connection()

        # ==================== 5. 设置控制台编码 ====================
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')

        # ==================== 6. 初始化基础属性 ====================
        self.start_time = time.time()
        self.resource_management_start_time = time.time()
        self.trade_cycle = 0
        self.open_positions = []
        self.max_positions = self.config.get('MAX_POSITIONS', 5)

        # 初始化持仓锁
        from threading import Lock
        self.position_lock = Lock()

        # 交易计数和统计
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = 0

        # 历史数据缓存
        self.historical_data_cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 3600  # 每小时清理一次

        # 偏向控制
        self.preferred_direction = None
        self.last_bias_update = 0

        # 质量分数历史（用于优化）
        self.quality_score_history = {}

        # 交易历史
        self.trade_history = []
        self.position_history = []
        self._load_position_history()

        # 相似交易模式跟踪
        self.similar_patterns_history = {}

        # 机器人状态
        self.is_running = False
        self.last_scan_time = 0
        self.signal_history = {}
        self.signal_smoothing_window = 3
        self.last_whale_intent = {}
        self.intent_change_count = {}
        # ==================== 7. 初始化基础组件 ====================
        # GameTheoryModule
        try:
            self.game_theory = GameTheoryModule()
            print_colored("✅ 博弈论模块初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ GameTheoryModule 初始化失败: {e}", Colors.WARNING)
            self.game_theory = None

        # RiskManager
        try:
            # RiskManager 需要具体的风险参数，而不是 config 和 logger
            self.risk_manager = RiskManager(
                max_position_size=self.config.get('MAX_POSITION_SIZE_PERCENT', 30.0),
                max_daily_loss=self.config.get('MAX_DAILY_LOSS_PERCENT', 5.0),
                max_drawdown=self.config.get('MAX_DRAWDOWN_PERCENT', 10.0),
                base_risk_per_trade=2.0  # 默认每笔交易风险2%
            )
        except Exception as e:
            print_colored(f"⚠️ RiskManager 初始化失败: {e}", Colors.WARNING)
            self.risk_manager = None

        # PerformanceMonitor
        try:
            # PerformanceMonitor 需要保存目录路径，而不是 logger
            self.performance_monitor = PerformanceMonitor(save_dir="performance_data")
        except Exception as e:
            print_colored(f"⚠️ PerformanceMonitor 初始化失败: {e}", Colors.WARNING)
            self.performance_monitor = None

        # ==================== 8. 添加新的智能止损组件 ====================
        try:
            from smart_trailing_stop import SmartTrailingStop
            from atr_dynamic_stop import ATRDynamicStopLoss

            self.smart_trailing_stop = SmartTrailingStop(self.logger)
            self.atr_stop_loss = ATRDynamicStopLoss(base_multiplier=2.0, logger=self.logger)
        except Exception as e:
            self.logger.error(f"止损组件初始化失败: {e}")
            print_colored(f"⚠️ 止损组件初始化失败: {e}", Colors.WARNING)
            self.smart_trailing_stop = None
            self.atr_stop_loss = None

        # ==================== 9. 初始化多时间框架协调器 ====================
        try:
            from multi_timeframe_module import MultiTimeframeCoordinator
            self.mtf_coordinator = MultiTimeframeCoordinator(self.client, self.logger)
        except Exception as e:
            self.logger.warning(f"多时间框架协调器初始化失败: {e}")
            self.mtf_coordinator = None

        # ==================== 10. 博弈论相关组件初始化 ====================
        self.use_game_theory = self.config.get("USE_GAME_THEORY", True)
        if self.use_game_theory:
            self._initialize_game_theory_components()

        # ==================== 11. 完成初始化 ====================
        print(f"✅ 交易机器人初始化完成 v{VERSION}")
        self.logger.info("初始化完成")

        # ==================== 13. 初始化交易优化组件 ====================
        try:
            self.trend_aware_rsi = TrendAwareRSI()
            self.signal_stabilizer = SignalStabilizer(
                confirmation_periods=self.config.get('SIGNAL_CONFIRMATION_PERIODS', 3),
                min_holding_periods=self.config.get('MIN_HOLDING_PERIODS', 5)
            )
            self.weight_manager = DynamicWeightManager()
            print_colored("✅ 高级交易优化系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 高级交易优化系统初始化失败: {e}", Colors.WARNING)
            self.logger.warning(f"高级交易优化系统初始化失败: {e}")
            self.trend_aware_rsi = None
            self.signal_stabilizer = None
            self.weight_manager = None

        # ==================== 初始化流动性猎手系统（需要在流动性感知止损之前）====================
        try:
            self.liquidity_hunter = LiquidityHunterSystem(self.client, self.logger)
            print_colored("✅ 流动性分析系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 流动性分析系统初始化失败: {e}", Colors.WARNING)
            self.liquidity_hunter = None

            # ==================== 14. 初始化流动性感知止损系统 ====================
        try:
            if hasattr(self, 'liquidity_hunter') and self.liquidity_hunter:
                self.liquidity_stop_loss = LiquidityAwareStopLoss(
                     liquidity_hunter=self.liquidity_hunter,
                     logger=self.logger
                 )
                print_colored("✅ 流动性感知止损系统初始化成功", Colors.GREEN)
            else:
                self.liquidity_stop_loss = None
                print_colored("⚠️ 流动性感知止损需要先初始化流动性猎手系统", Colors.WARNING)
        except Exception as e:
            print_colored(f"⚠️ 流动性感知止损初始化失败: {e}", Colors.WARNING)
            self.liquidity_stop_loss = None

    # ==================== 15. 初始化持仓可视化系统 ====================
        try:
            self.position_visualizer = PositionVisualizer(
            liquidity_hunter=getattr(self, 'liquidity_hunter', None),
            logger=self.logger
            )
            print_colored("✅ 持仓可视化系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 持仓可视化系统初始化失败: {e}", Colors.WARNING)
            self.position_visualizer = None

    # ==================== 初始化增强评分系统 ====================
        try:
            self.volume_spike_detector = VolumeSpikDetector()
            self.enhanced_scorer = EnhancedScoringSystem()
            print_colored("✅ 增强评分系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 增强评分系统初始化失败: {e}", Colors.WARNING)
            self.volume_spike_detector = None
            self.enhanced_scorer = None

        # ==================== 初始化高级形态识别系统 ====================

        try:
            self.pattern_recognition = AdvancedPatternRecognition(self.logger)
            self.market_auction = MarketAuctionAnalyzer(self.logger)
            self.scoring_system = EnhancedScoringSystem(self.logger)
            print_colored("✅ 高级分析系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 高级分析系统初始化失败: {e}", Colors.WARNING)
            self.pattern_recognition = None
            self.market_auction = None
            self.scoring_system = None

    def init_enhanced_stop_loss(self):
        """初始化增强止损系统"""
        try:
            # 确保已有流动性猎手
            if hasattr(self, 'liquidity_hunter') and self.liquidity_hunter:
                self.liquidity_stop_loss = LiquidityAwareStopLoss(
                    liquidity_hunter=self.liquidity_hunter,
                    logger=self.logger
                )
                print_colored("✅ 流动性感知止损系统初始化成功", Colors.GREEN)
            else:
                self.liquidity_stop_loss = None
                print_colored("⚠️ 流动性感知止损需要先初始化流动性猎手系统", Colors.WARNING)
        except Exception as e:
            print_colored(f"⚠️ 流动性感知止损初始化失败: {e}", Colors.WARNING)
            self.liquidity_stop_loss = None

    def init_position_visualizer(self):
        """初始化持仓可视化系统"""
        try:
            self.position_visualizer = PositionVisualizer(
                liquidity_hunter=getattr(self, 'liquidity_hunter', None),
                logger=self.logger
            )
            print_colored("✅ 持仓可视化系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 持仓可视化系统初始化失败: {e}", Colors.WARNING)
            self.position_visualizer = None

    def analyze_market_with_auction_theory(self, df: pd.DataFrame, symbol: str) -> Dict:
        """使用拍卖理论分析市场"""

        if not self.auction_analyzer:
            return {}

        try:
            # 获取市场数据
            market_data = {
                'symbol': symbol,
                'order_book': self.get_order_book_safe(symbol),
                'long_short_ratio': self.get_long_short_ratio_safe(symbol)
            }

            # 使用拍卖理论分析
            analysis = self.auction_game_integration.analyze_with_game_theory(df, market_data)

            # 打印分析结果
            if analysis.get('combined_signal', {}).get('action') != 'HOLD':
                signal = analysis['combined_signal']
                print_colored(
                    f"🎯 拍卖理论信号: {signal['action']} "
                    f"(置信度: {signal['confidence']:.2f})",
                    Colors.GREEN if signal['action'] == 'BUY' else Colors.RED
                )
                for reason in signal.get('reasoning', []):
                    print_colored(f"  - {reason}", Colors.INFO)

            return analysis

        except Exception as e:
            self.logger.error(f"拍卖理论分析错误: {e}")
            return {}

    def _check_account_status(self):
        """检查账户状态"""
        try:
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            available = float(account['availableBalance'])

            print_colored(f"💰 账户余额: {balance:.2f} USDT", Colors.INFO)
            print_colored(f"💰 可用余额: {available:.2f} USDT", Colors.INFO)

        except Exception as e:
            self.logger.error(f"获取账户状态失败: {e}")

    def scan_for_opportunities(self, account_balance: float):
        """扫描交易机会"""
        trade_pairs = self.config.get('TRADE_PAIRS', [])

        print_colored(f"扫描 {len(trade_pairs)} 个交易对...", Colors.INFO)

        for symbol in trade_pairs:
            try:
                # 检查是否已有持仓
                if self.has_position(symbol):
                    continue

                # 获取历史数据
                df = self.get_historical_data_with_cache(symbol)
                if df is None:
                    continue

                # 生成交易信号
                signal, score = self.generate_trade_signal(df, symbol)

                # 如果有信号且评分足够高
                min_score = self.config.get('MIN_SCORE', 6.0)
                if signal != "HOLD" and score >= min_score:
                    print_colored(f"✅ {symbol}: {signal} 信号 (评分: {score:.2f})", Colors.GREEN),

                # 这里可以执行交易
                # self.execute_trade(symbol, signal, account_balance)

            except Exception as e:
                self.logger.error(f"扫描 {symbol} 时出错: {e}")
                continue

    def _initialize_game_theory_components(self):
        """初始化博弈论系统组件"""
        print_colored("🎯 正在初始化博弈论系统...", Colors.CYAN)

        try:
            # 尝试从game_theory_module导入
            try:
                from game_theory_module import MarketDataCollector, SMCGameTheoryAnalyzer, IntegratedDecisionEngine

                # 初始化基础组件
                self.data_collector = MarketDataCollector(self.client) if hasattr(MarketDataCollector,
                                                                                  '__init__') else None
                self.game_analyzer = SMCGameTheoryAnalyzer() if hasattr(SMCGameTheoryAnalyzer, '__init__') else None
                self.decision_engine = IntegratedDecisionEngine() if hasattr(IntegratedDecisionEngine,
                                                                             '__init__') else None
            except ImportError as e:
                self.logger.warning(f"game_theory_module 导入失败: {e}")
                self.data_collector = None
                self.game_analyzer = None
                self.decision_engine = None

            # 拍卖理论组件
            try:
                from auction_module import (
                    AuctionTheoryFramework,
                    AuctionOrderFlowAnalyzer,
                    AuctionManipulationDetector
                )

                self.auction_analyzer = AuctionTheoryFramework()
                self.auction_manipulator = AuctionManipulationDetector()
                self.order_flow_analyzer = AuctionOrderFlowAnalyzer()
            except ImportError as e:
                self.logger.warning(f"拍卖模块导入失败: {e}")
                self.auction_analyzer = None
                self.auction_manipulator = None
                self.order_flow_analyzer = None

            # 市场微观结构组件
            try:
                from market_microstructure import (
                    OrderFlowToxicityAnalyzer,
                    SmartMoneyTracker
                )

                self.toxicity_analyzer = OrderFlowToxicityAnalyzer()
                self.smart_money_tracker = SmartMoneyTracker()
            except ImportError as e:
                self.logger.warning(f"市场微观结构模块导入失败: {e}")
                self.toxicity_analyzer = None
                self.smart_money_tracker = None

            # 尝试导入可选组件
            try:
                from market_microstructure import MicrostructureArbitrage
                self.arbitrage_detector = MicrostructureArbitrage()
            except:
                self.logger.warning("MicrostructureArbitrage 未找到，跳过")
                self.arbitrage_detector = None

            # 数据缓存
            self.market_data_cache = {}
            self.order_book_cache = {}
            self.order_book_history = {}

            # 检查是否有任何组件成功初始化
            if any([self.data_collector, self.game_analyzer, self.decision_engine,
                    self.auction_analyzer, self.toxicity_analyzer]):
                self.logger.info("✅ 博弈论系统部分组件初始化成功")
                print_colored("✅ 博弈论系统部分组件初始化成功", Colors.GREEN)
            else:
                raise Exception("所有博弈论组件初始化失败")

        except Exception as e:
            self.logger.error(f"博弈论系统初始化失败: {e}")
            print_colored(f"⚠️ 博弈论系统初始化失败，使用传统模式: {e}", Colors.WARNING)
            self.use_game_theory = False

            # 设置必要的占位符
            self.data_collector = None
            self.game_analyzer = None
            self.decision_engine = None
            self.auction_analyzer = None
            self.auction_manipulator = None
            self.order_flow_analyzer = None
            self.toxicity_analyzer = None
            self.smart_money_tracker = None
            self.arbitrage_detector = None

    def run_trading_cycle(self):
        """主交易循环"""
        try:
            self.trade_cycle += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"📊 交易循环 #{self.trade_cycle} - {current_time}", Colors.BLUE + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            # 1. 检查账户余额
            try:
                account_info = self.client.futures_account()
                account_balance = float(account_info['totalWalletBalance'])
                available_balance = float(account_info['availableBalance'])

                print_colored(f"💰 账户余额: {account_balance:.2f} USDT", Colors.INFO)
                print_colored(f"💰 可用余额: {available_balance:.2f} USDT", Colors.INFO)

            except Exception as e:
                self.logger.error(f"获取账户信息失败: {e}")
                print_colored(f"❌ 获取账户信息失败: {e}", Colors.ERROR)
                return

            # 2. 检查风险管理状态（如果有风险管理器）
            if hasattr(self, 'risk_manager') and self.risk_manager:
                can_trade, reason = self.risk_manager.can_open_position()
                if not can_trade:
                    print_colored(f"⚠️ 风险管理限制: {reason}", Colors.WARNING)
                    return

            # 3. 管理现有持仓
            print_colored("\n📋 检查现有持仓...", Colors.CYAN)
            self.manage_open_positions()

            # 4. 检查是否可以开新仓
            if len(self.open_positions) >= self.config.get('MAX_POSITIONS', 5):
                print_colored(f"⚠️ 已达到最大持仓数 ({len(self.open_positions)}/{self.config['MAX_POSITIONS']})",
                              Colors.WARNING)
                return

            # 5. 扫描交易机会
            print_colored("\n🔍 扫描交易机会...", Colors.CYAN)
            self.scan_for_opportunities(account_balance)

            # 6. 定期维护
            if self.trade_cycle % 12 == 0:  # 每小时执行一次
                self.perform_maintenance()

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}")
            print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()

    def perform_maintenance(self):
        """执行定期维护任务"""
        print_colored("\n🔧 执行定期维护...", Colors.CYAN)

        # 清理缓存
        if hasattr(self, 'cleanup_cache_if_needed'):
            self.cleanup_cache_if_needed()

        # 保存持仓历史
        if hasattr(self, '_save_position_history'):
            self._save_position_history()

        print_colored("✅ 维护任务完成", Colors.GREEN)

    def _run_integrated_analysis(self, account_balance: float):
        """
        运行整合分析 - 结合博弈论和技术分析
        """
        print_colored("\n🎯 运行整合式市场分析...", Colors.CYAN + Colors.BOLD)
        print_colored("=" * 80, Colors.BLUE)

        # 初始化增强版分析器
        if not hasattr(self, 'enhanced_analyzer'):
            self.enhanced_analyzer = EnhancedGameTheoryAnalyzer(self.client)

        # 分析结果收集
        trading_opportunities = []

        for idx, symbol in enumerate(self.config["TRADE_PAIRS"], 1):
            if self.has_position(symbol):
                print_colored(f"⏭️ {symbol} - 已有持仓，跳过分析", Colors.GRAY)
                continue

            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"📊 综合分析 {symbol} ({idx}/{len(self.config['TRADE_PAIRS'])})", Colors.BLUE + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            try:
                # 1. 执行博弈论分析（市场微观结构）
                df = self.get_market_data_sync(symbol)
                if df is None or df.empty:
                    print_colored(f"❌ {symbol} K线数据获取失败", Colors.ERROR)
                    game_theory_analysis = {
                        'whale_intent': 'NEUTRAL',
                        'confidence': 0.5,
                        'recommendation': 'HOLD',
                        'signals': []
                    }
                else:
                    # 获取订单簿
                    try:
                        depth_data = self.client.futures_order_book(symbol=symbol, limit=500)
                    except:
                        depth_data = {'bids': [], 'asks': []}

                    # 现在传入所有必需的参数
                    game_theory_analysis = self.enhanced_analyzer.analyze_market_intent(symbol, df, depth_data)

                # 2. 执行技术分析
                print_colored(f"\n📈 执行传统技术分析...", Colors.INFO)
                technical_analysis = self._perform_technical_analysis(symbol)

                # 3. 整合分析结果
                print_colored(f"\n🔗 整合博弈论与技术分析...", Colors.INFO)
                if self.config.get('USE_TREND_PRIORITY', True):
                    integrated_decision = self._integrate_analyses_trend_first(
                        game_theory_analysis,
                        technical_analysis,
                        symbol
                    )
                else:
                    integrated_decision = self._integrate_analyses(
                        game_theory_analysis,
                        technical_analysis,
                        symbol
                    )

                # 4. 计算风险调整后的交易参数
                if integrated_decision['action'] != 'HOLD':
                    print_colored(f"\n💡 计算风险调整参数...", Colors.INFO)
                    trade_params = self._calculate_risk_adjusted_params(
                        integrated_decision,
                        account_balance,
                        symbol
                    )

                    if trade_params:
                        integrated_decision['trade_params'] = trade_params
                        trading_opportunities.append(integrated_decision)

                        # 显示交易机会详情
                        self._display_trading_opportunity(integrated_decision)
                else:
                    print_colored(f"\n❌ 综合分析结果: 不建议交易", Colors.YELLOW)
                    print_colored(f"   原因: {integrated_decision.get('reason', '信号不一致或风险过高')}", Colors.INFO)

            except Exception as e:
                self.logger.error(f"分析{symbol}失败: {e}")
                print_colored(f"\n❌ 分析失败: {str(e)}", Colors.ERROR)

        print_colored(f"\n{'=' * 80}", Colors.BLUE)
        print_colored(f"📊 分析完成汇总", Colors.CYAN + Colors.BOLD)
        print_colored(f"{'=' * 80}", Colors.BLUE)

        # 按综合评分排序
        trading_opportunities.sort(key=lambda x: x['final_score'], reverse=True)

        if trading_opportunities:
            print_colored(f"\n✅ 发现 {len(trading_opportunities)} 个交易机会", Colors.GREEN)

            # 选择最佳机会
            best_opportunity = trading_opportunities[0]
            print_colored(f"\n🏆 最佳交易机会: {best_opportunity['symbol']}", Colors.GREEN + Colors.BOLD)
            print_colored(f"   • 方向: {best_opportunity['action']}", Colors.INFO)
            print_colored(f"   • 综合评分: {best_opportunity['final_score']:.2f}/10", Colors.INFO)
            print_colored(f"   • 预期风险回报比: 1:{best_opportunity['trade_params']['risk_reward_ratio']:.1f}",
                          Colors.INFO)

            # 执行交易
            print_colored(f"\n💫 准备执行交易...", Colors.CYAN)
            self._execute_integrated_trade(best_opportunity, account_balance)

        else:
            print_colored(f"\n⚠️ 未发现合适的交易机会", Colors.WARNING)
            print_colored(f"   建议: 继续观察市场，等待更明确的信号", Colors.INFO)

    def get_market_data_sync(self, symbol: str, interval: str = '5m', limit: int = 500) -> pd.DataFrame:
        """同步获取市场数据 - 修复版本"""
        try:
            # 替换 Colors.INFO 为 Colors.INFO
            print_colored(f"    📊 正在获取 {symbol} 的K线数据...", Colors.INFO)

            # 获取K线数据
            klines = None

            # 方法1：使用期货K线
            try:
                if hasattr(self.client, 'futures_klines'):
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit
                    )
            except:
                pass

            # 方法2：使用现货K线
            if not klines:
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit
                    )
                except Exception as e:
                    print_colored(f"    ⚠️ 获取K线失败: {e}", Colors.WARNING)
                    # 返回空DataFrame而不是抛出异常
                    return pd.DataFrame()

            if not klines:
                print_colored(f"    ⚠️ 未获取到K线数据", Colors.WARNING)
                return pd.DataFrame()

            # 转换为 DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])

            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 转换数值列
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 删除NaN
            df = df.dropna()

            print_colored(f"    ✅ 获取到 {len(df)} 条数据", Colors.GREEN)
            return df

        except Exception as e:
            self.logger.error(f"获取市场数据失败 {symbol}: {e}")
            print_colored(f"    ❌ 获取市场数据失败: {str(e)}", Colors.ERROR)
            return pd.DataFrame()

    def calculate_indicators_safe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """安全计算技术指标"""
        try:
            # 检查 DataFrame
            if df.empty:
                print_colored(f"    ⚠️ DataFrame 为空，跳过指标计算", Colors.WARNING)
                return df

            # 检查必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print_colored(f"    ⚠️ 缺少必要列: {missing_columns}", Colors.WARNING)
                return df

            # 尝试导入指标模块
            try:
                from indicators_module import calculate_optimized_indicators
                df = calculate_optimized_indicators(df)
                print_colored(f"    ✅ 技术指标计算完成", Colors.SUCCESS)
            except ImportError:
                print_colored(f"    ⚠️ 指标模块不可用，使用基础计算", Colors.WARNING)
                # 基础指标计算
                df = self.calculate_basic_indicators(df)
            except Exception as e:
                print_colored(f"    ❌ 计算优化指标失败: {e}", Colors.ERROR)
                # 降级到基础指标
                df = self.calculate_basic_indicators(df)

            return df

        except Exception as e:
            print_colored(f"    ❌ 指标计算失败: {e}", Colors.ERROR)
            return df

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        try:
            # RSI
            df['RSI'] = self.calculate_rsi(df['close'], 14)

            # 移动平均线
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

            # 布林带
            df['BB_Middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()

            # ADX（简化版）
            df['ADX'] = 25  # 默认值

            print_colored(f"    ✅ 基础指标计算完成", Colors.SUCCESS)
            return df

        except Exception as e:
            print_colored(f"    ❌ 基础指标计算失败: {e}", Colors.ERROR)
            # 添加默认值
            df['RSI'] = 50
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['BB_Middle'] = df['close'].mean() if 'close' in df else 0
            df['BB_Upper'] = df['BB_Middle'] * 1.02
            df['BB_Lower'] = df['BB_Middle'] * 0.98
            return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # 填充 NaN 值
        return rsi

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # 填充 NaN 值
        return rsi

    # 修复方案2：在 simple_trading_bot.py 中改进 _analyze_trend 方法

    def _analyze_trend(self, df):
        """分析趋势 - 修复版：即使没有DI指标也能判断趋势"""
        try:
            trend_info = {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'duration': 0
            }

            if len(df) < 20:
                return trend_info

            # 1. 使用ADX判断趋势强度（这部分没问题）
            adx_value = None
            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[-1]
                if not pd.isna(adx):
                    adx_value = float(adx)
                    if adx_value > 25:
                        trend_info['strength'] = min(adx_value / 50, 1.0)
                        print(f"  ADX趋势强度: {adx_value:.1f} -> 强度: {trend_info['strength']:.2f}")

            # 2. 判断趋势方向 - 使用多种方法
            direction_votes = {'UP': 0, 'DOWN': 0}

            # 方法1：使用DI（如果有）
            if 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
                plus_di = df['Plus_DI'].iloc[-1]
                minus_di = df['Minus_DI'].iloc[-1]
                if not pd.isna(plus_di) and not pd.isna(minus_di):
                    if plus_di > minus_di:
                        direction_votes['UP'] += 2
                        trend_info['confidence'] = abs(plus_di - minus_di) / (plus_di + minus_di)
                    else:
                        direction_votes['DOWN'] += 2
                        trend_info['confidence'] = abs(plus_di - minus_di) / (plus_di + minus_di)
                    print(f"  DI判断: +DI={plus_di:.1f}, -DI={minus_di:.1f}")

            # 方法2：使用均线判断（主要方法）
            if 'EMA20' in df.columns and 'EMA52' in df.columns:
                ema20 = df['EMA20'].iloc[-1]
                ema52 = df['EMA52'].iloc[-1]
                current_price = df['close'].iloc[-1]

                # 均线排列判断
                if current_price > ema20 > ema52:
                    direction_votes['UP'] += 3
                    print(f"  均线多头排列: 价格({current_price:.2f}) > EMA20({ema20:.2f}) > EMA52({ema52:.2f})")
                elif current_price < ema20 < ema52:
                    direction_votes['DOWN'] += 3
                    print(f"  均线空头排列: 价格({current_price:.2f}) < EMA20({ema20:.2f}) < EMA52({ema52:.2f})")

                # 短期均线斜率
                if len(df) >= 5:
                    ema20_slope = (ema20 - df['EMA20'].iloc[-5]) / df['EMA20'].iloc[-5] * 100
                    if ema20_slope > 0.5:
                        direction_votes['UP'] += 1
                        print(f"  EMA20上升斜率: {ema20_slope:.2f}%")
                    elif ema20_slope < -0.5:
                        direction_votes['DOWN'] += 1
                        print(f"  EMA20下降斜率: {ema20_slope:.2f}%")

            # 方法3：使用价格动量
            if len(df) >= 20:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
                if price_change > 2:
                    direction_votes['UP'] += 2
                    print(f"  20期价格上涨: {price_change:.2f}%")
                elif price_change < -2:
                    direction_votes['DOWN'] += 2
                    print(f"  20期价格下跌: {price_change:.2f}%")

            # 方法4：使用高低点判断
            if len(df) >= 10:
                recent_high = df['high'].iloc[-10:].max()
                recent_low = df['low'].iloc[-10:].min()
                older_high = df['high'].iloc[-20:-10].max()
                older_low = df['low'].iloc[-20:-10].min()

                if recent_high > older_high and recent_low > older_low:
                    direction_votes['UP'] += 1
                    print(f"  高低点上移")
                elif recent_high < older_high and recent_low < older_low:
                    direction_votes['DOWN'] += 1
                    print(f"  高低点下移")

            # 方法5：使用RSI趋势
            if 'RSI' in df.columns and len(df) >= 10:
                rsi = df['RSI'].iloc[-1]
                rsi_prev = df['RSI'].iloc[-10]
                if not pd.isna(rsi) and not pd.isna(rsi_prev):
                    if rsi > 50 and rsi > rsi_prev:
                        direction_votes['UP'] += 1
                    elif rsi < 50 and rsi < rsi_prev:
                        direction_votes['DOWN'] += 1

            # 综合判断趋势方向
            print(f"  趋势投票: UP={direction_votes['UP']}, DOWN={direction_votes['DOWN']}")

            if direction_votes['UP'] > direction_votes['DOWN'] and direction_votes['UP'] >= 3:
                trend_info['direction'] = 'UP'
                if trend_info['confidence'] == 0:
                    trend_info['confidence'] = direction_votes['UP'] / (direction_votes['UP'] + direction_votes['DOWN'])
            elif direction_votes['DOWN'] > direction_votes['UP'] and direction_votes['DOWN'] >= 3:
                trend_info['direction'] = 'DOWN'
                if trend_info['confidence'] == 0:
                    trend_info['confidence'] = direction_votes['DOWN'] / (
                                direction_votes['UP'] + direction_votes['DOWN'])
            else:
                trend_info['direction'] = 'NEUTRAL'

            # 如果ADX很高但方向不明确，使用价格位置判断
            if adx_value and adx_value > 40 and trend_info['direction'] == 'NEUTRAL':
                if 'EMA52' in df.columns:
                    if df['close'].iloc[-1] > df['EMA52'].iloc[-1]:
                        trend_info['direction'] = 'UP'
                        print(f"  ADX高但方向不明，使用价格>EMA52判断为上涨")
                    else:
                        trend_info['direction'] = 'DOWN'
                        print(f"  ADX高但方向不明，使用价格<EMA52判断为下跌")

            # 计算趋势持续时间
            if trend_info['direction'] != 'NEUTRAL':
                count = 0
                for i in range(len(df) - 1, max(0, len(df) - 50), -1):
                    if trend_info['direction'] == 'UP':
                        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                            count += 1
                        else:
                            break
                    else:
                        if df['close'].iloc[i] < df['close'].iloc[i - 1]:
                            count += 1
                        else:
                            break
                trend_info['duration'] = count

            print(
                f"  最终趋势判断: {trend_info['direction']} (强度: {trend_info['strength']:.2f}, 置信度: {trend_info['confidence']:.2f})")

            return trend_info

        except Exception as e:
            print(f"  ❌ 趋势分析错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'duration': 0
            }

    def _perform_technical_analysis(self, symbol: str, df: pd.DataFrame = None) -> Dict:
        """
        执行技术分析 - 修复版本

        参数:
            symbol: 交易对符号
            df: K线数据DataFrame（可选，如果没有提供会自动获取）

        返回:
            包含技术分析结果的字典
        """
        print_colored("📈 执行传统技术分析...", Colors.INFO)

        # 如果没有提供df，尝试获取
        if df is None:
            print_colored(f"    📊 正在获取 {symbol} 的K线数据...", Colors.INFO)
            df = self.get_market_data_sync(symbol)

            if df is None or df.empty:
                print_colored("❌ 无法获取有效数据", Colors.ERROR)
                return self.get_default_technical_analysis(symbol)

            print_colored(f"    ✅ 获取到 {len(df)} 条数据", Colors.SUCCESS)

            # 计算指标（如果还没计算）
            if 'RSI' not in df.columns:
                df = calculate_optimized_indicators(df)

        try:
            latest = df.iloc[-1]

            # 获取各项指标
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_signal', 0)
            macd_histogram = latest.get('MACD_histogram', 0)
            adx = latest.get('ADX', 25)

            # 修复：正确获取布林带位置
            if 'bb_position' in df.columns:
                bb_position = df['bb_position'].iloc[-1] * 100  # 转换为百分比
            else:
                # 如果没有预计算的bb_position，手动计算
                close = latest['close']
                bb_upper = latest.get('bb_upper', close)
                bb_lower = latest.get('bb_lower', close)
                if bb_upper != bb_lower:
                    bb_position = ((close - bb_lower) / (bb_upper - bb_lower)) * 100
                else:
                    bb_position = 50.0

            # 计算成交量比率
            volume_ratio = 1.0
            if 'volume' in df.columns and len(df) >= 20:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume

            # 判断趋势
            trend_direction = 'NEUTRAL'
            trend_strength = 0

            if 'EMA5' in df.columns and 'EMA20' in df.columns:
                ema5 = latest['EMA5']
                ema20 = latest['EMA20']
                ema_ratio = ema5 / ema20 if ema20 > 0 else 1

                if ema_ratio > 1.01:  # EMA5 > EMA20 * 1.01
                    trend_direction = 'UP'
                    trend_strength = min(1.0, (ema_ratio - 1) * 100)
                elif ema_ratio < 0.99:  # EMA5 < EMA20 * 0.99
                    trend_direction = 'DOWN'
                    trend_strength = min(1.0, (1 - ema_ratio) * 100)

            # 获取其他有用的指标
            williams_r = latest.get('Williams_R', -50)
            cci = latest.get('CCI', 0)
            momentum = latest.get('Momentum', 0)
            atr = latest.get('ATR', 0)

            # 分析MACD信号
            macd_signal_type = 'NEUTRAL'
            if macd > macd_signal and macd_histogram > 0:
                macd_signal_type = 'BULLISH'
            elif macd < macd_signal and macd_histogram < 0:
                macd_signal_type = 'BEARISH'

            # 获取价格信息
            current_price = latest.get('close', 0)
            high_24h = df['high'].tail(96).max() if len(df) > 96 else latest.get('high', current_price)
            low_24h = df['low'].tail(96).min() if len(df) > 96 else latest.get('low', current_price)
            price_change_24h = ((current_price - df['close'].iloc[-96]) / df['close'].iloc[-96] * 100) if len(
                df) > 96 else 0

            # 计算支撑阻力（简化版）
            recent_highs = df['high'].tail(20).nlargest(3).mean()
            recent_lows = df['low'].tail(20).nsmallest(3).mean()

            # 打印分析结果
            print_colored(f"📊 {symbol} 技术分析完成:", Colors.INFO)
            print_colored(f"  ✓ RSI: {rsi:.1f}", Colors.INFO)
            print_colored(f"  ✓ ADX: {adx:.1f}", Colors.INFO)
            print_colored(f"  ✓ 布林带位置: {bb_position:.1f}%", Colors.INFO)
            print_colored(f"  ✓ 成交量比率: {volume_ratio:.2f}x", Colors.INFO)
            print_colored(f"  ✓ 趋势: {trend_direction} (强度: {trend_strength:.2f})", Colors.INFO)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'macd_signal_type': macd_signal_type,
                'adx': adx,
                'bb_position': bb_position,  # 现在是实际值而不是固定的50.0
                'williams_r': williams_r,
                'cci': cci,
                'momentum': momentum,
                'atr': atr,
                'volume': {
                    'ratio': volume_ratio,
                    'trend': 'INCREASING' if volume_ratio > 1.3 else 'DECREASING' if volume_ratio < 0.7 else 'NEUTRAL'
                },
                'trend': {
                    'direction': trend_direction,
                    'strength': trend_strength
                },
                'price_info': {
                    'current': current_price,
                    'high_24h': high_24h,
                    'low_24h': low_24h,
                    'change_24h': price_change_24h
                },
                'support_resistance': {
                    'resistance': recent_highs,
                    'support': recent_lows
                },
                'df': df  # 添加df到返回值中，其他地方可能需要
            }

        except Exception as e:
            print_colored(f"❌ 技术分析错误: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()

            # 返回默认值
            return self.get_default_technical_analysis(symbol)

    def _track_spot_whale_flow(self, depth_data: Dict) -> Dict:
        """追踪现货大单流向"""
        # 这是一个简化版本，实际应该追踪大额订单
        bids = depth_data.get('bids', [])
        asks = depth_data.get('asks', [])

        # 定义大单阈值（这里简化处理）
        large_order_threshold = 10  # 根据具体交易对调整

        large_bids = [bid for bid in bids if float(bid[1]) > large_order_threshold]
        large_asks = [ask for ask in asks if float(ask[1]) > large_order_threshold]

        whale_activity = "无显著活动"
        if len(large_bids) > len(large_asks) * 1.5:
            whale_activity = "大户买入"
        elif len(large_asks) > len(large_bids) * 1.5:
            whale_activity = "大户卖出"
        elif len(large_bids) > 0 or len(large_asks) > 0:
            whale_activity = "大户活跃"

        print_colored("  🐋 追踪现货大单流向...", Colors.INFO)
        print_colored(f"    🐋 现货大单: {whale_activity}", Colors.INFO)

        return {
            'whale_activity': whale_activity,
            'large_bid_count': len(large_bids),
            'large_ask_count': len(large_asks)
        }

    def get_default_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """返回默认的技术分析结果"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'bb_position': 50,
            'signal_strength': 0,
            'trend': {
                'direction': 'NEUTRAL',
                'strength': 0
            },
            'volume': {
                'trend': 'NEUTRAL',
                'ratio': 1.0
            },
            'momentum': {
                'macd_signal': 'NEUTRAL'
            },
            'patterns': [],
            'error': 'Failed to get data'
        }

    def _calculate_simple_signal_strength(self, technical: Dict) -> float:
        """计算简单的信号强度（用于降级模式）"""
        signal_strength = 0

        # RSI
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            signal_strength += 2
        elif rsi > 70:
            signal_strength -= 2

        # MACD
        if technical.get('macd', 0) > technical.get('macd_signal', 0):
            signal_strength += 1.5
        else:
            signal_strength -= 1.5

        # 布林带
        bb_pos = technical.get('bb_position', 50)
        if bb_pos < 20:
            signal_strength += 1.5
        elif bb_pos > 80:
            signal_strength -= 1.5

        return signal_strength

    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """判断趋势方向"""
        if len(df) < 50:
            return 'NEUTRAL'

        # 使用EMA判断
        if 'EMA20' in df and 'EMA50' in df:
            if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
                return 'UP'
            elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]:
                return 'DOWN'

        # 使用价格判断
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        older_high = df['high'].iloc[-50:-20].max()
        older_low = df['low'].iloc[-50:-20].min()

        if recent_high > older_high and recent_low > older_low:
            return 'UP'
        elif recent_high < older_high and recent_low < older_low:
            return 'DOWN'

        return 'NEUTRAL'

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度（0-1）"""
        if 'ADX' in df:
            adx = df['ADX'].iloc[-1]
            return min(adx / 50, 1.0)  # ADX 50以上为最强

        # 使用价格斜率
        prices = df['close'].tail(20).values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # 归一化斜率
        price_std = df['close'].std()
        if price_std > 0:
            normalized_slope = abs(slope) / price_std
            return min(normalized_slope * 10, 1.0)

        return 0.5

    def _calculate_trend_duration(self, df: pd.DataFrame) -> int:
        """计算趋势持续周期数"""
        direction = self._determine_trend_direction(df)
        if direction == 'NEUTRAL':
            return 0

        count = 0
        for i in range(len(df) - 1, 0, -1):
            if direction == 'UP':
                if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                    count += 1
                else:
                    break
            else:  # DOWN
                if df['close'].iloc[i] < df['close'].iloc[i - 1]:
                    count += 1
                else:
                    break

        return count

    def _calculate_trend_quality(self, df: pd.DataFrame) -> float:
        """计算趋势质量（0-1）"""
        # 使用价格的R²值
        prices = df['close'].tail(20).values
        x = np.arange(len(prices))

        # 线性回归
        slope, intercept = np.polyfit(x, prices, 1)
        predicted = slope * x + intercept

        # 计算R²
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)

        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            return max(0, r_squared)

        return 0.5

    def _calculate_bb_position(self, df: pd.DataFrame) -> float:
        """计算价格在布林带中的位置（0-100）"""
        if all(col in df for col in ['BB_Upper', 'BB_Lower', 'close']):
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            close = df['close'].iloc[-1]

            if upper > lower:
                return ((close - lower) / (upper - lower)) * 100

        return 50

    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """计算布林带宽度"""
        if all(col in df for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            middle = df['BB_Middle'].iloc[-1]

            if middle > 0:
                return (upper - lower) / middle

        return 0.02

    def _get_mtf_signals(self, symbol: str) -> Dict:
        """获取多时间框架信号"""
        if not hasattr(self, 'mtf_coordinator') or not self.mtf_coordinator:
            return {}

        try:
            # 这里调用您现有的多时间框架分析
            return self.mtf_coordinator.get_all_timeframe_signals(symbol)
        except:
            return {}

    def _analyze_volume_debug(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析成交量模式 - 增强调试版本"""
        volume_analysis = {}

        print_colored(f"\n    🔍 成交量分析调试信息:", Colors.BLUE)

        if 'volume' not in df.columns:
            print_colored(f"    ❌ 数据中没有volume列!", Colors.ERROR)
            return volume_analysis

        try:
            # 获取基础数据
            current_vol = df['volume'].iloc[-1]

            # 计算不同周期的均量
            vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
            vol_ma10 = df['volume'].rolling(10).mean().iloc[-1]
            vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]

            # 输出最近10个成交量数据
            recent_volumes = df['volume'].tail(10).tolist()
            print_colored(f"    📊 最近10个成交量: {[f'{v:.0f}' for v in recent_volumes]}", Colors.INFO)

            # 输出各种均量
            print_colored(f"    📊 当前成交量: {current_vol:,.0f}", Colors.INFO)
            print_colored(f"    📊 5日均量: {vol_ma5:,.0f}", Colors.INFO)
            print_colored(f"    📊 10日均量: {vol_ma10:,.0f}", Colors.INFO)
            print_colored(f"    📊 20日均量: {vol_ma20:,.0f}", Colors.INFO)

            # 计算各种比率
            ratio_current_to_ma20 = current_vol / vol_ma20 if vol_ma20 > 0 else 0
            ratio_ma5_to_ma20 = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 0

            print_colored(f"    📊 当前/20日均量比: {ratio_current_to_ma20:.2f}x", Colors.INFO)
            print_colored(f"    📊 5日/20日均量比: {ratio_ma5_to_ma20:.2f}x", Colors.INFO)

            # 成交量趋势判断（多种方法）
            # 方法1：5日均量 vs 20日均量
            trend_5vs20 = vol_ma5 > vol_ma20
            print_colored(f"    📊 趋势判断1 (5MA>20MA): {'上升' if trend_5vs20 else '下降'}", Colors.INFO)

            # 方法2：最近5天平均 vs 前5-10天平均
            if len(df) >= 10:
                recent_5_avg = df['volume'].iloc[-5:].mean()
                previous_5_avg = df['volume'].iloc[-10:-5].mean()
                trend_recent = recent_5_avg > previous_5_avg
                print_colored(f"    📊 趋势判断2 (近5天vs前5天): {'上升' if trend_recent else '下降'} "
                              f"({recent_5_avg:.0f} vs {previous_5_avg:.0f})", Colors.INFO)

            # 方法3：成交量斜率
            if len(df) >= 5:
                vol_slope = np.polyfit(range(5), df['volume'].iloc[-5:].values, 1)[0]
                print_colored(f"    📊 成交量斜率: {vol_slope:.2f} {'(上升)' if vol_slope > 0 else '(下降)'}",
                              Colors.INFO)

            # 检测成交量异常
            vol_std = df['volume'].rolling(20).std().iloc[-1]
            vol_zscore = (current_vol - vol_ma20) / vol_std if vol_std > 0 else 0
            print_colored(f"    📊 成交量Z分数: {vol_zscore:.2f} "
                          f"({'异常放大' if vol_zscore > 2 else '异常缩小' if vol_zscore < -2 else '正常'})",
                          Colors.INFO)

            # 构建返回数据
            volume_analysis['current'] = current_vol
            volume_analysis['average'] = vol_ma20
            volume_analysis['ratio'] = ratio_current_to_ma20

            # 综合判断趋势
            if ratio_ma5_to_ma20 > 1.2:  # 5日均量比20日均量高20%以上
                volume_analysis['trend'] = 'INCREASING'
                volume_analysis['trend_strength'] = 'STRONG'
                print_colored(f"    ✅ 成交量趋势: 强势上升", Colors.GREEN)
            elif ratio_ma5_to_ma20 > 1.0:
                volume_analysis['trend'] = 'INCREASING'
                volume_analysis['trend_strength'] = 'MODERATE'
                print_colored(f"    ✅ 成交量趋势: 温和上升", Colors.GREEN)
            elif ratio_ma5_to_ma20 > 0.8:
                volume_analysis['trend'] = 'NEUTRAL'
                volume_analysis['trend_strength'] = 'NEUTRAL'
                print_colored(f"    ➖ 成交量趋势: 横盘整理", Colors.YELLOW)
            else:
                volume_analysis['trend'] = 'DECREASING'
                volume_analysis['trend_strength'] = 'WEAK'
                print_colored(f"    ❌ 成交量趋势: 萎缩", Colors.RED)

            # 添加额外的调试信息
            volume_analysis['debug_info'] = {
                'vol_ma5': vol_ma5,
                'vol_ma10': vol_ma10,
                'vol_ma20': vol_ma20,
                'ratio_5vs20': ratio_ma5_to_ma20,
                'vol_zscore': vol_zscore,
                'recent_volumes': recent_volumes[-5:]  # 最近5个成交量
            }

        except Exception as e:
            print_colored(f"    ❌ 成交量分析错误: {e}", Colors.ERROR)
            import traceback
            traceback.print_exc()

        return volume_analysis

    def _display_volume_analysis(self, volume_analysis: Dict):
        """显示成交量分析结果"""
        if not volume_analysis:
            print_colored(f"    • 成交量数据缺失", Colors.GRAY)
            return

        trend = volume_analysis.get('trend', 'UNKNOWN')
        strength = volume_analysis.get('trend_strength', '')
        ratio = volume_analysis.get('ratio', 0)

        # 根据趋势选择颜色
        if trend == 'INCREASING':
            color = Colors.GREEN
            icon = "📈"
        elif trend == 'DECREASING':
            color = Colors.RED
            icon = "📉"
        else:
            color = Colors.YELLOW
            icon = "➖"

        # 显示成交量状态
        print_colored(f"    • {icon} 成交量{trend.lower()} ({strength.lower()}) - 比率: {ratio:.2f}x", color)

        # 如果有调试信息，显示更多细节
        if 'debug_info' in volume_analysis:
            debug = volume_analysis['debug_info']
            print_colored(f"       5MA/20MA: {debug['ratio_5vs20']:.2f}, Z-score: {debug['vol_zscore']:.2f}",
                          Colors.INFO)

    def identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        识别价格形态
        """
        patterns = []

        try:
            # 获取最近的高低点
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]

            # 检查突破
            if current_price > recent_high * 0.995:
                patterns.append("突破近期高点")
            elif current_price < recent_low * 1.005:
                patterns.append("突破近期低点")

            # 检查V型反转
            mid_point = len(df) // 2
            first_half_trend = df['close'].iloc[:mid_point].mean()
            second_half_trend = df['close'].iloc[mid_point:].mean()

            if second_half_trend > first_half_trend * 1.02:
                patterns.append("V型反转（上涨）")
            elif second_half_trend < first_half_trend * 0.98:
                patterns.append("V型反转（下跌）")

            # 检查支撑/阻力
            if abs(current_price - recent_high) / recent_high < 0.01:
                patterns.append(f"接近阻力位 ${recent_high:.2f}")
            elif abs(current_price - recent_low) / recent_low < 0.01:
                patterns.append(f"接近支撑位 ${recent_low:.2f}")

        except Exception as e:
            self.logger.error(f"形态识别错误: {e}")

        return pattern

    def _integrate_analyses_v2(self, game_theory: Dict, technical: Dict, symbol: str) -> Dict[str, Any]:

        """
            整合博弈论和技术分析结果 - 优化版本
            使用趋势感知指标和信号稳定系统
            """
        integrated = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'final_score': 0.0,
            'signals': [],
            'risks': [],
            'reason': '',
            'strategy_type': 'UNKNOWN'
        }

        # 检查是否有错误
        if 'error' in game_theory or 'error' in technical:
            integrated['reason'] = '分析数据不完整'
            return integrated

        # 使用趋势感知RSI
        if hasattr(self, 'trend_aware_rsi') and self.trend_aware_rsi and 'df' in technical:
            df_with_rsi = self.trend_aware_rsi.calculate_rsi_score(technical['df'])
            rsi_score = df_with_rsi['rsi_score'].iloc[-1]
        else:
            # 降级到原始RSI逻辑
            rsi = technical.get('rsi', 50)
            rsi_score = (50 - rsi) if rsi < 50 else (70 - rsi) / 20 * 100

        # 获取市场状态和动态权重
        if hasattr(self, 'weight_manager') and self.weight_manager and 'df' in technical:
            market_regime = self.weight_manager.detect_market_regime(technical['df'])
            weights = self.weight_manager.calculate_adaptive_weights(market_regime)
        else:
            weights = {
                'RSI': 0.2,
                'MACD': 0.3,
                'CCI': 0.15,
                'Williams_R': 0.15,
                'EMA': 0.2
            }

        # 计算各指标信号
        indicator_signals = {
            'RSI': rsi_score / 100,
            'MACD': np.clip(technical.get('macd', 0) / 0.001, -1, 1),
            'CCI': np.clip(technical.get('cci', 0) / 100, -1, 1),
            'Williams_R': (technical.get('williams_r', -50) + 50) / 50,
            'EMA': 1 if technical.get('ema_signal', 0) > 0 else -1
        }

        # 应用权重计算技术分析得分
        tech_score = sum(indicator_signals[ind] * weights.get(ind, 0) for ind in indicator_signals) * 100

        # 博弈论得分（保持原有逻辑）
        whale_intent = game_theory.get('whale_intent', 'NEUTRAL')
        whale_confidence = game_theory.get('confidence', 0)

        game_score = 0
        if whale_intent == 'ACCUMULATION':
            game_score = 80 * whale_confidence
            integrated['signals'].append("庄家吸筹")
        if whale_intent == 'DISTRIBUTION':
            # 在上涨趋势中，派发信号权重降低
            trend_direction = technical.get('trend', {}).get('direction', 'NEUTRAL')
            trend_strength = technical.get('trend', {}).get('strength', 0)

            if trend_direction == 'UP' and trend_strength > 0.5:
                game_score = -40 * whale_confidence  # 从-80降到-40
                integrated['risks'].append("上涨趋势中的获利回吐")
            else:
                game_score = -80 * whale_confidence
                integrated['signals'].append("庄家派发")
        # 综合得分（调整权重：技术60%，博弈40%）
        combined_score = tech_score * 0.6 + game_score * 0.4

        # 使用信号稳定器
        if hasattr(self, 'signal_stabilizer') and self.signal_stabilizer:
            final_position, confirmed_signal = self.signal_stabilizer.confirm_signal(combined_score)

            if final_position == 1:
                integrated['action'] = 'BUY'
                integrated['confidence'] = min(abs(confirmed_signal) / 100, 0.9)
            elif final_position == -1:
                integrated['action'] = 'SELL'
                integrated['confidence'] = min(abs(confirmed_signal) / 100, 0.9)
            else:
                integrated['action'] = 'HOLD'
                integrated['reason'] = "信号未确认或在持仓保护期"
        else:
            # 降级到原始逻辑
            if combined_score > 30:
                integrated['action'] = 'BUY'
                integrated['confidence'] = min(combined_score / 100, 0.9)
            elif combined_score < -30:
                integrated['action'] = 'SELL'
                integrated['confidence'] = min(abs(combined_score) / 100, 0.9)

        integrated['final_score'] = abs(combined_score / 10)
        integrated['game_theory_analysis'] = game_theory
        integrated['technical_analysis'] = technical

        # 详细输出
        print_colored(f"\n🔗 优化后的信号整合:", Colors.CYAN + Colors.BOLD)
        print_colored(f"    • 技术得分: {tech_score:.1f} (权重: 60%)", Colors.INFO)
        print_colored(f"    • 博弈得分: {game_score:.1f} (权重: 40%)", Colors.INFO)
        print_colored(f"    • 综合得分: {combined_score:.1f}", Colors.CYAN)
        print_colored(f"    • 最终决策: {integrated['action']} (置信度: {integrated['confidence']:.1%})",
                      Colors.GREEN if integrated['action'] == 'BUY' else Colors.RED)

        return integrated

    def _integrate_analyses(self, game_theory: Dict, technical: Dict, symbol: str) -> Dict[str, Any]:
        """
        整合博弈论和技术分析结果
        """
        integrated = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'final_score': 0.0,
            'signals': [],
            'risks': [],
            'reason': ''
        }

        # 检查是否有错误
        if 'error' in game_theory or 'error' in technical:
            integrated['reason'] = '分析数据不完整'
            return integrated

        try:
            # 提取关键信息，确保是数字类型
            whale_intent = game_theory.get('whale_intent', 'NEUTRAL')
            whale_confidence = float(game_theory.get('confidence', 0))
            tech_strength = float(technical.get('signal_strength', 0))

            # 博弈论权重
            game_weight = 0.6  # 博弈论占60%权重
            tech_weight = 0.4  # 技术分析占40%权重

            # 计算博弈论得分
            game_score = 0
            if whale_intent == 'ACCUMULATION':
                game_score = 8 * whale_confidence
                integrated['signals'].append("庄家吸筹")
            elif whale_intent == 'DISTRIBUTION':
                game_score = -8 * whale_confidence
                integrated['signals'].append("庄家派发")
            elif whale_intent == 'MANIPULATION_UP':
                game_score = 5 * whale_confidence
                integrated['signals'].append("疑似拉升")
                integrated['risks'].append("可能是诱多")
            elif whale_intent == 'MANIPULATION_DOWN':
                game_score = -5 * whale_confidence
                integrated['signals'].append("疑似打压")
                integrated['risks'].append("可能是诱空")

            # 计算技术分析得分
            tech_score = min(max(tech_strength * 2, -10), 10)  # 限制在-10到10之间

            # 综合得分
            combined_score = (game_score * game_weight + tech_score * tech_weight)

            # 一致性检查
            if game_score * tech_score > 0:  # 同向
                combined_score *= 1.2  # 信号一致，增加20%权重
                integrated['signals'].append("博弈与技术共振")
            elif game_score * tech_score < 0:  # 反向
                combined_score *= 0.6  # 信号矛盾，降低40%权重
                integrated['risks'].append("信号存在分歧")

            # 决定交易方向
            if combined_score > 3:
                integrated['action'] = 'BUY'
                integrated['confidence'] = min(combined_score / 10, 0.9)
            elif combined_score < -3:
                integrated['action'] = 'SELL'
                integrated['confidence'] = min(abs(combined_score) / 10, 0.9)
            else:
                integrated['action'] = 'HOLD'
                integrated['reason'] = f"综合评分不足 ({combined_score:.1f})"

            # 最终评分（0-10）
            integrated['final_score'] = abs(combined_score)
            integrated['game_theory_analysis'] = game_theory
            integrated['technical_analysis'] = technical

            # 添加详细说明
            print_colored(f"    • 博弈论得分: {game_score:.1f} (权重: {game_weight:.0%})", Colors.INFO)
            print_colored(f"    • 技术分析得分: {tech_score:.1f} (权重: {tech_weight:.0%})", Colors.INFO)
            print_colored(f"    • 综合得分: {combined_score:.1f}", Colors.CYAN)

        except Exception as e:
            self.logger.error(f"整合分析错误: {e}")
            print_colored(f"❌ 整合分析错误: {str(e)}", Colors.ERROR)
            integrated['reason'] = f'整合分析失败: {str(e)}'

        return integrated

    def get_historical_data(self, symbol: str, interval: str = '5m', limit: int = 500) -> pd.DataFrame:
        """获取历史K线数据（异步版本）"""
        try:
            # 使用现有的 data_module.get_historical_data
            from data_module import get_historical_data

            # 转换为同步调用（如果您的系统需要异步，可以用 asyncio.to_thread）
            df = get_historical_data(self.client, symbol, interval, limit)

            return df

        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            raise

    def _integrate_analyses_trend_first(self, game_theory: Dict, technical: Dict, symbol: str) -> Dict[str, Any]:
        """整合分析 - 修复版本（包含动态阈值）"""
        integrated = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'final_score': 0.0,
            'signals': [],
            'risks': [],
            'reason': '',
            'strategy_type': 'TREND_TECH_GAME'
        }

        # 检查数据完整性
        if 'error' in game_theory or 'error' in technical:
            integrated['reason'] = '分析数据不完整'
            return integrated

        try:
            # 1. 获取技术分析得分（使用修复后的函数）
            tech_signal_strength = self.calculate_technical_score_v2(technical)

            # 2. 获取趋势数据
            trend = technical.get('trend', {})
            trend_direction = trend.get('direction', 'NEUTRAL')
            trend_strength = float(trend.get('strength', 0))
            adx = technical.get('adx', 20)  # 获取ADX值

            # 3. 获取博弈论数据
            whale_intent = game_theory.get('whale_intent', 'NEUTRAL')
            whale_confidence = float(game_theory.get('confidence', 0))

            # 4. 动态权重分配（基于市场环境）
            if adx > 30:  # 趋势市场
                weights = {
                    'trend': 0.45,
                    'technical': 0.30,
                    'game_theory': 0.25
                }
                print_colored(f"    • 市场环境: 趋势市场 (ADX: {adx:.1f})", Colors.INFO)
            elif technical.get('atr_ratio', 0.01) > 0.02:  # 波动市场
                weights = {
                    'trend': 0.25,
                    'technical': 0.35,
                    'game_theory': 0.40
                }
                print_colored(f"    • 市场环境: 波动市场", Colors.INFO)
            else:  # 震荡市场
                weights = {
                    'trend': 0.20,
                    'technical': 0.40,
                    'game_theory': 0.40
                }
                print_colored(f"    • 市场环境: 震荡市场", Colors.INFO)

            # 5. 计算趋势得分
            trend_score = 0
            if trend_direction == 'UP':
                trend_score = 5 * trend_strength
                integrated['signals'].append(f"上升趋势(强度:{trend_strength:.1%})")
            elif trend_direction == 'DOWN':
                trend_score = -5 * trend_strength
                integrated['signals'].append(f"下降趋势(强度:{trend_strength:.1%})")

            # 6. 计算博弈论得分（考虑趋势背景）
            game_score = 0
            if whale_intent == 'ACCUMULATION':
                game_score = 6 * whale_confidence
                integrated['signals'].append(f"庄家吸筹({whale_confidence:.0%})")
            elif whale_intent == 'DISTRIBUTION':
                # 在上涨趋势中，派发信号权重降低
                if trend_direction == 'UP' and trend_strength > 0.5:
                    game_score = -3 * whale_confidence  # 从-6降到-3
                    integrated['risks'].append("上涨趋势中的获利回吐")
                else:
                    game_score = -6 * whale_confidence
                    integrated['signals'].append(f"庄家派发({whale_confidence:.0%})")
            elif whale_intent == 'MANIPULATION_UP':
                game_score = 4 * whale_confidence
                integrated['signals'].append(f"疑似拉升({whale_confidence:.0%})")
            elif whale_intent == 'MANIPULATION_DOWN':
                game_score = -4 * whale_confidence
                integrated['signals'].append(f"疑似打压({whale_confidence:.0%})")

            # 7. 计算综合得分
            final_score = (
                    trend_score * weights['trend'] +
                    tech_signal_strength * weights['technical'] +
                    game_score * weights['game_theory']
            )

            # 8. 一致性检查和加成
            if tech_signal_strength * game_score > 0:  # 同向
                if abs(tech_signal_strength) > 2 and abs(game_score) > 2:
                    final_score *= 1.3  # 强信号共振
                    integrated['signals'].append("强信号共振")
                else:
                    final_score *= 1.15
                    integrated['signals'].append("信号共振")
            elif tech_signal_strength * game_score < -1:  # 明显反向
                final_score *= 0.7
                integrated['risks'].append("信号分歧较大")

            # 9. 趋势确认加成
            if trend_score * final_score > 0 and abs(trend_score) > 2:
                final_score *= 1.1
                integrated['signals'].append("趋势确认")

            # ============ 关键修复：定义动态阈值 ============
            # 先设置默认阈值
            buy_threshold = 1.2
            strong_buy_threshold = 2.5
            sell_threshold = -1.2
            strong_sell_threshold = -2.5

            # 根据趋势动态调整阈值
            if trend_direction == 'UP' and adx > 25:
                # 上涨趋势中，做多阈值降低，做空阈值提高
                buy_threshold = 1.0
                strong_buy_threshold = 2.0
                sell_threshold = -3.0
                strong_sell_threshold = -5.0
                print_colored(f"    • 上涨趋势阈值调整: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)
            elif trend_direction == 'DOWN' and adx > 25:
                # 下跌趋势中，做空阈值降低，做多阈值提高
                buy_threshold = 3.0
                strong_buy_threshold = 5.0
                sell_threshold = -1.0
                strong_sell_threshold = -2.0
                print_colored(f"    • 下跌趋势阈值调整: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)
            else:
                # 震荡市场使用默认阈值
                print_colored(f"    • 震荡市场标准阈值: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)

            # 10. 应用阈值进行决策
            if final_score >= buy_threshold:
                integrated['action'] = 'BUY'
                integrated['confidence'] = min(abs(final_score) / 8, 0.9)
                if final_score >= strong_buy_threshold:
                    integrated['action'] = 'STRONG_BUY'
                    integrated['confidence'] = min(abs(final_score) / 6, 0.95)
            elif final_score <= sell_threshold:
                integrated['action'] = 'SELL'
                integrated['confidence'] = min(abs(final_score) / 8, 0.9)
                if final_score <= strong_sell_threshold:
                    integrated['action'] = 'STRONG_SELL'
                    integrated['confidence'] = min(abs(final_score) / 6, 0.95)
            else:
                integrated['action'] = 'HOLD'
                integrated[
                    'reason'] = f"信号不够强烈 (得分: {final_score:.1f}, 阈值: 买>{buy_threshold:.1f}, 卖<{sell_threshold:.1f})"

            integrated['final_score'] = abs(final_score)

            # 11. 输出详细评分
            print_colored(f"\n    📊 综合评分系统", Colors.CYAN + Colors.BOLD)
            print_colored(f"    • 市场环境: {'趋势' if adx > 30 else '震荡'} (ADX: {adx:.1f})", Colors.INFO)
            print_colored(f"    • 趋势: {trend_direction} (强度: {trend_strength:.2f})", Colors.INFO)
            print_colored(f"    • 趋势得分: {trend_score:.1f} (权重: {weights['trend']:.0%})", Colors.INFO)
            print_colored(f"    • 技术得分: {tech_signal_strength:.1f} (权重: {weights['technical']:.0%})", Colors.INFO)
            print_colored(f"    • 博弈得分: {game_score:.1f} (权重: {weights['game_theory']:.0%})", Colors.INFO)
            print_colored(f"    • 最终得分: {final_score:.2f}", Colors.YELLOW)
            print_colored(f"    • 决策阈值: 买入>{buy_threshold:.1f}, 卖出<{sell_threshold:.1f}", Colors.INFO)

            if integrated['action'] != 'HOLD':
                action_color = Colors.GREEN if 'BUY' in integrated['action'] else Colors.RED
                print_colored(f"    • 📍 交易决策: {integrated['action']} (置信度: {integrated['confidence']:.1%})",
                              action_color + Colors.BOLD)

        except Exception as e:
            self.logger.error(f"整合分析错误: {e}")
            print_colored(f"❌ 整合分析错误: {str(e)}", Colors.ERROR)
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            integrated['reason'] = f'整合分析失败: {str(e)}'

        return integrated

    def calculate_technical_score_v2(self, technical_analysis: Dict) -> float:
        """
        计算技术分析综合得分 V2 - 考虑趋势背景
        返回范围：-10 到 10
        """
        score = 0.0

        # 获取趋势信息
        trend_info = technical_analysis.get('trend', {})
        trend_direction = trend_info.get('direction', 'NEUTRAL')
        trend_strength = trend_info.get('strength', 0)
        adx = technical_analysis.get('adx', 20)

        # 1. RSI得分（权重：3分） - 根据趋势调整
        rsi = technical_analysis.get('rsi', 50)

        if trend_direction == 'UP' and adx > 25:
            # 上涨趋势中，做多阈值降低，做空阈值提高
            buy_threshold = 1.0  # 从1.2降到1.0
            strong_buy_threshold = 2.0  # 从2.5降到2.0
            sell_threshold = -3.0  # 从-1.2降到-3.0
            strong_sell_threshold = -5.0  # 从-2.5降到-5.0

            print_colored(f"    • 上涨趋势阈值调整: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)
        elif trend_direction == 'DOWN' and adx > 25:
            # 下跌趋势中，做空阈值降低，做多阈值提高
            buy_threshold = 3.0  # 从1.2提到3.0
            strong_buy_threshold = 5.0  # 从2.5提到5.0
            sell_threshold = -1.0  # 从-1.2提到-1.0
            strong_sell_threshold = -2.0  # 从-2.5提到-2.0
            print_colored(f"    • 下跌趋势阈值调整: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)
        else:
            # 震荡市场使用标准阈值
            buy_threshold = 1.5
            strong_buy_threshold = 3.0
            sell_threshold = -1.5
            strong_sell_threshold = -3.0
            print_colored(f"    • 震荡市场标准阈值: 买入>{buy_threshold}, 卖出<{sell_threshold}", Colors.INFO)

        # 2. MACD得分（权重：2分）
        momentum = technical_analysis.get('momentum', {})
        if isinstance(momentum, dict):
            macd_signal = momentum.get('macd_signal', 'NEUTRAL')
            if macd_signal == 'BULLISH':
                score += 2.0
            elif macd_signal == 'BEARISH':
                score -= 2.0

        # 3. 布林带位置（权重：2分） - 考虑趋势
        bb_position = technical_analysis.get('bb_position', 50)

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

        # 4. 威廉指标（权重：1.5分） - 考虑趋势
        williams_r = technical_analysis.get('williams_r', -50)

        if trend_direction == 'UP':
            if williams_r < -80:
                score += 1.5  # 超卖是机会
            elif williams_r > -20:
                score -= 0.5  # 超买只是轻微警告
        elif trend_direction == 'DOWN':
            if williams_r > -20:
                score -= 1.5  # 超买是做空机会
            elif williams_r < -80:
                score += 0.5  # 超卖只是轻微机会
        else:
            if williams_r < -80:
                score += 1.5
            elif williams_r > -20:
                score -= 1.5

        # 5. CCI指标（权重：1.5分） - 考虑趋势
        cci = technical_analysis.get('cci', 0)

        if trend_direction == 'UP':
            if cci < -100:
                score += 1.5  # 超卖是机会
            elif cci > 150:
                score -= 0.5  # 极度超买才警告
        elif trend_direction == 'DOWN':
            if cci > 100:
                score -= 1.5  # 超买是做空机会
            elif cci < -150:
                score += 0.5  # 极度超卖才考虑
        else:
            if cci < -100:
                score += 1.5
            elif cci > 100:
                score -= 1.5

        # 6. 成交量确认（额外加成）
        volume_ratio = technical_analysis.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score *= 1.2  # 放量确认
        elif volume_ratio < 0.5:
            score *= 0.8  # 缩量减弱信号

        return np.clip(score, -10, 10)

    def _calculate_risk_adjusted_params_v2(self, decision: Dict, account_balance: float, symbol: str) -> Dict[str, Any]:
        """
        计算风险调整后的交易参数 - 修复版本V2
        确保返回 entry_price 键
        """
        # 调用之前的修复版本
        params = self._calculate_risk_adjusted_params(decision, account_balance, symbol)

        if params and 'price' in params and 'entry_price' not in params:
            # 添加 entry_price 键
            params['entry_price'] = params['price']

        return params

    # 完整修复方案 - 修改 _calculate_risk_adjusted_params 函数的返回值

    def _calculate_risk_adjusted_params(self, decision: Dict, account_balance: float, symbol: str) -> Dict[str, Any]:
        """
        计算风险调整后的交易参数 - 完整修复版本
        """
        try:
            # 获取当前价格 - 修复：从多个可能的位置获取
            current_price = None

            # 尝试从不同的位置获取当前价格
            if 'technical_analysis' in decision and 'current_price' in decision['technical_analysis']:
                current_price = decision['technical_analysis']['current_price']
            elif 'current_price' in decision:
                current_price = decision['current_price']
            else:
                # 如果都没有，尝试获取最新价格
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    print_colored(f"    📊 获取最新价格: ${current_price:.4f}", Colors.INFO)
                except Exception as e:
                    self.logger.error(f"获取价格失败: {e}")
                    return None

            if not current_price:
                print_colored(f"    ❌ 无法获取当前价格", Colors.ERROR)
                return None

            # ========== 智能计算仓位和杠杆 ==========
            confidence = decision.get('confidence', 0.5)
            position_calc = self._calculate_smart_position_size(account_balance, confidence)
            base_amount = position_calc['base_amount']
            leverage = position_calc['leverage']
            leveraged_value = position_calc['leveraged_value']

            # ========== 获取交易对的最小要求 ==========
            min_notional = self._get_symbol_min_notional(symbol)

            # 检查是否满足最小要求（使用10倍名义价值检查）
            if position_calc['nominal_value_10x'] < 100.0:
                print_colored(f"    ⚠️ 10x名义价值 ${position_calc['nominal_value_10x']:.2f} < 最小要求 $100",
                              Colors.WARNING)
                # 如果不满足，调整基础金额
                base_amount = 10.0  # 强制使用10 USDT
                leveraged_value = base_amount * leverage
                print_colored(f"    ✅ 调整基础金额为: ${base_amount:.2f}", Colors.INFO)

            # 确保杠杆后价值满足交易所最小要求
            if leveraged_value < min_notional:
                print_colored(f"    ⚠️ 杠杆后价值 ${leveraged_value:.2f} < 交易所要求 ${min_notional}", Colors.WARNING)
                # 增加基础金额以满足要求
                base_amount = min_notional / leverage * 1.05  # 加5%缓冲
                leveraged_value = base_amount * leverage
                print_colored(f"    ✅ 调整基础金额为: ${base_amount:.2f}", Colors.INFO)

            # ========== 风险调整 ==========
            risk_adjustment = 1.0

            # 根据市场条件调整
            risks = decision.get('risks', [])
            if any(risk in str(risks) for risk in ['诱多', '诱空', '操纵']):
                risk_adjustment *= 0.7  # 如果有操纵风险，减少30%
                print_colored(f"    ⚠️ 检测到操纵风险，仓位减少30%", Colors.WARNING)

            # 根据信号分歧调整
            if '信号分歧' in str(risks):
                risk_adjustment *= 0.8
                print_colored(f"    ⚠️ 信号存在分歧，仓位减少20%", Colors.WARNING)

            # 应用风险调整
            final_base_amount = base_amount * risk_adjustment
            final_leveraged_value = final_base_amount * leverage

            # ========== 计算止损和止盈 ==========
            action = decision.get('action', 'BUY')

            # 基础止损比例（根据置信度调整）
            base_stop_loss_pct = 0.02 if confidence > 0.7 else 0.015  # 2% 或 1.5%

            # 计算止损价格
            if 'BUY' in action:
                stop_loss_price = current_price * (1 - base_stop_loss_pct)
                # 计算止盈价格（风险回报比 1:2 到 1:3）
                risk_reward_ratio = 2.5 if confidence > 0.8 else 2.0
                take_profit_price = current_price * (1 + base_stop_loss_pct * risk_reward_ratio)
            else:  # SELL
                stop_loss_price = current_price * (1 + base_stop_loss_pct)
                risk_reward_ratio = 2.5 if confidence > 0.8 else 2.0
                take_profit_price = current_price * (1 - base_stop_loss_pct * risk_reward_ratio)

            # ========== 计算数量 ==========
            quantity = final_leveraged_value / current_price

            # 获取精度要求
            try:
                symbol_info = self._get_symbol_info(symbol)
                if symbol_info:
                    # 获取数量精度
                    quantity_precision = symbol_info.get('quantityPrecision', 3)
                    # 格式化数量
                    quantity = round(quantity, quantity_precision)

                    # 检查最小数量
                    min_qty = float(symbol_info.get('minQty', 0.001))
                    if quantity < min_qty:
                        quantity = min_qty
                        print_colored(f"    ⚠️ 调整到最小数量: {quantity}", Colors.WARNING)
            except:
                # 默认精度
                quantity = round(quantity, 3)

            # ========== 构建交易参数 - 关键修复：确保包含 entry_price ==========
            trade_params = {
                'symbol': symbol,
                'side': 'BUY' if 'BUY' in action else 'SELL',
                'type': 'LIMIT',
                'quantity': quantity,
                'price': current_price,
                'entry_price': current_price,  # 添加 entry_price 键
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'leverage': leverage,
                'base_amount': final_base_amount,
                'leveraged_value': final_leveraged_value,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'risk_adjustment': risk_adjustment
            }

            # 打印交易参数
            print_colored(f"\n    📋 交易参数计算完成:", Colors.GREEN)
            print_colored(f"    • 方向: {trade_params['side']}", Colors.INFO)
            print_colored(f"    • 当前价格: ${current_price:.4f}", Colors.INFO)
            print_colored(f"    • 数量: {quantity} {symbol.replace('USDT', '')}", Colors.INFO)
            print_colored(f"    • 基础资金: ${final_base_amount:.2f} USDT", Colors.INFO)
            print_colored(f"    • 杠杆: {leverage}x", Colors.INFO)
            print_colored(f"    • 名义价值: ${final_leveraged_value:.2f}", Colors.INFO)
            print_colored(f"    • 止损: ${stop_loss_price:.4f} ({base_stop_loss_pct * 100:.1f}%)", Colors.INFO)
            print_colored(f"    • 止盈: ${take_profit_price:.4f} (1:{risk_reward_ratio})", Colors.INFO)

            return trade_params

        except Exception as e:
            self.logger.error(f"计算交易参数失败: {e}")
            print_colored(f"    ❌ 计算交易参数失败: {str(e)}", Colors.ERROR)
            import traceback
            traceback.print_exc()
            return None


    def _integrate_analyses_v2(self, game_theory: Dict, technical: Dict, symbol: str) -> Dict[str, Any]:
        """
        整合分析 V2 - 趋势跟随策略

        核心改变：
        1. 技术分析权重提高到70%，博弈论降到30%
        2. 强调趋势跟随，避免逆势交易
        3. 使用多重确认机制
        """
        integrated = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'final_score': 0.0,
            'signals': [],
            'risks': [],
            'reason': '',
            'strategy_type': 'TREND_FOLLOWING'  # 明确策略类型
        }

        # 检查是否有错误
        if 'error' in game_theory or 'error' in technical:
            integrated['reason'] = '分析数据不完整'
            return integrated

        # ========== 新权重分配 ==========
        # 技术分析为主，博弈论为辅
        tech_weight = 0.7  # 技术分析占70%
        game_weight = 0.3  # 博弈论占30%

        # ========== 趋势判断（最重要）==========
        trend_direction = technical.get('trend', {}).get('direction', 'NEUTRAL')
        trend_strength = technical.get('trend', {}).get('strength', 0)
        adx = technical.get('adx', 20)

        # 获取技术指标
        rsi = technical.get('momentum', {}).get('rsi', 50)
        tech_score = technical.get('signal_strength', 0)

        # ========== 核心规则：顺势而为 ==========
        # 规则1：强趋势时只做趋势方向的交易
        if trend_direction == 'UP' and trend_strength >= 1.5:
            # 上升趋势中
            if rsi > 80:
                # RSI超买，但在强势上涨中，这是强势信号而非做空信号
                integrated['signals'].append("强势上涨中的超买")
                integrated['risks'].append("可能短期回调")
                # 等待回调再做多，而非做空
                integrated['action'] = 'HOLD'
                integrated['reason'] = "强势上涨但RSI过高，等待回调做多机会"
            elif 40 < rsi < 65:
                # RSI在合理区间，趋势健康
                integrated['action'] = 'BUY'
                integrated['confidence'] = 0.7
                integrated['signals'].append("趋势健康，动量适中")
                integrated['strategy_type'] = 'TREND_CONTINUATION'
            elif rsi < 40:
                # 上升趋势中的超卖，绝佳买入机会
                integrated['action'] = 'BUY'
                integrated['confidence'] = 0.85
                integrated['signals'].append("上升趋势回调买入机会")
                integrated['strategy_type'] = 'PULLBACK_BUY'

        elif trend_direction == 'DOWN' and trend_strength >= 1.5:
            # 下降趋势中
            if rsi < 20:
                # RSI超卖，但在下跌趋势中，这可能是继续下跌的信号
                integrated['signals'].append("弱势下跌中的超卖")
                integrated['risks'].append("可能短期反弹")
                integrated['action'] = 'HOLD'
                integrated['reason'] = "弱势下跌但RSI过低，等待反弹做空机会"
            elif 35 < rsi < 60:
                # RSI在合理区间，趋势健康
                integrated['action'] = 'SELL'
                integrated['confidence'] = 0.7
                integrated['signals'].append("下跌趋势继续")
                integrated['strategy_type'] = 'TREND_CONTINUATION'
            elif rsi > 60:
                # 下跌趋势中的超买，绝佳做空机会
                integrated['action'] = 'SELL'
                integrated['confidence'] = 0.85
                integrated['signals'].append("下跌趋势反弹做空机会")
                integrated['strategy_type'] = 'PULLBACK_SELL'

        else:
            # 无明确趋势或弱趋势
            integrated['action'] = 'HOLD'
            integrated['reason'] = "无明确趋势，等待突破"
            integrated['strategy_type'] = 'RANGE_BOUND'

        # ========== 博弈论验证（次要）==========
        if integrated['action'] != 'HOLD':
            whale_intent = game_theory.get('whale_intent', 'NEUTRAL')
            game_confidence = game_theory.get('confidence', 0)

            # 博弈论只用于确认，不用于反转信号
            if integrated['action'] == 'BUY':
                if whale_intent == 'ACCUMULATION':
                    integrated['confidence'] += 0.1
                    integrated['signals'].append("庄家吸筹确认")
                elif whale_intent == 'DISTRIBUTION' and game_confidence > 0.7:
                    # 强烈的派发信号，降低做多信心
                    integrated['confidence'] -= 0.2
                    integrated['risks'].append("庄家可能在派发")

            elif integrated['action'] == 'SELL':
                if whale_intent == 'DISTRIBUTION':
                    integrated['confidence'] += 0.1
                    integrated['signals'].append("庄家派发确认")
                elif whale_intent == 'ACCUMULATION' and game_confidence > 0.7:
                    integrated['confidence'] -= 0.2
                    integrated['risks'].append("庄家可能在吸筹")

        # ========== 最终决策 ==========
        if integrated['confidence'] < 0.5:
            integrated['action'] = 'HOLD'
            integrated['reason'] = "信号不够强烈"

        # 计算最终评分
        integrated['final_score'] = integrated['confidence'] * 10
        integrated['game_theory_analysis'] = game_theory
        integrated['technical_analysis'] = technical

        # 打印决策逻辑
        print_colored(f"\n    🎯 策略决策:", Colors.CYAN + Colors.BOLD)
        print_colored(f"      • 趋势方向: {trend_direction} (强度: {trend_strength:.1f})", Colors.INFO)

        # 确保 RSI 不是 NaN
        if pd.isna(rsi):
            rsi = 50

        print_colored(f"      • RSI: {rsi:.1f}", Colors.INFO)
        print_colored(f"      • 策略类型: {integrated['strategy_type']}", Colors.INFO)
        print_colored(f"      • 决策: {integrated['action']}", Colors.INFO)
        print_colored(f"      • 置信度: {integrated['confidence']:.1%}", Colors.INFO)

        return integrated

    def _update_positions_enhanced(self):
        """
        增强的持仓更新 - 提供详细的持仓状态
        """
        if not self.open_positions:
            print_colored("\n📊 当前无持仓", Colors.GRAY)
            return

        print_colored(f"\n{'=' * 60}", Colors.BLUE)
        print_colored("📊 持仓状态更新", Colors.CYAN + Colors.BOLD)
        print_colored(f"{'=' * 60}", Colors.BLUE)

        total_pnl = 0
        total_value = 0
        positions_to_close = []

        for idx, position in enumerate(self.open_positions, 1):
            try:
                symbol = position['symbol']

                # 获取当前价格
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # 计算盈亏
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side']
                leverage = position.get('leverage', 10)

                if side == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    pnl_amount = (current_price - entry_price) * quantity
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    pnl_amount = (entry_price - current_price) * quantity

                # 计算持仓时间
                holding_time = (time.time() - position['open_time']) / 3600  # 小时

                # 计算到止损/止盈的距离
                if side == 'BUY':
                    to_sl = (position['stop_loss'] - current_price) / current_price * 100
                    to_tp = (position['take_profit'] - current_price) / current_price * 100
                else:
                    to_sl = (current_price - position['stop_loss']) / current_price * 100
                    to_tp = (current_price - position['take_profit']) / current_price * 100

                # 显示颜色
                pnl_color = Colors.GREEN if pnl_pct > 0 else Colors.RED

                print_colored(f"\n持仓 #{idx}: {symbol} {side}", Colors.CYAN)
                print_colored(f"  📍 入场价: ${entry_price:.4f}", Colors.INFO)
                print_colored(f"  💹 当前价: ${current_price:.4f}", Colors.INFO)
                print_colored(f"  📊 盈亏: {pnl_color}{pnl_pct:+.2f}% (${pnl_amount:+.2f}){Colors.RESET}", Colors.INFO)
                print_colored(f"  ⏱️ 持仓时间: {holding_time:.1f}小时", Colors.INFO)
                print_colored(f"  🎯 止盈距离: {to_tp:+.1f}%", Colors.GREEN if to_tp > 0 else Colors.GRAY)
                print_colored(f"  🛡️ 止损距离: {to_sl:+.1f}%", Colors.RED if to_sl < 5 else Colors.GRAY)
                print_colored(f"  📈 杠杆: {leverage}x", Colors.INFO)

                # 风险警告
                if to_sl < 1:
                    print_colored(f"  ⚠️ 警告: 接近止损位！", Colors.YELLOW + Colors.BOLD)
                elif pnl_pct > 5:
                    print_colored(f"  💰 建议: 可考虑部分止盈", Colors.GREEN)
                elif holding_time > 24 and abs(pnl_pct) < 1:
                    print_colored(f"  ⏰ 提示: 持仓超过24小时但盈亏有限", Colors.YELLOW)

                # 检查是否需要调整止损（移动止损）
                if pnl_pct > 3 and side == 'BUY':
                    suggested_sl = current_price * 0.98
                    if suggested_sl > position['stop_loss']:
                        print_colored(f"  💡 建议: 可将止损上移至 ${suggested_sl:.4f}", Colors.CYAN)
                elif pnl_pct > 3 and side == 'SELL':
                    suggested_sl = current_price * 1.02
                    if suggested_sl < position['stop_loss']:
                        print_colored(f"  💡 建议: 可将止损下移至 ${suggested_sl:.4f}", Colors.CYAN)

                # 更新总计
                total_pnl += pnl_amount
                total_value += position.get('position_value', 0)

                # 检查是否触发止损止盈
                if (side == 'BUY' and current_price <= position['stop_loss']) or \
                        (side == 'SELL' and current_price >= position['stop_loss']):
                    positions_to_close.append((position, current_price, 'STOP_LOSS'))
                elif (side == 'BUY' and current_price >= position['take_profit']) or \
                        (side == 'SELL' and current_price <= position['take_profit']):
                    positions_to_close.append((position, current_price, 'TAKE_PROFIT'))

            except Exception as e:
                self.logger.error(f"更新持仓{position['symbol']}失败: {e}")
                print_colored(f"  ❌ 更新失败: {str(e)}", Colors.ERROR)

        # 显示汇总
        print_colored(f"\n{'=' * 40}", Colors.BLUE)
        print_colored(f"💰 持仓汇总:", Colors.CYAN)
        print_colored(f"  • 总持仓数: {len(self.open_positions)}", Colors.INFO)
        print_colored(f"  • 总持仓价值: ${total_value:.2f}", Colors.INFO)
        total_pnl_color = Colors.GREEN if total_pnl > 0 else Colors.RED
        print_colored(f"  • 总盈亏: {total_pnl_color}${total_pnl:+.2f}{Colors.RESET}", Colors.INFO)
        print_colored(f"{'=' * 40}", Colors.BLUE)

        # 处理需要平仓的持仓
        for position, exit_price, reason in positions_to_close:
            print_colored(f"\n⚠️ 触发{reason}，准备平仓 {position['symbol']}", Colors.YELLOW)
            self._close_position(position, exit_price, reason)

    def run_trading_cycle_v2(self):
        """
        改进的交易循环 - 包含详细的持仓管理
        """
        try:
            self.trade_cycle += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print_colored(f"\n{'=' * 80}", Colors.BLUE)
            print_colored(f"🚀 交易循环 #{self.trade_cycle} - {current_time}", Colors.BLUE + Colors.BOLD)
            print_colored(f"{'=' * 80}", Colors.BLUE)

            # 1. 首先更新和显示当前持仓
            self._update_positions_enhanced()

            # 2. 检查账户状态
            try:
                account_info = self.client.futures_account()
                account_balance = float(account_info['totalWalletBalance'])
                available_balance = float(account_info['availableBalance'])
                unrealized_pnl = float(account_info['totalUnrealizedProfit'])

                print_colored(f"\n💰 账户状态:", Colors.CYAN)
                print_colored(f"   总余额: ${account_balance:.2f}", Colors.INFO)
                print_colored(f"   可用余额: ${available_balance:.2f}", Colors.INFO)
                print_colored(f"   未实现盈亏: ${unrealized_pnl:+.2f}",
                              Colors.GREEN if unrealized_pnl > 0 else Colors.RED)
                print_colored(f"   已用保证金: ${account_balance - available_balance:.2f}", Colors.INFO)

            except Exception as e:
                self.logger.error(f"获取账户信息失败: {e}")
                print_colored(f"❌ 获取账户信息失败: {e}", Colors.ERROR)
                return

            # 3. 风险检查
            if self.risk_manager:
                can_trade, reason = self.risk_manager.can_open_position()
                if not can_trade:
                    print_colored(f"\n⚠️ 风险管理限制: {reason}", Colors.WARNING)
                    return

                # 显示风险状态
                risk_summary = self.risk_manager.get_risk_summary()
                print_colored(f"\n📊 风险状态:", Colors.CYAN)
                print_colored(f"   日内亏损: {risk_summary['daily_loss']:.2f}%", Colors.INFO)
                print_colored(f"   当前回撤: {risk_summary['current_drawdown']:.2f}%", Colors.INFO)
                print_colored(f"   风险等级: {risk_summary['risk_status']}", Colors.INFO)

            # 4. 检查是否可以开新仓
            current_positions = len(self.open_positions)
            max_positions = self.max_positions

            print_colored(f"\n📈 仓位管理:", Colors.CYAN)
            print_colored(f"   当前持仓: {current_positions}/{max_positions}", Colors.INFO)

            if current_positions >= max_positions:
                print_colored(f"   状态: 已达最大持仓，专注管理现有仓位", Colors.WARNING)
                return
            else:
                print_colored(f"   状态: 可开新仓 (剩余名额: {max_positions - current_positions})", Colors.GREEN)

            # 5. 寻找新的交易机会
            print_colored(f"\n🔍 扫描市场机会...", Colors.CYAN)
            asyncio.run(self._run_integrated_analysis(account_balance))

            # 6. 显示循环总结
            self._print_enhanced_summary()

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}", exc_info=True)
            print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)

    def _print_enhanced_summary(self):
        """增强的循环总结"""
        print_colored(f"\n{'=' * 60}", Colors.BLUE)
        print_colored(f"📊 循环 #{self.trade_cycle} 总结", Colors.CYAN + Colors.BOLD)

        # 性能统计
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            stats = self.performance_monitor.get_current_stats()
            basic_stats = stats['basic_stats']

            if basic_stats['total_trades'] > 0:
                print_colored(f"   累计交易: {basic_stats['total_trades']}笔", Colors.INFO)
                print_colored(f"   胜率: {basic_stats['win_rate'] * 100:.1f}%", Colors.INFO)
                print_colored(f"   盈利因子: {basic_stats['profit_factor']:.2f}", Colors.INFO)

        print_colored(f"   下次扫描: {self.config.get('SCAN_INTERVAL', 300)}秒后", Colors.INFO)
        print_colored(f"{'=' * 60}", Colors.BLUE)

    def _display_trading_opportunity(self, opportunity: Dict):
        """
        显示交易机会的详细信息 - 修复版本
        """
        try:
            print_colored(f"\n🎯 交易机会详情:", Colors.CYAN + Colors.BOLD)

            params = opportunity.get('trade_params', {})
            if not params:
                print_colored("   ❌ 交易参数缺失", Colors.ERROR)
                return

            action_color = Colors.GREEN if 'BUY' in opportunity.get('action', '') else Colors.RED

            # 获取价格信息 - 兼容不同的键名
            entry_price = params.get('entry_price') or params.get('price', 0)
            stop_loss = params.get('stop_loss', 0)
            take_profit = params.get('take_profit', 0)

            # 基本信息
            print_colored(f"   方向: {action_color}{opportunity.get('action', 'UNKNOWN')}{Colors.RESET}", Colors.INFO)
            print_colored(f"   入场价: ${entry_price:.4f}", Colors.INFO)

            # 止损信息
            if stop_loss and entry_price:
                if 'BUY' in opportunity.get('action', ''):
                    stop_loss_pct = abs((stop_loss - entry_price) / entry_price * 100)
                else:  # SELL
                    stop_loss_pct = abs((entry_price - stop_loss) / entry_price * 100)
                print_colored(f"   止损价: ${stop_loss:.4f} (-{stop_loss_pct:.1f}%)", Colors.INFO)

            # 止盈信息
            if take_profit and entry_price:
                if 'BUY' in opportunity.get('action', ''):
                    take_profit_pct = abs((take_profit - entry_price) / entry_price * 100)
                else:  # SELL
                    take_profit_pct = abs((entry_price - take_profit) / entry_price * 100)
                print_colored(f"   止盈价: ${take_profit:.4f} (+{take_profit_pct:.1f}%)", Colors.INFO)

            # 风险回报比
            risk_reward_ratio = params.get('risk_reward_ratio', 0)
            if risk_reward_ratio:
                print_colored(f"   风险回报比: 1:{risk_reward_ratio:.1f}", Colors.INFO)

            # 仓位信息
            if 'quantity' in params:
                print_colored(f"   数量: {params['quantity']}", Colors.INFO)

            if 'leveraged_value' in params:
                print_colored(f"   名义价值: ${params['leveraged_value']:.2f}", Colors.INFO)

            if 'leverage' in params:
                print_colored(f"   杠杆: {params['leverage']}x", Colors.INFO)

            # 置信度
            confidence = opportunity.get('confidence', 0)
            if confidence:
                confidence_color = Colors.GREEN if confidence > 0.7 else Colors.YELLOW if confidence > 0.5 else Colors.RED
                print_colored(f"   置信度: {confidence_color}{confidence:.1%}{Colors.RESET}", Colors.INFO)

            # 信号列表
            signals = opportunity.get('signals', [])
            if signals:
                print_colored(f"   信号:", Colors.INFO)
                for signal in signals[:5]:  # 最多显示5个信号
                    print_colored(f"     • {signal}", Colors.SUCCESS)

            # 风险提示
            risks = opportunity.get('risks', [])
            if risks:
                print_colored(f"   风险:", Colors.INFO)
                for risk in risks[:3]:  # 最多显示3个风险
                    print_colored(f"     • {risk}", Colors.WARNING)

        except Exception as e:
            print_colored(f"   ❌ 显示交易机会详情失败: {str(e)}", Colors.ERROR)
            import traceback
            traceback.print_exc()

    def _determine_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """判断市场趋势"""
        # 使用多重移动均线判断
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            sma20 = df['SMA_20'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            price = df['close'].iloc[-1]

            if price > sma20 > sma50 > sma200:
                return {'direction': 'UP', 'strength': 2.0}
            elif price < sma20 < sma50 < sma200:
                return {'direction': 'DOWN', 'strength': 2.0}
            elif price > sma20 > sma50:
                return {'direction': 'UP', 'strength': 1.0}
            elif price < sma20 < sma50:
                return {'direction': 'DOWN', 'strength': 1.0}

        return {'direction': 'NEUTRAL', 'strength': 0.0}

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析动量指标 - 确保返回字典类型"""
        momentum = {}

        try:
            # RSI
            if 'RSI' in df.columns:
                momentum['rsi'] = float(df['RSI'].iloc[-1])
            else:
                momentum['rsi'] = 50.0

            # MACD信号
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                signal = df['MACD_signal'].iloc[-1]

                # 添加安全检查
                if len(df) >= 2:
                    prev_macd = df['MACD'].iloc[-2]
                    prev_signal = df['MACD_signal'].iloc[-2]

                    if macd > signal and prev_macd <= prev_signal:
                        momentum['macd_signal'] = 'BULLISH'
                    elif macd < signal and prev_macd >= prev_signal:
                        momentum['macd_signal'] = 'BEARISH'
                    else:
                        momentum['macd_signal'] = 'NEUTRAL'
                else:
                    momentum['macd_signal'] = 'NEUTRAL'
            else:
                momentum['macd_signal'] = 'NEUTRAL'

            # Stochastic
            if 'STOCHk' in df.columns:
                momentum['stoch_k'] = float(df['STOCHk'].iloc[-1])
                if 'STOCHd' in df.columns:
                    momentum['stoch_d'] = float(df['STOCHd'].iloc[-1])
                else:
                    momentum['stoch_d'] = 50.0
            else:
                momentum['stoch_k'] = 50.0
                momentum['stoch_d'] = 50.0

            # 确保返回的是字典，不是numpy对象
            return dict(momentum)

        except Exception as e:
            print_colored(f"⚠️ 动量分析错误: {e}", Colors.WARNING)
            # 返回默认值
            return {
                'rsi': 50.0,
                'macd_signal': 'NEUTRAL',
                'stoch_k': 50.0,
                'stoch_d': 50.0
            }

    def fix_initialization_issues():
        """修复初始化问题"""
        print_colored("🔧 开始修复初始化问题...", Colors.CYAN)

        # 1. 检查必要的依赖
        required_modules = ['numpy', 'pandas', 'talib', 'binance']
        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
                print_colored(f"✅ {module} 模块正常", Colors.GREEN)
            except ImportError:
                missing_modules.append(module)
                print_colored(f"❌ 缺少 {module} 模块", Colors.RED)

        if missing_modules:
            print_colored(f"请安装缺少的模块: pip install {' '.join(missing_modules)}", Colors.WARNING)
            return False

        # 2. 检查配置文件
        config_files = ['config.json', 'api_keys.json']
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    import json
                    json.load(f)
                print_colored(f"✅ {config_file} 配置正常", Colors.GREEN)
            except FileNotFoundError:
                print_colored(f"⚠️ {config_file} 配置文件不存在", Colors.WARNING)
            except json.JSONDecodeError:
                print_colored(f"❌ {config_file} 配置文件格式错误", Colors.RED)

        # 3. 修复流动性感知止损初始化问题
        print_colored("🔧 修复流动性感知止损初始化...", Colors.CYAN)
        print_colored("建议：先初始化流动性猎手系统，再初始化止损系统", Colors.INFO)

        return True

    def safe_get_dict_value(obj, key, default='NEUTRAL'):
        """安全获取字典值，处理类型错误"""
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                print_colored(f"⚠️ 预期字典类型，实际类型: {type(obj)}", Colors.WARNING)
                return default
        except Exception as e:
            print_colored(f"⚠️ 获取字典值错误: {e}", Colors.WARNING)
            return default

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析成交量模式 - 增强版"""
        volume_analysis = {
            'current': 0,
            'average': 0,
            'ratio': 0,
            'trend': 'NEUTRAL',
            'has_spike': False,
            'spike_info': {}
        }

        if 'volume' not in df.columns or len(df) < 20:
            return volume_analysis

        try:
            # 基础数据
            current_vol = df['volume'].iloc[-1]
            vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]

            # 使用成交量突变检测器
            if hasattr(self, 'volume_spike_detector') and self.volume_spike_detector:
                spike_result = self.volume_spike_detector.detect_volume_spike(df)

                if spike_result['has_spike']:
                    volume_analysis['has_spike'] = True
                    volume_analysis['spike_info'] = spike_result

                    # 设置趋势
                    if spike_result['spike_direction'] == 'UP':
                        volume_analysis['trend'] = 'EXPANDING_UP'
                    elif spike_result['spike_direction'] == 'DOWN':
                        volume_analysis['trend'] = 'EXPANDING_DOWN'
                    else:
                        volume_analysis['trend'] = 'CONTRACTING'

            # 如果没有突变，使用传统方法
            if not volume_analysis['has_spike']:
                vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
                ratio_5vs20 = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0

                if ratio_5vs20 > 1.1:
                    volume_analysis['trend'] = 'INCREASING'
                elif ratio_5vs20 < 0.9:
                    volume_analysis['trend'] = 'DECREASING'
                else:
                    volume_analysis['trend'] = 'NEUTRAL'

            # 填充基础数据
            volume_analysis['current'] = current_vol
            volume_analysis['average'] = vol_ma20
            volume_analysis['ratio'] = current_vol / vol_ma20 if vol_ma20 > 0 else 1.0

        except Exception as e:
            self.logger.error(f"成交量分析错误: {e}")

        return volume_analysis

    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """寻找关键支撑阻力位"""
        levels = {'supports': [], 'resistances': []}

        # 使用最近的高低点
        recent_highs = df['high'].rolling(20).max().dropna().unique()[-5:]
        recent_lows = df['low'].rolling(20).min().dropna().unique()[-5:]

        current_price = df['close'].iloc[-1]

        # 支撑位（低于当前价）
        levels['supports'] = sorted([low for low in recent_lows if low < current_price * 0.995], reverse=True)[:3]

        # 阻力位（高于当前价）
        levels['resistances'] = sorted([high for high in recent_highs if high > current_price * 1.005])[:3]

        return levels

    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """检测价格形态"""
        patterns = []

        # 简单的形态检测
        closes = df['close'].tail(10).values

        # 检测V型反转
        if len(closes) >= 5:
            mid_idx = len(closes) // 2
            left_min = min(closes[:mid_idx])
            right_min = min(closes[mid_idx:])
            current = closes[-1]

            if left_min < current * 0.98 and right_min < current * 0.98 and current > closes[0]:
                patterns.append("V型反转")

        # 检测突破
        recent_high = df['high'].tail(20).max()
        if df['close'].iloc[-1] > recent_high * 0.995:
            patterns.append("突破近期高点")

        recent_low = df['low'].tail(20).min()
        if df['close'].iloc[-1] < recent_low * 1.005:
            patterns.append("突破近期低点")

        return patterns

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _execute_integrated_trade(self, opportunity: Dict, account_balance: float):
        """执行整合分析后的交易 - 智能最小订单版本"""
        try:
            symbol = opportunity['symbol']
            action = opportunity['action']
            params = opportunity['trade_params']

            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"💰 执行交易: {symbol}", Colors.CYAN + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            # 再次确认市场状态
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 价格滑点检查
            slippage = abs(current_price - params['entry_price']) / params['entry_price']
            if slippage > 0.002:  # 0.2%滑点
                print_colored(f"⚠️ 价格滑点: {slippage:.2%}", Colors.WARNING)
                print_colored(f"   计划价格: ${params['entry_price']:.4f}", Colors.INFO)
                print_colored(f"   当前价格: ${current_price:.4f}", Colors.INFO)

                # 重新计算参数
                params = self._recalculate_params_with_slippage(params, current_price, action)
                if not params:
                    print_colored("❌ 滑点过大，取消交易", Colors.ERROR)
                    return

            # 获取交易规则
            symbol_info = next((s for s in self.client.futures_exchange_info()['symbols'] if s['symbol'] == symbol),
                               None)
            if not symbol_info:
                print_colored(f"❌ 无法获取{symbol}交易规则", Colors.ERROR)
                return

            # 获取精度信息
            step_size = float(next(f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'))
            min_qty = float(next(f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'))
            price_precision = symbol_info['pricePrecision']
            quantity_precision = symbol_info['quantityPrecision']

            # ========== 使用向上取整确保满足最小金额 ==========
            MIN_ORDER_VALUE = params.get('min_notional', 15.0)

            # 计算满足最小金额的数量（加缓冲）
            min_quantity_for_value = (MIN_ORDER_VALUE * 1.02) / current_price

            # 取较大值
            target_quantity = max(params['quantity'], min_quantity_for_value)

            # 精度调整 - 向上取整
            quantity = self._round_quantity_up(target_quantity, step_size)

            # 再次验证
            expected_value = quantity * current_price

            print_colored(f"\n📊 订单验证:", Colors.CYAN)
            print_colored(f"   最小名义价值: ${MIN_ORDER_VALUE:.2f}", Colors.INFO)
            print_colored(f"   调整后数量: {quantity:.{quantity_precision}f}", Colors.INFO)
            print_colored(f"   预期订单价值: ${expected_value:.2f}", Colors.INFO)

            # 最终检查
            if expected_value < MIN_ORDER_VALUE * 0.98:  # 允许2%的误差
                print_colored(f"❌ 订单价值 ${expected_value:.2f} 小于最小要求 ${MIN_ORDER_VALUE}", Colors.ERROR)
                return

            # 确保满足最小数量要求
            if quantity < min_qty:
                print_colored(f"❌ 数量 {quantity} 小于最小要求 {min_qty}", Colors.ERROR)
                return

            # 设置杠杆
            leverage = params.get('leverage', 10)
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                print_colored(f"✅ 设置杠杆: {leverage}x", Colors.GREEN)
            except Exception as e:
                # 如果是杠杆已经设置的错误，可以忽略
                if "No need to change leverage" not in str(e):
                    print_colored(f"⚠️ 设置杠杆失败: {e}", Colors.WARNING)

            # 设置逐仓模式（可选）
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
            except Exception as e:
                # 如果已经是逐仓模式，忽略错误
                if "No need to change margin type" not in str(e):
                    print_colored(f"⚠️ 设置保证金模式失败: {e}", Colors.WARNING)

            # 下单
            print_colored(f"\n📤 发送订单...", Colors.INFO)
            print_colored(f"   交易对: {symbol}", Colors.INFO)
            print_colored(f"   方向: {action}", Colors.INFO)
            print_colored(f"   数量: {quantity:.{quantity_precision}f}", Colors.INFO)
            print_colored(f"   预期价值: ${expected_value:.2f}", Colors.INFO)
            print_colored(f"   杠杆: {leverage}x", Colors.INFO)

            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=action,
                    type='MARKET',
                    quantity=quantity
                )

                if order.get('status') == 'FILLED':
                    fill_price = float(order['avgPrice'])
                    actual_value = quantity * fill_price
                    commission = float(order.get('commission', 0))

                    print_colored(f"\n✅ 订单成交!", Colors.GREEN + Colors.BOLD)
                    print_colored(f"   订单ID: {order['orderId']}", Colors.INFO)
                    print_colored(f"   成交价格: ${fill_price:.{price_precision}f}", Colors.INFO)
                    print_colored(f"   成交数量: {quantity:.{quantity_precision}f}", Colors.INFO)
                    print_colored(f"   实际成交价值: ${actual_value:.2f}", Colors.INFO)
                    if commission > 0:
                        print_colored(f"   手续费: ${commission:.4f}", Colors.INFO)

                    # 创建持仓记录
                    position = {
                        'symbol': symbol,
                        'side': action,
                        'quantity': quantity,
                        'entry_price': fill_price,
                        'stop_loss': params['stop_loss'],
                        'take_profit': params['take_profit'],
                        'open_time': time.time(),
                        'order_id': order['orderId'],
                        'leverage': leverage,
                        'position_value': actual_value,
                        'analysis': {
                            'whale_intent': opportunity['game_theory_analysis'].get('whale_intent'),
                            'confidence': opportunity['confidence'],
                            'signals': opportunity['signals']
                        }
                    }

                    self.open_positions.append(position)

                    # 设置止损止盈
                    print_colored(f"\n⚙️ 设置止损止盈...", Colors.INFO)
                    self._set_stop_orders_enhanced(position, price_precision)

                    # 记录到性能监控
                    if hasattr(self, 'performance_monitor') and self.performance_monitor:
                        self.performance_monitor.record_trade_open({
                            'symbol': symbol,
                            'side': action,
                            'price': fill_price,
                            'quantity': quantity,
                            'leverage': leverage,
                            'stop_loss': params['stop_loss'],
                            'take_profit': params['take_profit'],
                            'market_analysis': opportunity['game_theory_analysis'],
                            'strategy_tags': ['integrated'] + opportunity['signals']
                        })

                    # 更新风险管理
                    if hasattr(self, 'risk_manager') and self.risk_manager:
                        self.risk_manager.update_daily_stats(0, account_balance)  # 开仓时还没有盈亏

                    print_colored(f"\n🎉 交易执行成功!", Colors.GREEN + Colors.BOLD)
                    print_colored(f"{'=' * 60}", Colors.BLUE)

                else:
                    print_colored(f"❌ 订单状态异常: {order.get('status')}", Colors.ERROR)
                    self.logger.error(f"订单状态异常: {order}")

            except Exception as order_error:
                error_msg = str(order_error)
                print_colored(f"\n❌ 下单失败: {error_msg}", Colors.ERROR)

                # 特殊错误处理
                if "Order's notional must be no smaller than" in error_msg:
                    # 提取实际要求的最小值
                    import re
                    match = re.search(r'than (\d+)', error_msg)
                    if match:
                        actual_min = float(match.group(1))
                        print_colored(f"   该交易对实际最小订单要求: ${actual_min}", Colors.WARNING)
                        print_colored(f"   当前订单价值: ${expected_value:.2f}", Colors.WARNING)
                        print_colored(f"   建议：增加仓位或选择其他交易对", Colors.INFO)

                self.logger.error(f"下单失败: {order_error}", exc_info=True)

        except Exception as e:
            self.logger.error(f"执行交易失败: {e}", exc_info=True)
            print_colored(f"❌ 执行交易失败: {str(e)}", Colors.ERROR)

    def _round_quantity_up(self, quantity: float, step_size: float) -> float:
        """
        按交易所规则调整数量精度 - 向上取整版本
        确保满足最小金额要求
        """
        import math

        # 计算需要多少个step_size
        steps = quantity / step_size

        # 向上取整到最近的step_size倍数
        rounded_steps = math.ceil(steps)

        # 计算最终数量
        rounded_quantity = rounded_steps * step_size

        # 确保精度正确
        precision = len(str(step_size).split('.')[-1])
        return round(rounded_quantity, precision)

    def _set_stop_orders_enhanced(self, position: Dict, price_precision: int):
        """
        设置止损止盈订单 - 增强版
        """
        try:
            symbol = position['symbol']
            quantity = position['quantity']

            # 止损单
            stop_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            stop_price = round(position['stop_loss'], price_precision)

            try:
                stop_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_price,
                    reduceOnly=True
                )
                print_colored(f"   ✅ 止损设置成功 @ ${stop_price}", Colors.GREEN)
                position['stop_order_id'] = stop_order['orderId']
            except Exception as e:
                print_colored(f"   ❌ 止损设置失败: {e}", Colors.ERROR)
                # 可以考虑使用其他方式，如限价止损单
                try:
                    stop_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=stop_side,
                        type='STOP',
                        quantity=quantity,
                        price=stop_price,
                        stopPrice=stop_price,
                        reduceOnly=True,
                        timeInForce='GTC'
                    )
                    print_colored(f"   ✅ 限价止损设置成功 @ ${stop_price}", Colors.GREEN)
                except Exception as e2:
                    print_colored(f"   ❌ 限价止损也失败: {e2}", Colors.ERROR)

            # 止盈单
            tp_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            tp_price = round(position['take_profit'], price_precision)

            try:
                tp_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=tp_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=tp_price,
                    reduceOnly=True
                )
                print_colored(f"   ✅ 止盈设置成功 @ ${tp_price}", Colors.GREEN)
                position['tp_order_id'] = tp_order['orderId']
            except Exception as e:
                print_colored(f"   ❌ 止盈设置失败: {e}", Colors.ERROR)
                # 可以考虑使用限价单
                try:
                    tp_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type='LIMIT',
                        quantity=quantity,
                        price=tp_price,
                        reduceOnly=True,
                        timeInForce='GTC'
                    )
                    print_colored(f"   ✅ 限价止盈设置成功 @ ${tp_price}", Colors.GREEN)
                except Exception as e2:
                    print_colored(f"   ❌ 限价止盈也失败: {e2}", Colors.ERROR)

        except Exception as e:
            self.logger.error(f"设置止损止盈失败: {e}")
            print_colored(f"   ❌ 设置止损止盈失败: {e}", Colors.ERROR)

    def _calculate_smart_position_size(self, account_balance: float, confidence: float) -> Dict[str, float]:
        """
        智能计算仓位大小和杠杆

        规则：
        1. 杠杆范围：15x-20x（根据置信度调整）
        2. 最小仓位：10 USDT（实际资金）× 10倍 = 100 USDT杠杆后价值
        3. 当账户≥1000 USDT时，使用账户的1%作为基础金额

        返回：
            包含 base_amount（基础金额）、leverage（杠杆）、leveraged_value（杠杆后价值）
        """
        # 账户余额阈值
        WEALTH_THRESHOLD = 1000.0  # 账户达到1000 USDT后改变策略

        # 杠杆范围
        MIN_LEVERAGE = 15
        MAX_LEVERAGE = 20

        # 根据置信度计算杠杆（置信度越高，杠杆越高）
        # confidence 范围通常是 0.4-0.9
        leverage_range = MAX_LEVERAGE - MIN_LEVERAGE
        leverage = MIN_LEVERAGE + (leverage_range * max(0, min(1, (confidence - 0.4) / 0.5)))
        leverage = int(leverage)  # 取整

        # 计算基础金额（实际投入的资金）
        if account_balance < WEALTH_THRESHOLD:
            # 账户小于1000 USDT时，固定使用10 USDT
            base_amount = 10.0
            position_mode = "固定金额模式"
        else:
            # 账户大于等于1000 USDT时，使用1%
            base_amount = account_balance * 0.01
            position_mode = "百分比模式(1%)"

        # 确保基础金额不超过账户余额的某个比例（安全限制）
        max_base_amount = account_balance * 0.10  # 最多使用10%的资金作为基础金额
        if base_amount > max_base_amount:
            base_amount = max_base_amount
            position_mode += " (已限制到10%)"

        # 计算10倍杠杆后的名义价值（用于检查最小要求）
        nominal_value_10x = base_amount * 10

        # 计算实际杠杆后的价值
        leveraged_value = base_amount * leverage

        # 打印计算过程
        print_colored(f"\n    📊 智能仓位计算:", Colors.CYAN)
        print_colored(f"      • 账户余额: ${account_balance:.2f}", Colors.INFO)
        print_colored(f"      • 仓位模式: {position_mode}", Colors.INFO)
        print_colored(f"      • 基础金额: ${base_amount:.2f}", Colors.INFO)
        print_colored(f"      • 置信度: {confidence:.1%}", Colors.INFO)
        print_colored(f"      • 选择杠杆: {leverage}x", Colors.INFO)
        print_colored(f"      • 10x名义价值: ${nominal_value_10x:.2f} (最小要求检查)", Colors.INFO)
        print_colored(f"      • 实际杠杆价值: ${leveraged_value:.2f}", Colors.INFO)

        return {
            'base_amount': base_amount,
            'leverage': leverage,
            'leveraged_value': leveraged_value,
            'nominal_value_10x': nominal_value_10x,
            'position_mode': position_mode
        }

    def _get_symbol_min_notional(self, symbol: str) -> float:
        """获取交易对的最小名义价值要求"""
        try:
            # 获取交易规则
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    filters = s.get('filters', [])
                    for f in filters:
                        if f['filterType'] == 'MIN_NOTIONAL':
                            return float(f['notional'])
            # 默认值
            return 5.0
        except:
            return 5.0

    def _get_symbol_info(self, symbol: str) -> Dict:
        """获取交易对信息"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    # 提取关键信息
                    info = {
                        'symbol': symbol,
                        'quantityPrecision': s.get('quantityPrecision', 3),
                        'pricePrecision': s.get('pricePrecision', 2),
                    }

                    # 从过滤器中提取限制
                    filters = s.get('filters', [])
                    for f in filters:
                        if f['filterType'] == 'LOT_SIZE':
                            info['minQty'] = f.get('minQty', '0.001')
                            info['maxQty'] = f.get('maxQty', '10000')
                            info['stepSize'] = f.get('stepSize', '0.001')
                        elif f['filterType'] == 'PRICE_FILTER':
                            info['minPrice'] = f.get('minPrice', '0.01')
                            info['maxPrice'] = f.get('maxPrice', '100000')
                            info['tickSize'] = f.get('tickSize', '0.01')
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            info['minNotional'] = f.get('notional', '5')

                    return info
            return {}
        except Exception as e:
            self.logger.error(f"获取交易对信息失败: {e}")
            return {}

    def _recalculate_params_with_slippage(self, params: Dict, current_price: float, action: str) -> Optional[Dict]:
        """考虑滑点重新计算交易参数"""
        # 更新入场价
        params['entry_price'] = current_price

        # 重新计算风险回报比
        if action == 'BUY':
            risk = current_price - params['stop_loss']
            reward = params['take_profit'] - current_price
        else:
            risk = params['stop_loss'] - current_price
            reward = current_price - params['take_profit']

        if risk <= 0 or reward <= 0:
            return None

        risk_reward_ratio = reward / risk

        # 如果风险回报比仍然可接受
        if risk_reward_ratio >= 1.3:  # 降低到1.3
            params['risk_reward_ratio'] = risk_reward_ratio
            params['quantity'] = params['position_value'] / current_price
            return params

        return None

    def run_trading_cycle(self):
        """
        主交易循环 - 使用整合分析
        """
        try:
            self.trade_cycle += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print_colored(f"\n{'=' * 80}", Colors.BLUE)
            print_colored(f"🚀 交易循环 #{self.trade_cycle} - {current_time}", Colors.BLUE + Colors.BOLD)
            print_colored(f"{'=' * 80}", Colors.BLUE)

            # 检查账户
            try:
                account_info = self.client.futures_account()
                account_balance = float(account_info['totalWalletBalance'])
                available_balance = float(account_info['availableBalance'])

                print_colored(f"\n💰 账户状态:", Colors.CYAN)
                print_colored(f"   总余额: ${account_balance:.2f}", Colors.INFO)
                print_colored(f"   可用余额: ${available_balance:.2f}", Colors.INFO)
                print_colored(f"   已用保证金: ${account_balance - available_balance:.2f}", Colors.INFO)

            except Exception as e:
                self.logger.error(f"获取账户信息失败: {e}")
                print_colored(f"❌ 获取账户信息失败: {e}", Colors.ERROR)
                return

            # 检查风险管理状态
            if self.risk_manager:
                can_trade, reason = self.risk_manager.can_open_position()
                if not can_trade:
                    print_colored(f"\n⚠️ 风险管理限制: {reason}", Colors.WARNING)
                    return

            # 更新现有持仓
            self._update_positions()

            # 检查持仓数量
            current_positions = len(self.open_positions)
            print_colored(f"\n📊 持仓管理:", Colors.CYAN)
            print_colored(f"   当前持仓: {current_positions}/{self.max_positions}", Colors.INFO)

            if current_positions > 0:
                total_pnl = 0
                for pos in self.open_positions:
                    # 获取当前价格
                    ticker = self.client.futures_symbol_ticker(symbol=pos['symbol'])
                    current_price = float(ticker['price'])

                    if pos['side'] == 'BUY':
                        pnl = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    else:
                        pnl = (pos['entry_price'] - current_price) / pos['entry_price'] * 100

                    total_pnl += pnl

                    pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                    print_colored(f"   • {pos['symbol']} {pos['side']}: {pnl_color}{pnl:+.2f}%{Colors.RESET}",
                                  Colors.INFO)

                avg_pnl = total_pnl / current_positions
                pnl_color = Colors.GREEN if avg_pnl > 0 else Colors.RED
                print_colored(f"   平均盈亏: {pnl_color}{avg_pnl:+.2f}%{Colors.RESET}", Colors.INFO)

            # 如果达到最大持仓，只管理现有持仓
            if current_positions >= self.max_positions:
                print_colored(f"\n⚠️ 已达到最大持仓数量，专注于管理现有持仓", Colors.WARNING)
                return

            # 运行整合分析
            print_colored(f"\n🔍 开始市场分析...", Colors.CYAN)
            self._run_integrated_analysis(account_balance)

            # 打印循环摘要
            self._print_cycle_summary()

        except Exception as e:
            self.logger.error(f"交易循环错误: {e}", exc_info=True)
            print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)

    def _run_traditional_analysis(self, account_balance: float):
        """
        运行传统技术分析 - 增强版，包含详细输出
        """
        print_colored("\n📊 运行传统技术分析...", Colors.CYAN)
        print_colored("=" * 60, Colors.BLUE)

        # 分析所有交易对
        candidates = []
        analyzed_count = 0

        # 使用线程池并行分析
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {}

            for symbol in self.config["TRADE_PAIRS"]:
                if not self.has_position(symbol):
                    analyzed_count += 1
                    print_colored(f"\n🔍 分析 {symbol} ({analyzed_count}/{len(self.config['TRADE_PAIRS'])})",
                                  Colors.BLUE)
                    future = executor.submit(self._analyze_symbol_traditional_enhanced, symbol)
                    future_to_symbol[future] = symbol
                else:
                    print_colored(f"⏭️ {symbol} - 已有持仓，跳过", Colors.GRAY)

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)

                        # 显示分析结果
                        score = result['score']
                        signal = result['signal']

                        signal_symbol = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⭕"
                        print_colored(f"  {signal_symbol} {symbol}: {signal} (评分: {score:.2f}/10)",
                                      Colors.GREEN if score >= 7 else Colors.YELLOW if score >= 5 else Colors.RED)

                        # 显示关键指标
                        metrics = result.get('metrics', {})
                        if metrics:
                            print_colored(f"    • RSI: {metrics.get('rsi', 'N/A'):.1f}", Colors.INFO)
                            print_colored(f"    • 趋势: {metrics.get('trend', 'N/A')}", Colors.INFO)

                except Exception as e:
                    self.logger.error(f"分析{symbol}失败: {e}")
                    print_colored(f"  ❌ 分析失败: {str(e)}", Colors.ERROR)

        print_colored("=" * 60, Colors.BLUE)

        # 按评分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 显示分析摘要
        print_colored(f"\n📊 分析摘要:", Colors.CYAN)
        print_colored(f"  • 分析交易对: {analyzed_count} 个", Colors.INFO)
        print_colored(f"  • 发现机会: {len(candidates)} 个", Colors.INFO)

        # 降低最低评分要求：从6.0降到5.0
        MIN_SCORE = 5.0
        qualified_candidates = [c for c in candidates if c['score'] >= MIN_SCORE]

        # 选择最佳交易机会
        if qualified_candidates:
            best_candidate = qualified_candidates[0]
            print_colored(f"\n🎯 最佳交易机会: {best_candidate['symbol']}", Colors.GREEN + Colors.BOLD)
            print_colored(f"  • 信号: {best_candidate['signal']}", Colors.INFO)
            print_colored(f"  • 评分: {best_candidate['score']:.2f}/10", Colors.INFO)
            print_colored(f"  • 当前价格: ${best_candidate['current_price']:.4f}", Colors.INFO)

            # 显示其他候选
            if len(qualified_candidates) > 1:
                print_colored(f"\n其他候选机会:", Colors.CYAN)
                for i, candidate in enumerate(qualified_candidates[1:4], 1):
                    print_colored(f"  {i}. {candidate['symbol']} - {candidate['signal']} "
                                  f"(评分: {candidate['score']:.2f})", Colors.INFO)

            # 执行交易
            print_colored(f"\n💫 准备执行交易...", Colors.CYAN)
            self._execute_traditional_trade(best_candidate, account_balance)
        else:
            print_colored(f"\n⚠️ 未找到评分高于 {MIN_SCORE:.1f} 的交易机会", Colors.WARNING)
            if candidates:
                best_low_score = candidates[0]
                print_colored(f"  • 最高评分: {best_low_score['symbol']} ({best_low_score['score']:.2f}/10)",
                              Colors.INFO)
                print_colored(f"  • 建议: 等待更好的市场机会", Colors.INFO)

    def _analyze_symbol_traditional_enhanced(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用传统技术指标分析单个交易对 - 增强版
        """
        try:
            # 获取历史数据
            df = get_historical_data(self.client, symbol, interval='15m', limit=100)
            if df is None or len(df) < 50:
                print_colored(f"    ⚠️ 数据不足", Colors.WARNING)
                return None

            # 计算技术指标
            df = calculate_optimized_indicators(df)

            # 获取当前价格和基础信息
            current_price = df['close'].iloc[-1]
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100)  # 1小时变化

            print_colored(f"    💰 价格: ${current_price:.4f} ({price_change:+.2f}%)",
                          Colors.GREEN if price_change > 0 else Colors.RED)

            # 计算质量评分
            quality_score, metrics = calculate_quality_score(
                df, self.client, symbol, None, self.config, self.logger
            )

            # 生成交易信号 - 使用更宽松的条件
            signal = "HOLD"
            confidence = 0.0

            # RSI策略
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                metrics['rsi'] = rsi

                if rsi < 35:  # 超卖（原来可能是30）
                    signal = "BUY"
                    confidence = (35 - rsi) / 35
                    print_colored(f"    📊 RSI超卖: {rsi:.1f}", Colors.GREEN)
                elif rsi > 65:  # 超买（原来可能是70）
                    signal = "SELL"
                    confidence = (rsi - 65) / 35
                    print_colored(f"    📊 RSI超买: {rsi:.1f}", Colors.RED)

            # MACD策略
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                macd_prev = df['MACD'].iloc[-2]
                macd_signal_prev = df['MACD_signal'].iloc[-2]

                # MACD金叉
                if macd > macd_signal and macd_prev <= macd_signal_prev:
                    if signal == "HOLD":
                        signal = "BUY"
                        confidence = 0.6
                    print_colored(f"    📊 MACD金叉", Colors.GREEN)
                # MACD死叉
                elif macd < macd_signal and macd_prev >= macd_signal_prev:
                    if signal == "HOLD":
                        signal = "SELL"
                        confidence = 0.6
                    print_colored(f"    📊 MACD死叉", Colors.RED)

            # 移动均线策略
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]

                if current_price > sma20 > sma50:
                    if signal == "HOLD":
                        signal = "BUY"
                        confidence = 0.5
                    print_colored(f"    📊 均线多头排列", Colors.GREEN)
                elif current_price < sma20 < sma50:
                    if signal == "HOLD":
                        signal = "SELL"
                        confidence = 0.5
                    print_colored(f"    📊 均线空头排列", Colors.RED)

            # 调整评分，使其更容易达到交易标准
            if signal != "HOLD":
                quality_score = quality_score * 1.2  # 给出20%的加成
                quality_score = min(quality_score, 10.0)  # 确保不超过10分

            if signal != "HOLD" or quality_score >= 5.0:  # 降低门槛
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'score': quality_score,
                    'confidence': confidence,
                    'metrics': metrics,
                    'current_price': current_price
                }

            return None

        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            print_colored(f"    ❌ 错误: {str(e)}", Colors.ERROR)
            return None

    def get_market_data_for_game_theory(self, symbol: str) -> tuple:
        """获取博弈论分析所需的数据"""
        try:
            # 获取K线数据
            df = self.get_market_data_sync(symbol)

            # 获取订单簿数据
            depth_data = self.get_order_book(symbol)

            return df, depth_data
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return None, None

    def analyze_symbol(self, symbol: str, account_balance: float) -> Dict[str, Any]:
        """
        分析单个交易对 - 完全同步版本

        参数:
            symbol: 交易对符号 (如 BTCUSDT)
            account_balance: 账户余额

        返回:
            包含分析结果和交易决策的字典
        """
        try:
            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"📊 综合分析 {symbol}", Colors.CYAN + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            # 记录开始时间
            start_time = time.time()

            # 1. 获取K线数据（只获取一次，后续都使用这个）
            print_colored(f"    📊 正在获取 {symbol} 的K线数据...", Colors.INFO)
            df = self.get_market_data_sync(symbol)

            if df is None or df.empty:
                print_colored(f"❌ {symbol} 数据获取失败，跳过分析", Colors.ERROR)
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'error': '数据获取失败',
                    'confidence': 0,
                    'final_score': 0,
                    'reason': 'K线数据获取失败'
                }

            print_colored(f"    ✅ 获取到 {len(df)} 条数据", Colors.SUCCESS)

            # 计算基础指标
            try:
                df = calculate_optimized_indicators(df)
                if df is None or df.empty:
                    print_colored("❌ 指标计算失败", Colors.ERROR)
                    return {
                        'symbol': symbol,
                        'action': 'HOLD',
                        'error': '指标计算失败',
                        'confidence': 0,
                        'final_score': 0
                    }
            except Exception as e:
                print_colored(f"❌ 指标计算错误: {e}", Colors.ERROR)

            # 2. 执行博弈论分析（市场微观结构）
            print_colored(f"\n🔍 深度分析 {symbol} 市场结构...", Colors.INFO)
            game_theory_analysis = {}

            try:
                # 获取订单簿数据
                depth_data = None
                try:
                    depth_data = self.client.futures_order_book(symbol=symbol, limit=500)
                except Exception as e:
                    print_colored(f"⚠️ 订单簿获取失败: {e}", Colors.WARNING)
                    depth_data = {'bids': [], 'asks': []}

                # 调用博弈论分析（传入所有必需参数）
                game_theory_analysis = self.enhanced_analyzer.analyze_market_intent(symbol, df, depth_data)

            except Exception as e:
                print_colored(f"❌ 博弈论分析失败: {e}", Colors.ERROR)
                game_theory_analysis = {
                    'whale_intent': 'NEUTRAL',
                    'confidence': 0.5,
                    'recommendation': 'HOLD',
                    'signals': [],
                    'market_phase': 'UNKNOWN',
                    'manipulation_type': 'NONE',
                    'error': str(e)
                }

            # 3. 执行技术分析（传入df参数）
            print_colored(f"\n📈 执行传统技术分析...", Colors.INFO)
            technical_analysis = {}

            try:
                technical_analysis = self._perform_technical_analysis(symbol, df)

                # 确保technical_analysis包含必要的字段
                if 'df' not in technical_analysis:
                    technical_analysis['df'] = df

            except Exception as e:
                print_colored(f"❌ 技术分析失败: {e}", Colors.ERROR)
                technical_analysis = {
                    'rsi': 50,
                    'macd': 0,
                    'macd_signal': 0,
                    'adx': 25,
                    'bb_position': 50.0,
                    'volume': {'ratio': 1.0, 'trend': 'NEUTRAL'},
                    'trend': {'direction': 'NEUTRAL', 'strength': 0},
                    'current_price': df['close'].iloc[-1] if not df.empty else 0,
                    'df': df,
                    'error': str(e)
                }

            # 4. 执行高级形态识别（如果可用）
            pattern_analysis = {}
            if hasattr(self, 'pattern_recognition') and self.pattern_recognition:
                print_colored(f"\n🎯 执行高级形态识别...", Colors.INFO)
                try:
                    pattern_analysis = self.pattern_recognition.detect_all_patterns(
                        df,
                        technical_analysis.get('current_price', 0)
                    )
                    print_colored(f"    • 检测到 {len(pattern_analysis.get('signals', []))} 个形态", Colors.INFO)
                except Exception as e:
                    print_colored(f"❌ 形态识别失败: {e}", Colors.ERROR)
                    pattern_analysis = {'signals': [], 'error': str(e)}

            # 5. 执行市场拍卖理论分析（如果可用）
            auction_analysis = {}
            if hasattr(self, 'market_auction') and self.market_auction:
                print_colored(f"\n🔨 执行市场拍卖理论分析...", Colors.INFO)
                try:
                    auction_analysis = self.market_auction.analyze_market_structure(df)
                    if 'key_levels' in auction_analysis:
                        print_colored(f"    • POC: ${auction_analysis['key_levels']['poc']:.4f}", Colors.INFO)
                        print_colored(f"    • 价值区域: ${auction_analysis['key_levels']['val']:.4f} - "
                                      f"${auction_analysis['key_levels']['vah']:.4f}", Colors.INFO)
                except Exception as e:
                    print_colored(f"❌ 拍卖理论分析失败: {e}", Colors.ERROR)
                    auction_analysis = {'error': str(e)}

            # 6. 整合所有分析结果
            print_colored(f"\n🔗 整合博弈论与技术分析...", Colors.INFO)

            # 确保有USE_TREND_PRIORITY配置
            use_trend_priority = self.config.get('USE_TREND_PRIORITY', True) if hasattr(self.config, 'get') else True

            if use_trend_priority:
                integrated_decision = self._integrate_analyses_trend_first(
                    game_theory_analysis,
                    technical_analysis,
                    symbol
                )
            else:
                integrated_decision = self._integrate_analyses(
                    game_theory_analysis,
                    technical_analysis,
                    symbol
                )

            # 添加额外的分析结果
            integrated_decision['pattern_analysis'] = pattern_analysis
            integrated_decision['auction_analysis'] = auction_analysis

            # 7. 计算风险调整后的交易参数
            if integrated_decision.get('action') != 'HOLD' and integrated_decision.get('action') is not None:
                print_colored(f"\n💡 计算风险调整参数...", Colors.INFO)

                try:
                    trade_params = self._calculate_risk_adjusted_params(
                        integrated_decision,
                        account_balance,
                        symbol
                    )

                    if trade_params:
                        integrated_decision['trade_params'] = trade_params

                        # 显示交易机会详情
                        self._display_trading_opportunity(integrated_decision)
                    else:
                        print_colored("⚠️ 风险参数计算失败，取消交易", Colors.WARNING)
                        integrated_decision['action'] = 'HOLD'
                        integrated_decision['reason'] = '风险参数计算失败'

                except Exception as e:
                    print_colored(f"❌ 风险参数计算错误: {e}", Colors.ERROR)
                    integrated_decision['action'] = 'HOLD'
                    integrated_decision['reason'] = f'风险参数计算错误: {str(e)}'
            else:
                print_colored(f"\n❌ 综合分析结果: 不建议交易", Colors.YELLOW)
                print_colored(f"   原因: {integrated_decision.get('reason', '信号不一致或风险过高')}", Colors.INFO)

            # 8. 记录分析时间
            analysis_time = time.time() - start_time
            integrated_decision['analysis_time'] = analysis_time
            print_colored(f"\n⏱️ 分析耗时: {analysis_time:.2f}秒", Colors.GRAY)

            # 更新计数器（如果有的话）
            if hasattr(self, 'analyzed_count'):
                self.analyzed_count += 1

            return integrated_decision

        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            print_colored(f"\n❌ 分析失败: {str(e)}", Colors.ERROR)

            # 打印详细错误信息帮助调试
            import traceback
            traceback.print_exc()

            return {
                'symbol': symbol,
                'action': 'HOLD',
                'error': str(e),
                'confidence': 0,
                'final_score': 0,
                'reason': f'分析失败: {str(e)}'
            }

    def analyze_symbols(self, symbols: List[str], account_balance: float) -> List[Dict]:
        """
        分析多个交易对 - 完全同步版本

        参数:
            symbols: 交易对列表
            account_balance: 账户余额

        返回:
            包含所有交易机会的列表
        """
        trading_opportunities = []

        # 初始化计数器
        if not hasattr(self, 'analyzed_count'):
            self.analyzed_count = 0

        self.symbols_to_scan = symbols  # 保存总数用于显示进度

        for i, symbol in enumerate(symbols, 1):
            try:
                print_colored(f"\n{'=' * 60}", Colors.BLUE)
                print_colored(f"📊 综合分析 {symbol} ({i}/{len(symbols)})", Colors.CYAN + Colors.BOLD)
                print_colored(f"{'=' * 60}", Colors.BLUE)

                # 直接调用analyze_symbol（不使用await）
                integrated_decision = self.analyze_symbol(symbol, account_balance)

                # 检查是否有交易机会
                if integrated_decision.get('action') != 'HOLD' and 'error' not in integrated_decision:
                    if integrated_decision.get('trade_params'):
                        trading_opportunities.append(integrated_decision)

                        # 显示找到的交易机会
                        action = integrated_decision['action']
                        confidence = integrated_decision.get('confidence', 0)
                        score = integrated_decision.get('final_score', 0)

                        action_color = Colors.GREEN if action == 'BUY' else Colors.RED
                        print_colored(
                            f"\n🎯 找到交易机会: {action_color}{action}{Colors.RESET} "
                            f"(置信度: {confidence * 100:.1f}%, 评分: {score:.2f})",
                            Colors.YELLOW + Colors.BOLD
                        )

                # 可选：添加短暂延迟，避免API限流
                if i < len(symbols):
                    time.sleep(0.5)  # 500ms延迟

            except Exception as e:
                self.logger.error(f"分析{symbol}失败: {e}")
                print_colored(f"\n❌ 分析失败: {str(e)}", Colors.ERROR)
                continue

        # 打印总结
        print_colored(f"\n{'=' * 60}", Colors.BLUE)
        print_colored(f"📊 分析完成总结", Colors.CYAN + Colors.BOLD)
        print_colored(f"{'=' * 60}", Colors.BLUE)
        print_colored(f"✅ 分析了 {len(symbols)} 个交易对", Colors.INFO)
        print_colored(f"🎯 找到 {len(trading_opportunities)} 个交易机会", Colors.INFO)

        # 如果找到交易机会，显示摘要
        if trading_opportunities:
            print_colored(f"\n📋 交易机会摘要:", Colors.YELLOW + Colors.BOLD)
            for i, opp in enumerate(trading_opportunities, 1):
                action_color = Colors.GREEN if opp['action'] == 'BUY' else Colors.RED
                print_colored(
                    f"  {i}. {opp['symbol']} - {action_color}{opp['action']}{Colors.RESET} "
                    f"(置信度: {opp.get('confidence', 0) * 100:.1f}%, "
                    f"评分: {opp.get('final_score', 0):.2f})",
                    Colors.INFO
                )
        else:
            print_colored(f"\n⚠️ 没有找到合适的交易机会", Colors.YELLOW)

        return trading_opportunities

    def analyze_symbol_sync(self, symbol: str, account_balance: float) -> Dict[str, Any]:
        """
        分析单个交易对 - 同步版本

        Args:
            symbol: 交易对符号 (如 BTCUSDT)
            account_balance: 账户余额

        Returns:
            包含分析结果和交易决策的字典
        """
        try:
            print_colored(f"\n{'=' * 60}", Colors.BLUE)
            print_colored(f"📊 综合分析 {symbol} ({self.analyzed_count + 1}/{len(self.symbols_to_scan)})",
                          Colors.CYAN + Colors.BOLD)
            print_colored(f"{'=' * 60}", Colors.BLUE)

            # 记录开始时间
            start_time = time.time()

            # 1. 执行博弈论分析（市场微观结构）
            print_colored(f"\n🔍 深度分析 {symbol} 市场结构...", Colors.INFO)
            df = self.get_market_data_sync(symbol)
            if df is None or df.empty:
                print_colored(f"❌ {symbol} K线数据获取失败", Colors.ERROR)
                game_theory_analysis = {
                    'whale_intent': 'NEUTRAL',
                    'confidence': 0.5,
                    'recommendation': 'HOLD',
                    'signals': []
                }
            else:
                # 获取订单簿
                try:
                    depth_data = self.client.futures_order_book(symbol=symbol, limit=500)
                except:
                    depth_data = {'bids': [], 'asks': []}

                # 现在传入所有必需的参数
                game_theory_analysis = self.enhanced_analyzer.analyze_market_intent(symbol, df, depth_data)

            # 2. 执行技术分析
            print_colored(f"\n📈 执行传统技术分析...", Colors.INFO)
            technical_analysis = self._perform_technical_analysis(symbol)

            # 3. 执行高级形态识别（如果可用）
            pattern_analysis = {}
            if hasattr(self, 'pattern_recognition') and self.pattern_recognition:
                print_colored(f"\n🎯 执行高级形态识别...", Colors.INFO)
                df = technical_analysis.get('df', None)
                if df is not None:
                    pattern_analysis = self.pattern_recognition.detect_all_patterns(
                        df,
                        technical_analysis.get('current_price', 0)
                    )
                    print_colored(f"    • 检测到 {len(pattern_analysis.get('signals', []))} 个形态", Colors.INFO)

            # 4. 执行市场拍卖理论分析（如果可用）
            auction_analysis = {}
            if hasattr(self, 'market_auction') and self.market_auction:
                print_colored(f"\n🔨 执行市场拍卖理论分析...", Colors.INFO)
                df = technical_analysis.get('df', None)
                if df is not None:
                    auction_analysis = self.market_auction.analyze_market_structure(df)
                    if 'key_levels' in auction_analysis:
                        print_colored(f"    • POC: ${auction_analysis['key_levels']['poc']:.4f}", Colors.INFO)
                        print_colored(f"    • 价值区域: ${auction_analysis['key_levels']['val']:.4f} - "
                                      f"${auction_analysis['key_levels']['vah']:.4f}", Colors.INFO)

            # 5. 整合所有分析结果
            print_colored(f"\n🔗 整合博弈论与技术分析...", Colors.INFO)

            # 如果有增强评分系统，使用它
            if hasattr(self, 'scoring_system') and self.scoring_system:
                # 构建完整的分析数据
                analysis_data = {
                    'symbol': symbol,
                    'current_price': technical_analysis.get('current_price', 0),
                    'technical_indicators': {
                        'RSI': technical_analysis.get('rsi', 50),
                        'MACD': technical_analysis.get('macd', 0),
                        'MACD_signal': technical_analysis.get('macd_signal', 0),
                        'ADX': technical_analysis.get('adx', 25),
                        'volume_ratio': technical_analysis.get('volume', {}).get('ratio', 1.0),
                        'bb_position': technical_analysis.get('bb_position', 50),
                        'trend_direction': technical_analysis.get('trend', {}).get('direction', 'NEUTRAL')
                    },
                    'trend': technical_analysis.get('trend', {}),
                    'classical_patterns': pattern_analysis.get('classical', []),
                    'game_theory_patterns': pattern_analysis.get('game_theory', []),
                    'market_auction': auction_analysis.get('key_levels', {})
                }

                # 计算综合得分
                score_report = self.scoring_system.calculate_comprehensive_score(analysis_data)

                # 创建增强的决策
                integrated_decision = {
                    'symbol': symbol,
                    'action': score_report['action'],
                    'confidence': score_report['confidence'],
                    'final_score': abs(score_report['final_score']),
                    'signals': [],
                    'risks': [],
                    'reason': f"增强系统: {score_report['market_regime']}市场",
                    'strategy_type': 'ENHANCED_SYSTEM',
                    'score_report': score_report
                }

                # 添加主要信号
                for pattern in pattern_analysis.get('signals', [])[:3]:
                    integrated_decision['signals'].append(
                        f"{pattern['type']} ({pattern.get('direction', 'NEUTRAL')})"
                    )

            else:
                # 使用传统整合方法
                if self.config.get('USE_TREND_PRIORITY', True):
                    integrated_decision = self._integrate_analyses_trend_first(
                        game_theory_analysis,
                        technical_analysis,
                        symbol
                    )
                else:
                    integrated_decision = self._integrate_analyses(
                        game_theory_analysis,
                        technical_analysis,
                        symbol
                    )

            # 6. 计算风险调整后的交易参数
            if integrated_decision['action'] != 'HOLD':
                print_colored(f"\n💡 计算风险调整参数...", Colors.INFO)
                trade_params = self._calculate_risk_adjusted_params(
                    integrated_decision,
                    account_balance,
                    symbol
                )

                if trade_params:
                    integrated_decision['trade_params'] = trade_params

                    # 显示交易机会详情
                    self._display_trading_opportunity(integrated_decision)

                    # 记录分析时间
                    analysis_time = time.time() - start_time
                    integrated_decision['analysis_time'] = analysis_time

                    return integrated_decision
            else:
                print_colored(f"\n❌ 综合分析结果: 不建议交易", Colors.YELLOW)
                print_colored(f"   原因: {integrated_decision.get('reason', '信号不一致或风险过高')}", Colors.INFO)

            # 记录分析时间
            analysis_time = time.time() - start_time
            print_colored(f"\n⏱️ 分析耗时: {analysis_time:.2f}秒", Colors.GRAY)

            # 更新计数器
            self.analyzed_count += 1

            return integrated_decision

        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            print_colored(f"\n❌ 分析失败: {str(e)}", Colors.ERROR)
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'error': str(e),
                'confidence': 0,
                'final_score': 0
            }

    def _analyze_symbol_traditional(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用传统技术指标分析单个交易对
        """
        try:
            # 获取历史数据
            df = get_historical_data(self.client, symbol, interval='15m', limit=100)
            if df is None or len(df) < 50:
                return None

            # 计算技术指标
            df = calculate_optimized_indicators(df)

            # 计算质量评分
            quality_score, metrics = calculate_quality_score(
                df, self.client, symbol, None, self.config, self.logger
            )

            # 生成交易信号
            signal, confidence = self.generate_trade_signal(df, symbol)

            if signal != "HOLD":
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'score': quality_score,
                    'confidence': confidence,
                    'metrics': metrics,
                    'current_price': df['close'].iloc[-1]
                }

            return None

        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            return None

    def _run_game_theory_analysis(self, account_balance: float):
        """
        运行博弈论分析（异步）- 增强版，包含详细输出
        """
        print_colored("\n🎮 运行博弈论分析...", Colors.CYAN)
        print_colored("=" * 60, Colors.BLUE)

        # 收集所有交易对的博弈论信号
        trade_signals = []
        analyzed_count = 0

        for symbol in self.config["TRADE_PAIRS"]:
            if self._has_position(symbol):
                print_colored(f"⏭️ {symbol} - 已有持仓，跳过分析", Colors.GRAY)
                continue

            analyzed_count += 1
            print_colored(f"\n📊 分析 {symbol} ({analyzed_count}/{len(self.config['TRADE_PAIRS'])})", Colors.BLUE)

            try:
                # 收集市场数据
                print_colored(f"  📡 收集市场数据...", Colors.INFO)
                market_data = self.data_collector.collect_full_market_data(symbol)

                if market_data and 'error' not in market_data:
                    # 获取基础价格信息
                    df = market_data.get('price_data')
                    if df is not None and len(df) > 0:
                        current_price = df['close'].iloc[-1]
                        price_change_24h = ((df['close'].iloc[-1] - df['close'].iloc[-96]) / df['close'].iloc[
                            -96] * 100) if len(df) > 96 else 0

                        print_colored(f"  💰 当前价格: ${current_price:.4f}", Colors.INFO)
                        print_colored(f"  📈 24小时变化: {price_change_24h:+.2f}%",
                                      Colors.GREEN if price_change_24h > 0 else Colors.RED)

                    # 运行博弈论分析
                    print_colored(f"  🧮 执行博弈论分析...", Colors.INFO)
                    analysis = self.game_analyzer.analyze_market_structure(market_data)

                    # 显示分析结果
                    if analysis:
                        # 操纵概率
                        manip_prob = analysis.get('manipulation_probability', 0)
                        print_colored(f"  🎯 操纵概率: {manip_prob:.1%}",
                                      Colors.RED if manip_prob > 0.7 else Colors.YELLOW if manip_prob > 0.4 else Colors.GREEN)

                        # 市场结构
                        market_structure = analysis.get('market_structure', {})
                        trend = market_structure.get('trend', 'NEUTRAL')
                        trend_strength = market_structure.get('strength', 0)
                        print_colored(f"  📊 市场结构: {trend} (强度: {trend_strength:.2f})", Colors.INFO)

                        # 订单流分析
                        order_flow = analysis.get('order_flow', {})
                        if order_flow:
                            toxicity = order_flow.get('toxicity', 0)
                            print_colored(f"  ☠️ 订单流毒性: {toxicity:.2f}",
                                          Colors.RED if toxicity > 0.35 else Colors.YELLOW if toxicity > 0.2 else Colors.GREEN)

                        # 多空动态
                        ls_dynamics = analysis.get('long_short_dynamics', {})
                        if ls_dynamics:
                            smart_retail_div = ls_dynamics.get('smart_retail_divergence', 0)
                            print_colored(f"  🧠 聪明钱vs散户分歧: {smart_retail_div:.2f}", Colors.INFO)

                    # 获取交易决策
                    print_colored(f"  🤔 生成交易决策...", Colors.INFO)
                    decision = self.decision_engine.make_trading_decision(market_data)

                    # 显示决策结果
                    action = decision.get('action', 'HOLD')
                    confidence = decision.get('confidence', 0)

                    # 使用彩色显示决策
                    if action == 'BUY':
                        action_color = Colors.GREEN
                        action_symbol = "🟢"
                    elif action == 'SELL':
                        action_color = Colors.RED
                        action_symbol = "🔴"
                    else:
                        action_color = Colors.GRAY
                        action_symbol = "⭕"

                    print_colored(f"  {action_symbol} 决策: {action} (置信度: {confidence:.1%})", action_color)

                    # 显示决策理由
                    if 'reasoning' in decision and decision['reasoning']:
                        print_colored(f"  📝 理由:", Colors.INFO)
                        for reason in decision['reasoning'][:3]:  # 只显示前3个理由
                            print_colored(f"     • {reason}", Colors.INFO)

                    # 降低门槛：原来是0.6，现在改为0.4
                    MIN_CONFIDENCE = 0.4

                    if action != 'HOLD' and confidence >= MIN_CONFIDENCE:
                        # 计算建议的交易参数
                        if action == 'BUY':
                            suggested_stop = current_price * 0.98  # 2%止损
                            suggested_target = current_price * 1.04  # 4%止盈
                        else:
                            suggested_stop = current_price * 1.02
                            suggested_target = current_price * 0.96

                        print_colored(f"  ✅ 满足交易条件!", Colors.GREEN + Colors.BOLD)
                        print_colored(f"     建议止损: ${suggested_stop:.4f}", Colors.INFO)
                        print_colored(f"     建议止盈: ${suggested_target:.4f}", Colors.INFO)

                        trade_signals.append({
                            'symbol': symbol,
                            'decision': decision,
                            'analysis': analysis,
                            'market_data': market_data,
                            'current_price': current_price
                        })
                    else:
                        # 解释为什么不交易
                        if action == 'HOLD':
                            print_colored(f"  ❌ 不交易: 市场信号不明确", Colors.YELLOW)
                        elif confidence < MIN_CONFIDENCE:
                            print_colored(f"  ❌ 不交易: 置信度不足 ({confidence:.1%} < {MIN_CONFIDENCE:.1%})",
                                          Colors.YELLOW)
                            # 显示需要什么条件才会交易
                            needed_confidence = MIN_CONFIDENCE - confidence
                            print_colored(f"     需要额外 {needed_confidence:.1%} 的置信度", Colors.INFO)

            except Exception as e:
                self.logger.error(f"博弈论分析{symbol}失败: {e}")
                print_colored(f"  ❌ 分析失败: {str(e)}", Colors.ERROR)

                # 如果是因为缺少方法，尝试使用简化分析
                if "has no attribute" in str(e):
                    print_colored(f"  🔄 尝试简化分析...", Colors.YELLOW)
                    # 这里可以调用传统分析方法作为后备

            # 添加分隔线
            if analyzed_count < len(self.config['TRADE_PAIRS']):
                print_colored("-" * 60, Colors.GRAY)

        print_colored("=" * 60, Colors.BLUE)

        # 按置信度排序
        trade_signals.sort(key=lambda x: x['decision']['confidence'], reverse=True)

        # 显示分析摘要
        print_colored(f"\n📊 分析摘要:", Colors.CYAN)
        print_colored(f"  • 分析交易对: {analyzed_count} 个", Colors.INFO)
        print_colored(f"  • 发现机会: {len(trade_signals)} 个", Colors.INFO)

        # 执行最佳交易
        if trade_signals:
            best_signal = trade_signals[0]
            print_colored(f"\n🎯 最佳交易机会: {best_signal['symbol']}", Colors.GREEN + Colors.BOLD)
            print_colored(f"  • 方向: {best_signal['decision']['action']}", Colors.INFO)
            print_colored(f"  • 置信度: {best_signal['decision']['confidence']:.1%}", Colors.INFO)
            print_colored(f"  • 当前价格: ${best_signal['current_price']:.4f}", Colors.INFO)

            # 显示其他候选
            if len(trade_signals) > 1:
                print_colored(f"\n其他候选机会:", Colors.CYAN)
                for i, signal in enumerate(trade_signals[1:4], 1):  # 显示前3个候选
                    print_colored(f"  {i}. {signal['symbol']} - {signal['decision']['action']} "
                                  f"(置信度: {signal['decision']['confidence']:.1%})", Colors.INFO)

            # 确认是否执行交易
            print_colored(f"\n💫 准备执行交易...", Colors.CYAN)
            self._execute_game_theory_trade(best_signal, account_balance)
        else:
            print_colored(f"\n⚠️ 未找到满足条件的交易机会", Colors.WARNING)
            print_colored(f"  • 最低置信度要求: 40%", Colors.INFO)
            print_colored(f"  • 建议: 等待更明确的市场信号", Colors.INFO)

    def _execute_traditional_trade(self, candidate: Dict[str, Any], account_balance: float):
        """
        执行传统模式交易
        """
        try:
            symbol = candidate['symbol']
            signal = candidate['signal']
            current_price = candidate['current_price']

            # 计算交易参数
            trade_params = self._calculate_trade_parameters(
                symbol=symbol,
                signal=signal,
                current_price=current_price,
                account_balance=account_balance,
                quality_score=candidate['score']
            )

            if not trade_params:
                return

            # 执行交易
            self._place_order(trade_params)

        except Exception as e:
            self.logger.error(f"执行传统交易失败: {e}")
            print_colored(f"❌ 执行交易失败: {e}", Colors.ERROR)

    def _execute_game_theory_trade(self, signal_data: Dict[str, Any], account_balance: float):
        """
        执行博弈论模式交易
        """
        try:
            symbol = signal_data['symbol']
            decision = signal_data['decision']
            analysis = signal_data['analysis']

            # 使用博弈论风险管理
            market_analysis = {
                'manipulation_score': analysis.get('manipulation_probability', 0),
                'order_flow_toxicity': analysis.get('order_flow', {}).get('toxicity', 0),
                'smart_money_divergence': analysis.get('long_short_dynamics', {}).get('smart_retail_divergence', 0)
            }

            # 计算仓位
            position_params = self.risk_manager.calculate_position_size(
                account_balance=account_balance,
                entry_price=decision['entry_price'],
                stop_loss=decision['stop_loss'],
                market_analysis=market_analysis
            )

            # 准备交易参数
            trade_params = {
                'symbol': symbol,
                'side': 'BUY' if decision['action'] == 'BUY' else 'SELL',
                'quantity': position_params['position_size'],
                'entry_price': decision['entry_price'],
                'stop_loss': decision['stop_loss'],
                'take_profit': decision['take_profit'],
                'confidence': decision['confidence'],
                'analysis': analysis
            }

            # 执行交易
            self._place_order(trade_params)

            # 记录到性能监控
            self.performance_monitor.record_trade_open({
                'symbol': symbol,
                'side': trade_params['side'],
                'price': trade_params['entry_price'],
                'quantity': trade_params['quantity'],
                'market_analysis': market_analysis,
                'strategy_tags': ['game_theory'] + analysis.get('signals', [])
            })

        except Exception as e:
            self.logger.error(f"执行博弈论交易失败: {e}")
            print_colored(f"❌ 执行博弈论交易失败: {e}", Colors.ERROR)

    def _calculate_trade_parameters(self, symbol: str, signal: str, current_price: float,
                                    account_balance: float, quality_score: float) -> Optional[Dict[str, Any]]:
        """
        计算交易参数
        """
        try:
            # 获取交易规则
            symbol_info = self.client.futures_exchange_info()['symbols']
            symbol_rules = next((s for s in symbol_info if s['symbol'] == symbol), None)

            if not symbol_rules:
                return None

            # 计算仓位大小
            position_value = account_balance * (self.config['ORDER_AMOUNT_PERCENT'] / 100)
            quantity = position_value / current_price

            # 调整精度
            step_size = float(next(f['stepSize'] for f in symbol_rules['filters'] if f['filterType'] == 'LOT_SIZE'))
            quantity = self._round_quantity(quantity, step_size)

            # 计算止损止盈
            if signal == 'BUY':
                stop_loss = current_price * (1 - self.config['STOP_LOSS_PERCENT'] / 100)
                take_profit = current_price * (1 + self.config['TAKE_PROFIT_PERCENT'] / 100)
            else:
                stop_loss = current_price * (1 + self.config['STOP_LOSS_PERCENT'] / 100)
                take_profit = current_price * (1 - self.config['TAKE_PROFIT_PERCENT'] / 100)

            return {
                'symbol': symbol,
                'side': signal,
                'quantity': quantity,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quality_score': quality_score
            }

        except Exception as e:
            self.logger.error(f"计算交易参数失败: {e}")
            return None

    def _place_order(self, trade_params: Dict[str, Any]):
        """
        下单执行交易
        """
        try:
            symbol = trade_params['symbol']
            side = trade_params['side']
            quantity = trade_params['quantity']

            # 下市价单
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            if order['status'] == 'FILLED':
                # 记录持仓
                position = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': float(order['avgPrice']),
                    'stop_loss': trade_params['stop_loss'],
                    'take_profit': trade_params['take_profit'],
                    'open_time': time.time(),
                    'order_id': order['orderId']
                }

                self.open_positions.append(position)

                print_colored(f"✅ 交易执行成功: {symbol} {side} @ {position['entry_price']}", Colors.GREEN)
                self.logger.info(f"交易执行成功", extra=position)

                # 设置止损止盈单
                self._set_stop_orders(position)

            else:
                print_colored(f"❌ 订单未成交: {order['status']}", Colors.ERROR)

        except Exception as e:
            self.logger.error(f"下单失败: {e}")
            print_colored(f"❌ 下单失败: {e}", Colors.ERROR)

    def _update_positions(self):
        """
        更新所有持仓状态
        """
        if not self.open_positions:
            return

        positions_to_close = []

        for position in self.open_positions:
            try:
                symbol = position['symbol']

                # 获取当前价格
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # 更新性能监控
                if hasattr(self, 'performance_monitor'):
                    self.performance_monitor.update_position(
                        position.get('order_id', f"{symbol}_{position['open_time']}"),
                        current_price
                    )

                # 计算盈亏
                if position['side'] == 'BUY':
                    pnl = (current_price - position['entry_price']) / position['entry_price']
                    should_close = current_price >= position['take_profit'] or current_price <= position['stop_loss']
                else:
                    pnl = (position['entry_price'] - current_price) / position['entry_price']
                    should_close = current_price <= position['take_profit'] or current_price >= position['stop_loss']

                # 检查是否需要平仓
                if should_close:
                    positions_to_close.append((position, current_price, pnl))

            except Exception as e:
                self.logger.error(f"更新持仓{position['symbol']}失败: {e}")

        # 处理需要平仓的持仓
        for position, exit_price, pnl in positions_to_close:
            self._close_position(position, exit_price, 'target_reached' if pnl > 0 else 'stop_loss')

    def close_position(self, symbol: str, position_side: str) -> Tuple[bool, Dict]:
        """平仓"""
        try:
            # 查找持仓
            position = None
            for pos in self.open_positions:
                if pos['symbol'] == symbol and pos.get('position_side', 'LONG') == position_side:
                    position = pos
                    break

            if not position:
                print_colored(f"⚠️ 未找到 {symbol} {position_side} 持仓", Colors.WARNING)
                return False, {}

            # 确定平仓方向
            side = 'SELL' if position_side == 'LONG' else 'BUY'

            # 下平仓单
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=position['quantity'],
                reduceOnly=True
            )

            if order:
                print_colored(f"✅ {symbol} {position_side} 平仓成功", Colors.GREEN)

                # 记录到历史
                self.position_history.append({
                    'symbol': symbol,
                    'side': position['side'],
                    'position_side': position_side,
                    'entry_price': position['entry_price'],
                    'exit_price': float(order.get('avgPrice', position['mark_price'])),
                    'quantity': position['quantity'],
                    'pnl': position.get('unrealized_pnl', 0),
                    'open_time': position['open_time'],
                    'close_time': time.time(),
                    'reason': 'manual_close'
                })

                # 保存历史
                self._save_position_history()

                return True, order
            else:
                return False, {}

        except Exception as e:
            self.logger.error(f"平仓失败 {symbol}: {e}")
            print_colored(f"❌ 平仓失败: {e}", Colors.ERROR)
            return False, {}

    def has_position(self, symbol: str) -> bool:
        """检查是否已有该交易对的持仓"""
        for pos in self.open_positions:
            if pos['symbol'] == symbol:
                return True
        return False

    def record_new_position(self, symbol: str, side: str, position_side: str,
                            entry_price: float, quantity: float,
                            initial_stop_loss: float = -0.02, entry_atr: float = 0):
        """记录新持仓"""
        position_data = {
            'symbol': symbol,
            'side': side,
            'position_side': position_side,
            'quantity': quantity,
            'entry_price': entry_price,
            'open_time': time.time(),
            'current_stop_level': entry_price * (1 + initial_stop_loss) if position_side == 'LONG' else entry_price * (
                        1 - initial_stop_loss),
            'trailing_active': False,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'entry_atr': entry_atr,
            'initial_stop_loss': initial_stop_loss,
            'trailing_activation': 0.01,  # 1%激活
            'trailing_distance': 0.008,  # 0.8%跟踪
            'expected_profit': 0.025  # 期望2.5%利润
        }

        self.open_positions.append(position_data)
        self.logger.info(f"记录新持仓: {symbol} {position_side}", extra=position_data)

    def _round_quantity(self, quantity: float, step_size: float) -> float:
        """
        按交易所规则调整数量精度
        向上取整以确保满足最小金额要求
        """
        import math
        precision = len(str(step_size).split('.')[-1])

        # 向上取整，确保不会因为精度问题导致金额不足
        rounded_down = round(quantity - (quantity % step_size), precision)
        rounded_up = rounded_down + step_size

        # 返回向上取整的结果
        return rounded_up


    def _print_cycle_summary(self):
        """
        打印交易循环摘要
        """
        print_colored(f"\n{'=' * 60}", Colors.BLUE)
        print_colored(f"📊 循环 #{self.trade_cycle} 完成", Colors.BLUE)
        print_colored(f"当前持仓: {len(self.open_positions)}", Colors.INFO)

        if hasattr(self, 'risk_manager'):
            risk_summary = self.risk_manager.get_risk_summary()
            print_colored(f"日内亏损: {risk_summary['daily_loss']:.2f}%", Colors.INFO)
            print_colored(f"风险状态: {risk_summary['risk_status']}", Colors.INFO)

        if hasattr(self, 'performance_monitor'):
            perf_stats = self.performance_monitor.get_current_stats()
            basic_stats = perf_stats['basic_stats']
            print_colored(f"总交易: {basic_stats['total_trades']}, 胜率: {basic_stats['win_rate'] * 100:.1f}%",
                          Colors.INFO)

        print_colored(f"{'=' * 60}\n", Colors.BLUE)

    def get_futures_balance(self) -> float:
        """
        获取期货账户余额
        """
        try:
            account = self.client.futures_account()
            return float(account['totalWalletBalance'])
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return 0.0

    def _log_game_theory_analysis(self, symbol, market_data, decision, game_analysis):
        """记录博弈论分析详情"""

        # 提取关键数据
        ls_ratio = market_data.get('long_short_ratio', {})
        funding_rate = market_data.get('funding_rate', 0)
        toxicity = game_analysis.get('order_flow_toxicity', {})
        smart_money = game_analysis.get('smart_money_flow', {})
        manipulation = game_analysis.get('manipulation_detection', {})

        log_message = f"""
        ==================== {symbol} 博弈论分析 ====================
        时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        【市场数据】
        - 当前价格: {market_data['kline_data']['close'].iloc[-1] if market_data.get('kline_data') is not None else 'N/A'}
        - 资金费率: {funding_rate:.4f}
        - 持仓量: {market_data.get('open_interest', 'N/A')}

        【多空博弈】
        - 顶级交易员多空比: {ls_ratio.get('top_traders', {}).get('ratio', 'N/A')}
        - 散户多空比: {ls_ratio.get('global', {}).get('ratio', 'N/A')} 
        - 主动买卖比: {ls_ratio.get('takers', {}).get('ratio', 'N/A')}

        【订单流分析】
        - 订单流毒性: {toxicity.get('toxicity_level', 'N/A')} (VPIN: {toxicity.get('vpin', 0):.3f})
        - 聪明钱方向: {smart_money.get('smart_money_direction', 'N/A')}
        - 资金流向: {smart_money.get('net_flow', 0):.2f}

        【市场操纵检测】
        - 操纵评分: {manipulation.get('total_manipulation_score', 0):.2f}
        - 最可能类型: {manipulation.get('most_likely', 'N/A')}

        【决策结果】
        - 操作: {decision['action']}
        - 置信度: {decision['confidence']:.2f}
        - 综合评分: {game_analysis.get('comprehensive_score', 0):.2f}

        【推理依据】
        {chr(10).join(['- ' + r for r in decision.get('reasoning', [])])}
        =========================================================
        """

        self.logger.info(log_message)

        # 同时打印简要信息到控制台
        if decision['action'] != 'HOLD':
            color = Colors.GREEN if decision['action'] == 'BUY' else Colors.RED
            print_colored(f"""
            {symbol} 信号:
            - 操作: {decision['action']}
            - 置信度: {decision['confidence']:.2f}
            - 毒性: {toxicity.get('toxicity_level', 'N/A')}
            - 聪明钱: {smart_money.get('smart_money_direction', 'N/A')}
            """, color)

    def _calculate_game_theory_score(self, game_analysis):
        """计算博弈论综合评分"""
        score = 5.0  # 基础分

        # 根据各项分析调整分数
        # 订单流毒性
        toxicity = game_analysis.get('order_flow_toxicity', {}).get('toxicity_level', 'MEDIUM')
        if toxicity == 'LOW':
            score += 1.0
        elif toxicity == 'HIGH':
            score -= 1.0

        # 聪明钱方向
        smart_money = game_analysis.get('smart_money_flow', {}).get('conviction_level', 'LOW')
        if smart_money == 'HIGH':
            score += 1.5
        elif smart_money == 'MEDIUM':
            score += 0.5

        # 市场操纵
        manipulation_score = game_analysis.get('manipulation_detection', {}).get('total_manipulation_score', 0)
        score -= manipulation_score * 2  # 操纵会降低分数

        # 套利机会
        if game_analysis.get('arbitrage_opportunities', {}).get('best_opportunity'):
            score += 1.0

        return max(0, min(10, score))


    def _round_step_size(self, quantity, step_size):
        """按步长调整数量精度"""
        precision = len(str(step_size).split('.')[-1])
        return round(quantity - (quantity % step_size), precision)

    def _round_price(self, price, tick_size):
        """按价格精度调整"""
        return round(price - (price % tick_size), len(str(tick_size).split('.')[-1]))

    def place_market_order(self, symbol, side, quantity):
        """下市价单"""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            return order
        except Exception as e:
            self.logger.error(f"下单失败 {symbol}: {e}")
            return None

    def _set_stop_orders(self, position):
        """设置止损止盈订单"""
        try:
            symbol = position['symbol']
            quantity = position['quantity']

            # 止损单
            if position['side'] == 'LONG':
                stop_side = 'SELL'
                stop_price = position['stop_loss']
                take_side = 'SELL'
                take_price = position['take_profit']
            else:
                stop_side = 'BUY'
                stop_price = position['stop_loss']
                take_side = 'BUY'
                take_price = position['take_profit']

            # 下止损单
            stop_order = self.client.futures_create_order(
                symbol=symbol,
                side=stop_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price
            )

            # 下止盈单
            take_order = self.client.futures_create_order(
                symbol=symbol,
                side=take_side,
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=take_price
            )

            print_colored(f"✅ 止损止盈订单设置成功", Colors.GREEN)

        except Exception as e:
            self.logger.error(f"设置止损止盈失败: {e}")
            print_colored(f"❌ 设置止损止盈失败: {e}", Colors.ERROR)

    def _run_game_theory_cycle(self, account_balance):
        """博弈论交易循环"""
        print_colored("🎮 使用博弈论分析模式", Colors.CYAN)

        # 收集所有交易对的信号
        trade_candidates = []

        for symbol in self.config["TRADE_PAIRS"]:
            try:
                print_colored(f"\n{'=' * 30} 分析 {symbol} {'=' * 30}", Colors.BLUE)

                # 1. 收集综合市场数据
                market_data = self.data_collector.collect_comprehensive_data(symbol)

                # 缓存数据
                self.market_data_cache[symbol] = market_data

                # 更新订单簿历史
                if market_data.get('order_book'):
                    if symbol not in self.order_book_history:
                        self.order_book_history[symbol] = []
                    self.order_book_history[symbol].append(market_data['order_book'])
                    # 保留最近100个快照
                    if len(self.order_book_history[symbol]) > 100:
                        self.order_book_history[symbol].pop(0)

                # 2. 执行完整的博弈论分析
                game_analysis = self._perform_complete_game_analysis(symbol, market_data)

                # 3. 运行决策引擎
                decision = self.decision_engine.make_trading_decision(market_data)

                # 4. 记录分析结果
                self._log_game_theory_analysis(symbol, market_data, decision, game_analysis)

                # 5. 如果有交易信号且置信度足够
                min_confidence = self.config.get("MIN_GAME_THEORY_CONFIDENCE", 0.5)
                if decision['action'] != 'HOLD' and decision['confidence'] >= min_confidence:
                    # 检查是否已有该交易对的持仓
                    if not self.has_position(symbol):
                        trade_candidates.append({
                            'symbol': symbol,
                            'decision': decision,
                            'market_data': market_data,
                            'game_analysis': game_analysis,
                            'priority': decision['confidence']
                        })
                        print_colored(
                            f"✅ {symbol} 生成交易信号: {decision['action']} (置信度: {decision['confidence']:.2f})",
                            Colors.GREEN)
                    else:
                        print_colored(f"⚠️ {symbol} 已有持仓，跳过新信号", Colors.YELLOW)
                else:
                    print_colored(f"❌ {symbol} 无有效信号或置信度不足", Colors.GRAY)

            except Exception as e:
                self.logger.error(f"分析{symbol}失败: {e}", exc_info=True)
                print_colored(f"❌ 分析{symbol}失败: {e}", Colors.ERROR)
                continue

        # 执行交易
        if trade_candidates:
            # 按优先级排序
            trade_candidates.sort(key=lambda x: x['priority'], reverse=True)
            print_colored(f"\n📊 共找到 {len(trade_candidates)} 个交易机会", Colors.CYAN)

            # 执行最高优先级的交易
            max_concurrent = self.config.get("MAX_CONCURRENT_TRADES", 3)
            for i, candidate in enumerate(trade_candidates[:max_concurrent]):
                print_colored(f"\n执行交易 {i + 1}/{min(len(trade_candidates), max_concurrent)}", Colors.BLUE)
                self._execute_game_theory_trade(candidate, account_balance)
        else:
            print_colored("\n❌ 未发现合适的交易机会", Colors.YELLOW)

    def _perform_complete_game_analysis(self, symbol, market_data):
        """执行完整的博弈论分析"""

        game_analysis = {
            'auction_analysis': {},
            'order_flow_toxicity': {},
            'smart_money_flow': {},
            'arbitrage_opportunities': {},
            'manipulation_detection': {},
            'comprehensive_score': 0
        }

        try:
            # 1. 拍卖理论分析
            if self.order_book_history.get(symbol):
                recent_trades = market_data.get('recent_trades', {}).get('large_trades', [])
                auction_result = self.auction_analyzer.analyze_price_discovery_mechanism(
                    self.order_book_history[symbol][-50:],  # 最近50个订单簿快照
                    recent_trades
                )
                game_analysis['auction_analysis'] = auction_result

                # 检测拍卖操纵
                manipulation = self.auction_manipulator.detect_manipulation_patterns(
                    self.order_book_history[symbol][-20:],
                    recent_trades
                )
                game_analysis['manipulation_detection'] = manipulation

            # 2. 订单流毒性分析
            if market_data.get('kline_data') is not None:
                vpin_result = self.toxicity_analyzer.calculate_vpin(
                    market_data['kline_data'],
                    bucket_size=50
                )
                game_analysis['order_flow_toxicity'] = vpin_result

                # 分析交易信息含量
                if market_data.get('recent_trades'):
                    trade_info = self.toxicity_analyzer.analyze_trade_informativeness(
                        market_data['recent_trades'].get('large_trades', [])
                    )
                    game_analysis['order_flow_toxicity']['trade_informativeness'] = trade_info

            # 3. 聪明钱流向分析
            smart_money = self.smart_money_tracker.track_smart_money_flow(
                market_data.get('kline_data'),
                self.order_book_history.get(symbol, [])
            )
            game_analysis['smart_money_flow'] = smart_money

            # 4. 订单流分析（结合多空比）
            if market_data.get('order_book') and market_data.get('long_short_ratio'):
                order_flow = self.order_flow_analyzer.analyze_order_flow_with_ls_ratio(
                    market_data['order_book'],
                    market_data['long_short_ratio'],
                    market_data.get('recent_trades')
                )
                game_analysis['order_flow_analysis'] = order_flow

            # 5. 套利机会检测
            arbitrage = self.arbitrage_detector.detect_arbitrage_opportunities(
                market_data.get('kline_data'),
                market_data.get('order_book'),
                market_data.get('funding_rate')
            )
            game_analysis['arbitrage_opportunities'] = arbitrage

            # 6. 计算综合评分
            game_analysis['comprehensive_score'] = self._calculate_game_theory_score(game_analysis)

        except Exception as e:
            self.logger.error(f"博弈分析错误 {symbol}: {e}")

        return game_analysis

    def _execute_game_theory_trade(self, candidate, account_balance):
        """执行博弈论交易"""
        symbol = candidate['symbol']
        decision = candidate['decision']
        game_analysis = candidate.get('game_analysis', {})

        try:
            # 检查是否已有该交易对的持仓
            if self.has_position(symbol):
                print_colored(f"{symbol} 已有持仓，跳过", Colors.WARNING)
                return

            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 获取交易精度
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

            if not symbol_info:
                print_colored(f"无法获取{symbol}交易信息", Colors.ERROR)
                return

            # 计算交易参数
            trade_params = self._calculate_game_theory_trade_params(
                decision,
                current_price,
                account_balance,
                symbol_info,
                game_analysis
            )

            # 打印交易详情
            print_colored(f"""
            📈 准备执行交易:
            交易对: {symbol}
            方向: {decision['action']}
            当前价格: {current_price:.4f}
            数量: {trade_params['quantity']}
            止损: {trade_params['stop_loss']:.4f}
            止盈: {trade_params['take_profit']:.4f}
            原因: {', '.join(decision['reasoning'][:2])}
            """, Colors.CYAN)

            # 执行交易
            order_result = None

            if decision['action'] == 'BUY':
                order_result = self.place_market_order(
                    symbol=symbol,
                    side='BUY',
                    quantity=trade_params['quantity']
                )

                if order_result and order_result.get('status') == 'FILLED':
                    # 记录持仓
                    position = {
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_price': float(order_result.get('avgPrice', current_price)),
                        'quantity': trade_params['quantity'],
                        'stop_loss': trade_params['stop_loss'],
                        'take_profit': trade_params['take_profit'],
                        'entry_time': datetime.now(),
                        'reason': decision['reasoning'],
                        'confidence': decision['confidence'],
                        'game_analysis': game_analysis
                    }
                    self.open_positions.append(position)
                    self.daily_trades += 1

                    print_colored(f"✅ 做多订单成功: {symbol} @ {position['entry_price']:.4f}", Colors.GREEN)

            elif decision['action'] == 'SELL':
                order_result = self.place_market_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=trade_params['quantity']
                )

                if order_result and order_result.get('status') == 'FILLED':
                    # 记录持仓
                    position = {
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_price': float(order_result.get('avgPrice', current_price)),
                        'quantity': trade_params['quantity'],
                        'stop_loss': trade_params['stop_loss'],
                        'take_profit': trade_params['take_profit'],
                        'entry_time': datetime.now(),
                        'reason': decision['reasoning'],
                        'confidence': decision['confidence'],
                        'game_analysis': game_analysis
                    }
                    self.open_positions.append(position)
                    self.daily_trades += 1

                    print_colored(f"✅ 做空订单成功: {symbol} @ {position['entry_price']:.4f}", Colors.RED)

            # 设置止损止盈订单
            if order_result and order_result.get('status') == 'FILLED':
                self._set_stop_orders(position)

                # 记录交易日志
                self.logger.info(f"""
                博弈论交易执行成功:
                交易对: {symbol}
                方向: {decision['action']}
                入场价: {position['entry_price']}
                数量: {trade_params['quantity']}
                置信度: {decision['confidence']:.2f}
                """)

        except BinanceAPIException as e:
            self.logger.error(f"币安API错误 {symbol}: {e}")
            print_colored(f"❌ 交易执行失败 (API): {e}", Colors.ERROR)
        except Exception as e:
            self.logger.error(f"执行博弈论交易失败 {symbol}: {e}", exc_info=True)
            print_colored(f"❌ 交易执行失败: {e}", Colors.ERROR)

    def _run_traditional_cycle(self, account_balance):
        """传统交易循环 - 保留您原有的逻辑"""
        print_colored("📊 使用传统技术指标模式", Colors.CYAN)

        # 获取最佳交易候选
        candidates = []

        for symbol in self.config["TRADE_PAIRS"]:
            try:
                # 获取历史数据
                df = get_historical_data(self.client, symbol)
                if df is None or df.empty:
                    continue

                # 计算指标
                df = calculate_optimized_indicators(df)

                # 计算市场评分
                score = score_market(df)

                # 生成信号（使用您原有的逻辑）
                signal = self.generate_trade_signal(df)

                if signal and score >= self.min_score:
                    candidates.append({
                        'symbol': symbol,
                        'signal': signal,
                        'score': score
                    })

            except Exception as e:
                self.logger.error(f"处理{symbol}时出错: {e}")
                continue

        # 按评分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 执行交易
        for candidate in candidates[:3]:  # 最多同时交易3个
            if self.has_position(candidate['symbol']):
                continue

            # 使用您原有的下单逻辑
            self.execute_trade(
                candidate['symbol'],
                candidate['signal'],
                account_balance
            )

    def _calculate_game_theory_trade_params(self, decision, current_price, account_balance, symbol_info, game_analysis):
        """计算博弈论交易参数"""

        # 获取交易规则
        filters = {f['filterType']: f for f in symbol_info['filters']}
        lot_size_filter = filters.get('LOT_SIZE', {})
        price_filter = filters.get('PRICE_FILTER', {})

        # 获取精度
        step_size = float(lot_size_filter.get('stepSize', 0.001))
        min_qty = float(lot_size_filter.get('minQty', 0.001))
        tick_size = float(price_filter.get('tickSize', 0.01))

        # 基础交易金额
        base_amount = account_balance * self.config.get("ORDER_AMOUNT_PERCENT", 5) / 100

        # 根据置信度调整仓位
        confidence_multiplier = decision['confidence']

        # 根据博弈分析调整仓位
        game_multiplier = 1.0

        # 如果检测到操纵，减小仓位
        if game_analysis.get('manipulation_detection', {}).get('total_manipulation_score', 0) > 0.7:
            game_multiplier *= 0.5
            print_colored("⚠️ 检测到市场操纵，减小仓位", Colors.YELLOW)

        # 如果订单流毒性高，减小仓位
        toxicity = game_analysis.get('order_flow_toxicity', {}).get('toxicity_level', 'LOW')
        if toxicity == 'HIGH':
            game_multiplier *= 0.6
            print_colored("⚠️ 订单流毒性高，减小仓位", Colors.YELLOW)
        elif toxicity == 'MEDIUM':
            game_multiplier *= 0.8

        # 如果有套利机会，增加仓位
        if game_analysis.get('arbitrage_opportunities', {}).get('best_opportunity'):
            game_multiplier *= 1.2
            print_colored("✅ 发现套利机会，增加仓位", Colors.GREEN)

        # 聪明钱方向确认，增加仓位
        smart_money = game_analysis.get('smart_money_flow', {}).get('smart_money_direction', 'NEUTRAL')
        if (decision['action'] == 'BUY' and 'ACCUMULATING' in smart_money) or \
                (decision['action'] == 'SELL' and 'DISTRIBUTING' in smart_money):
            game_multiplier *= 1.15
            print_colored("✅ 聪明钱方向一致，增加仓位", Colors.GREEN)

        # 最终交易金额
        trade_amount = base_amount * confidence_multiplier * game_multiplier

        # 限制最大交易金额
        max_trade_amount = account_balance * 0.2  # 单笔最大20%
        trade_amount = min(trade_amount, max_trade_amount)

        # 计算数量
        quantity = trade_amount / current_price

        # 调整到交易精度
        quantity = self._round_step_size(quantity, step_size)
        quantity = max(quantity, min_qty)

        # 止损止盈设置
        if decision['action'] == 'BUY':
            # 根据市场环境动态调整止损
            if game_analysis.get('order_flow_analysis', {}).get('stop_hunt_zones'):
                # 如果有止损猎杀区域，设置更宽的止损
                stop_loss_pct = 0.025  # 2.5%
                print_colored("⚠️ 检测到止损猎杀区域，使用更宽止损", Colors.YELLOW)
            else:
                stop_loss_pct = self.config['GAME_THEORY_CONFIG'].get('TIGHT_STOP_LOSS', 0.015)

            stop_loss = self._round_price(current_price * (1 - stop_loss_pct), tick_size)
            take_profit = self._round_price(current_price * (1 + self.config.get("TAKE_PROFIT_PERCENT", 3) / 100),
                                            tick_size)
        else:
            if game_analysis.get('order_flow_analysis', {}).get('stop_hunt_zones'):
                stop_loss_pct = 0.025
                print_colored("⚠️ 检测到止损猎杀区域，使用更宽止损", Colors.YELLOW)
            else:
                stop_loss_pct = self.config['GAME_THEORY_CONFIG'].get('TIGHT_STOP_LOSS', 0.015)

            stop_loss = self._round_price(current_price * (1 + stop_loss_pct), tick_size)
            take_profit = self._round_price(current_price * (1 - self.config.get("TAKE_PROFIT_PERCENT", 3) / 100),
                                            tick_size)

        return {
            'quantity': quantity,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trade_amount': trade_amount
        }

    def manage_open_positions(self):
        """增强版持仓管理 - 集成智能移动止盈和ATR动态止损"""

        # 收集所有持仓的市场数据
        if self.open_positions and hasattr(self, 'position_visualizer') and self.position_visualizer:
            market_data = {}
            for position in self.open_positions:
                symbol = position['symbol']
                try:
                    # 获取实时价格
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])

                    # 获取历史数据
                    df = self.get_historical_data(symbol)

                    market_data[symbol] = {
                        'current_price': current_price,
                        'df': df
                    }
                except Exception as e:
                    self.logger.error(f"获取{symbol}市场数据失败: {e}")

            # 显示可视化仪表板
            self.position_visualizer.display_position_dashboard(
                self.open_positions,
                market_data
            )

        self.load_existing_positions()

        if not self.open_positions:
            self.logger.info("当前无持仓")
            return

        current_time = time.time()
        positions_to_remove = []

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos.get("position_side", "LONG")
            entry_price = pos["entry_price"]

            try:
                # 获取当前价格
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # 获取最新市场数据（强制刷新）
                df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                if df is None:
                    continue

                # 计算所需指标（包括RVI）
                df = self.calculate_simplified_indicators(df)

                # 计算当前盈亏
                if position_side == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - current_price) / entry_price * 100

                # 1. 检查RVI出场信号（如果启用）
                if self.config.get('USE_RVI_FILTER', True) and 'RVI' in df.columns:
                    from rvi_indicator import rvi_exit_signal
                    rvi_exit, rvi_reason = rvi_exit_signal(df, position_side, profit_pct)
                    if rvi_exit and profit_pct > 0.5:  # 只在有利润时考虑RVI出场
                        print_colored(f"📊 {symbol} RVI出场信号: {rvi_reason}", Colors.YELLOW)
                        if profit_pct > 2.0:  # 利润超过2%直接出场
                            success, closed = self.close_position(symbol, position_side)
                            if success:
                                print_colored(f"✅ {symbol} 根据RVI信号平仓，利润: {profit_pct:.2f}%", Colors.GREEN)
                                positions_to_remove.append(pos)
                                continue

                # 2. 获取各项分析数据
                market_analysis = self.analyze_market_environment(df)
                game_analysis = self.get_game_theory_analysis(symbol)
                technical_analysis = self.get_technical_analysis(df, position_side)

                # 准备市场数据包
                market_data = {
                    'current_price': current_price,
                    'game_analysis': game_analysis,
                    'technical_analysis': technical_analysis,
                    'market_analysis': market_analysis,
                    'df': df
                }

                # 3. 应用ATR动态止损更新
                if self.config.get('USE_ATR_STOP_LOSS', True) and 'ATR' in df.columns:
                    current_atr = df['ATR'].iloc[-1]
                    atr_result = self.atr_stop_loss.update_stop_loss_dynamically(
                        pos, current_atr, current_price, market_analysis
                    )

                    if atr_result['should_adjust'] and atr_result['new_stop_loss']:
                        # 确保新止损只会对持仓有利
                        old_stop = pos.get('current_stop_level', 0)
                        new_stop = atr_result['new_stop_loss']

                        if position_side == 'LONG':
                            # 多头：新止损必须高于旧止损
                            if new_stop > old_stop:
                                pos['current_stop_level'] = new_stop
                                print_colored(f"📈 {symbol} ATR止损上移: {old_stop:.6f} → {new_stop:.6f}", Colors.GREEN)
                        else:  # SHORT
                            # 空头：新止损必须低于旧止损
                            if old_stop == 0 or new_stop < old_stop:
                                pos['current_stop_level'] = new_stop
                                print_colored(f"📉 {symbol} ATR止损下移: {old_stop:.6f} → {new_stop:.6f}", Colors.GREEN)

                # 4. 应用智能移动止盈（只在盈利超过1%时激活）
                if profit_pct >= 1.0:
                    trailing_result = self.smart_trailing_stop.apply_trailing_stop(pos, market_data)

                    # 检查是否触发止损
                    if trailing_result['should_close']:
                        print_colored(f"🔔 {symbol} {position_side} {trailing_result['close_reason']}", Colors.YELLOW)
                        success, closed = self.close_position(symbol, position_side)
                        if success:
                            positions_to_remove.append(pos)
                            continue

                    # 更新止损位（确保只向有利方向移动）
                    if trailing_result['trailing_info']['should_update']:
                        new_stop = trailing_result['trailing_info']['new_stop_level']
                        old_stop = pos.get('current_stop_level', 0)

                        if position_side == 'LONG':
                            if new_stop > old_stop:
                                pos['current_stop_level'] = new_stop
                                pos['trailing_active'] = True
                                pos['highest_price'] = current_price
                                print_colored(f"🚀 {symbol} 移动止盈激活/更新: {new_stop:.6f}", Colors.CYAN)
                        else:  # SHORT
                            if old_stop == 0 or new_stop < old_stop:
                                pos['current_stop_level'] = new_stop
                                pos['trailing_active'] = True
                                pos['lowest_price'] = current_price
                                print_colored(f"🚀 {symbol} 移动止盈激活/更新: {new_stop:.6f}", Colors.CYAN)

                            # 检查是否需要更新止损
                            if hasattr(self, 'liquidity_stop_loss') and self.liquidity_stop_loss:
                                stop_update = self.liquidity_stop_loss.update_position_stop_loss(
                                    position, current_price, {'df': df}
                                )

                                if stop_update['should_update']:
                                    # 更新持仓信息
                                    position['current_stop_level'] = stop_update['new_stop_level']
                                    position['trailing_active'] = True
                                    position['last_stop_update'] = datetime.now()

                                    # 如果是流动性调整，记录原因
                                    if stop_update.get('liquidity_adjusted'):
                                        position['stop_adjustment_reason'] = stop_update['adjustment_reason']

                # 5. 检查是否触发止损
                current_stop = pos.get('current_stop_level', 0)
                if current_stop > 0:
                    if (position_side == 'LONG' and current_price <= current_stop) or \
                            (position_side == 'SHORT' and current_price >= current_stop):
                        stop_type = '移动' if pos.get('trailing_active', False) else 'ATR'
                        print_colored(
                            f"⚠️ {symbol} 触发{stop_type}止损: 价格{current_price:.6f} vs 止损{current_stop:.6f}",
                            Colors.YELLOW
                        )
                        success, closed = self.close_position(symbol, position_side)
                        if success:
                            print_colored(f"✅ {symbol} 止损平仓成功，盈亏: {profit_pct:.2f}%", Colors.GREEN)
                            positions_to_remove.append(pos)
                            continue

                # 6. 显示持仓状态
                self.display_position_status(pos, current_price, df)

            except Exception as e:
                self.logger.error(f"管理{symbol}持仓时出错: {e}")
                print_colored(f"❌ 管理{symbol}持仓时出错: {e}", Colors.ERROR)

        # 移除已平仓的持仓
        for pos in positions_to_remove:
            self.open_positions.remove(pos)

        # 定期清理缓存
        self.cleanup_cache_if_needed()

    def get_game_theory_analysis(self, symbol: str) -> Dict:
        """获取博弈论分析数据"""

        try:
            # 这里应该调用您现有的博弈论分析
            # 简化版本，返回必要的结构
            return {
                'manipulation_detection': {
                    'total_manipulation_score': 0.3
                },
                'order_flow_toxicity': {
                    'toxicity_level': 'LOW'
                },
                'smart_money_flow': {
                    'smart_money_direction': 'NEUTRAL'
                },
                'position_side': 'LONG'  # 从实际持仓获取
            }
        except Exception as e:
            self.logger.error(f"获取博弈论分析失败: {e}")
            return {}

    def display_position_status(self, pos: Dict, current_price: float, df: pd.DataFrame):
        """显示持仓状态"""

        symbol = pos['symbol']
        position_side = pos.get('position_side', 'LONG')
        entry_price = pos['entry_price']

        # 计算盈亏
        if position_side == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # 获取止损信息
        current_stop = pos.get('current_stop_level', 0)
        trailing_active = pos.get('trailing_active', False)

        # 显示状态
        profit_color = Colors.GREEN if profit_pct > 0 else Colors.RED
        print_colored(
            f"{symbol} {position_side}: "
            f"盈亏 {profit_color}{profit_pct:+.2f}%{Colors.RESET} | "
            f"{'移动' if trailing_active else 'ATR'}止损: {current_stop:.6f}",
            Colors.INFO
        )

        # 显示ATR信息
        if 'ATR' in df.columns:
            current_atr = df['ATR'].iloc[-1]
            entry_atr = pos.get('entry_atr', current_atr)
            atr_change = (current_atr - entry_atr) / entry_atr * 100 if entry_atr > 0 else 0

            print_colored(
                f"  ATR: {current_atr:.6f} (变化: {atr_change:+.1f}%)",
                Colors.GRAY
            )

        # 显示RVI信息
        if 'RVI' in df.columns:
            rvi_value = df['RVI'].iloc[-1]
            rvi_signal = df['RVI_Signal'].iloc[-1] if 'RVI_Signal' in df.columns else 0
            print_colored(
                f"  RVI: {rvi_value:.3f} / 信号: {rvi_signal:.3f}",
                Colors.GRAY
            )

    def cleanup_cache_if_needed(self):
        """定期清理缓存"""

        current_time = time.time()

        if current_time - self.last_cache_cleanup > self.cache_cleanup_interval:
            # 清理过期的历史数据缓存
            expired_keys = []
            for key, cache_item in self.historical_data_cache.items():
                if current_time - cache_item['timestamp'] > self.cache_ttl * 2:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.historical_data_cache[key]

            if expired_keys:
                print_colored(f"🧹 清理了{len(expired_keys)}个过期缓存项", Colors.INFO)
                self.logger.info(f"缓存清理完成", extra={"cleaned_items": len(expired_keys)})

            # 更新清理时间
            self.last_cache_cleanup = current_time

            # 运行垃圾回收
            import gc
            gc.collect()

    def analyze_market_environment(self, df: pd.DataFrame) -> Dict:
        """分析市场环境"""

        environment = {
            'environment': 'unknown',
            'volatility_level': 'NORMAL',
            'trend_strength': 0,
            'current_price': df['close'].iloc[-1]
        }

        try:
            # ATR波动率分析
            if 'ATR' in df.columns:
                recent_atr = df['ATR'].iloc[-1]
                avg_atr = df['ATR'].iloc[-20:].mean()

                if recent_atr > avg_atr * 1.5:
                    environment['volatility_level'] = 'HIGH'
                elif recent_atr > avg_atr * 2:
                    environment['volatility_level'] = 'EXTREME'
                elif recent_atr < avg_atr * 0.7:
                    environment['volatility_level'] = 'LOW'

            # 趋势分析
            if 'EMA20' in df.columns and 'EMA50' in df.columns:
                ema20 = df['EMA20'].iloc[-1]
                ema50 = df['EMA50'].iloc[-1]
                price = df['close'].iloc[-1]

                if price > ema20 > ema50:
                    environment['environment'] = 'trending'
                    environment['trend_strength'] = 0.8
                elif price < ema20 < ema50:
                    environment['environment'] = 'trending'
                    environment['trend_strength'] = -0.8
                else:
                    environment['environment'] = 'ranging'

            # 突破检测
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                price = df['close'].iloc[-1]

                if price > bb_upper or price < bb_lower:
                    environment['environment'] = 'breakout'

        except Exception as e:
            self.logger.error(f"分析市场环境时出错: {e}")

        return environment

    def get_technical_analysis(self, df: pd.DataFrame, position_side: str) -> Dict:
        """获取技术分析数据"""

        current_price = df['close'].iloc[-1]

        analysis = {
            'current_price': current_price,
            'position_side': position_side,
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
            'atr_ratio': df['ATR'].iloc[-1] / current_price if 'ATR' in df.columns else 0.01,
            'trend_strength': 0,
            'macd_signal': 'NEUTRAL',
            'volume_surge': False
        }

        # 计算趋势强度
        if 'EMA20' in df.columns and 'EMA50' in df.columns:
            ema20 = df['EMA20'].iloc[-1]
            ema50 = df['EMA50'].iloc[-1]

            if current_price > ema20 > ema50:
                analysis['trend_strength'] = 0.8
            elif current_price < ema20 < ema50:
                analysis['trend_strength'] = -0.8
            else:
                analysis['trend_strength'] = 0.3

        # MACD信号
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_signal'].iloc[-1]

            if macd > signal:
                analysis['macd_signal'] = 'BULLISH'
            elif macd < signal:
                analysis['macd_signal'] = 'BEARISH'

        # 成交量激增检测
        if 'volume_ratio' in df.columns:
            analysis['volume_surge'] = df['volume_ratio'].iloc[-1] > 1.5

        return analysis

    def calculate_simplified_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算简化后的核心指标"""

        try:
            # 1. 趋势指标
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['EMA50'] = df['close'].ewm(span=50).mean()

            # 2. ATR
            df['H-L'] = df['high'] - df['low']
            df['H-PC'] = abs(df['high'] - df['close'].shift(1))
            df['L-PC'] = abs(df['low'] - df['close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # 3. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 4. MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

            # 5. 成交量
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # 6. 布林带
            df['BB_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

            # 7. RVI（如果启用）
            if self.config.get('USE_RVI_FILTER', True):
                from rvi_indicator import calculate_rvi
                df = calculate_rvi(df, period=10)

            # 8. FVG检测（保留）
            from fvg_module import detect_fair_value_gap
            fvg_data = detect_fair_value_gap(df)
            df['has_fvg'] = len(fvg_data) > 0

            return df

        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return df

    def calculate_expected_profit(self, symbol, side, current_price, df=None):
        """
        计算预期收益百分比，用于开仓决策

        参数:
            symbol: 交易对符号
            side: 交易方向 (BUY 或 SELL)
            current_price: 当前价格
            df: 可选的数据帧，如未提供则获取

        返回:
            预期收益百分比，无法计算则返回0
        """
        try:
            # 如果未提供数据，则获取
            if df is None:
                df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                if df is None:
                    return 0.0
                df = calculate_optimized_indicators(df)
                if df is None or df.empty:
                    return 0.0

            # 获取当前的市场状态和趋势
            trend_data = get_smc_trend_and_duration(df, None, self.logger)[2]

            # 从 risk_management 导入计算函数
            from risk_management import (
                calculate_max_movement_range,
                analyze_volatility_pattern,
                analyze_market_stage,
                estimate_support_resistance_range,
                estimate_structure_move
            )

            # 创建模拟持仓对象，用于计算最大波动区间
            position = {
                "position_side": "LONG" if side == "BUY" else "SHORT",
                "entry_price": current_price,
                "initial_stop_loss": 0.008  # 默认初始止损0.8%
            }

            # 计算波动区间收敛值
            volatility_pattern = analyze_volatility_pattern(df)
            market_stage = analyze_market_stage(df)

            # 1. 波动率方法估算最大盈利空间
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.005
            atr_ratio = atr / current_price  # ATR占价格的比例

            # 基于ATR估算合理的最大波动幅度 (根据市场阶段调整)
            if market_stage == "trending":
                max_atr_multiplier = 5.0
            elif market_stage == "ranging":
                max_atr_multiplier = 3.0
            else:  # transitioning
                max_atr_multiplier = 4.0

            max_expected_move_atr = atr_ratio * max_atr_multiplier

            # 2. 基于支撑/阻力位估算最大波动
            support_resist_move = estimate_support_resistance_range(df, position["position_side"], current_price)

            # 3. 基于价格结构估算最大波动
            structure_move = estimate_structure_move(df, position["position_side"], current_price, volatility_pattern)

            # 4. 加权合并三种方法的结果
            weights = {
                'atr': 0.4,
                'sr': 0.3,
                'structure': 0.3
            }

            max_expected_move = (
                    weights['atr'] * max_expected_move_atr +
                    weights['sr'] * support_resist_move +
                    weights['structure'] * structure_move
            )

            # 检查预期收益是否大于1%
            print_colored(
                f"{symbol} {side} 预期收益计算 - "
                f"ATR法: {max_expected_move_atr:.2%}, "
                f"支撑阻力法: {support_resist_move:.2%}, "
                f"结构法: {structure_move:.2%}",
                Colors.INFO
            )

            print_colored(
                f"{symbol} {side} 最终预期收益: {max_expected_move:.2%}, "
                f"波动模式: {volatility_pattern}, 市场阶段: {market_stage}",
                Colors.GREEN if max_expected_move >= 0.01 else Colors.YELLOW
            )

            return max_expected_move

        except Exception as e:
            print_colored(f"计算预期收益失败: {e}", Colors.ERROR)
            return 0.0

    def record_entry_reason(self, symbol, side, entry_price, expected_profit):
        """记录开仓原因和预期收益"""
        timestamp = time.time()
        entry_record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "expected_profit": expected_profit,
            "timestamp": timestamp,
            "date": datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存到文件
        try:
            if not hasattr(self, 'entry_records'):
                self.entry_records = []
                # 尝试从文件加载
                if os.path.exists("entry_records.json"):
                    with open("entry_records.json", "r") as f:
                        self.entry_records = json.load(f)

            self.entry_records.append(entry_record)

            # 保存到文件
            with open("entry_records.json", "w") as f:
                json.dump(self.entry_records, f, indent=4)

            print_colored(f"✅ 已记录 {symbol} {side} 开仓原因，预期收益: {expected_profit:.2%}", Colors.GREEN)

        except Exception as e:
            print_colored(f"❌ 记录开仓原因失败: {e}", Colors.ERROR)

    def calculate_dynamic_order_amount(self, risk, account_balance):
        """基于风险和账户余额计算适当的订单金额"""
        # 基础订单百分比 - 默认账户的5%
        base_pct = 5.0

        # 根据风险调整订单百分比
        if risk > 0.05:  # 高风险
            adjusted_pct = base_pct * 0.6  # 减小到基础的60%
        elif risk > 0.03:  # 中等风险
            adjusted_pct = base_pct * 0.8  # 减小到基础的80%
        elif risk < 0.01:  # 低风险
            adjusted_pct = base_pct * 1.2  # 增加到基础的120%
        else:
            adjusted_pct = base_pct

        # 计算订单金额
        order_amount = account_balance * (adjusted_pct / 100)

        # 确保订单金额在合理范围内
        min_amount = 5.0  # 最小5 USDC
        max_amount = account_balance * 0.1  # 最大为账户10%

        order_amount = max(min_amount, min(order_amount, max_amount))

        print_colored(f"动态订单金额: {order_amount:.2f} USDC ({adjusted_pct:.1f}% 账户余额)", Colors.INFO)

        return order_amount

    def check_and_reconnect_api(self):
        """检查API连接并在必要时重新连接"""
        try:
            # 简单测试API连接
            self.client.ping()
            print("✅ API连接检查: 连接正常")
            return True
        except Exception as e:
            print(f"⚠️ API连接检查失败: {e}")
            self.logger.warning(f"API连接失败，尝试重新连接", extra={"error": str(e)})

            # 重试计数
            retry_count = 3
            reconnected = False

            for attempt in range(retry_count):
                try:
                    print(f"🔄 尝试重新连接API (尝试 {attempt + 1}/{retry_count})...")
                    # 重新创建客户端
                    self.client = Client(self.api_key, self.api_secret)

                    # 验证连接
                    self.client.ping()

                    print("✅ API重新连接成功")
                    self.logger.info("API重新连接成功")
                    reconnected = True
                    break
                except Exception as reconnect_error:
                    print(f"❌ 第{attempt + 1}次重连失败: {reconnect_error}")
                    time.sleep(5 * (attempt + 1))  # 指数退避

            if not reconnected:
                print("❌ 所有重连尝试失败，将在下一个周期重试")
                self.logger.error("API重连失败", extra={"attempts": retry_count})
                return False

            return reconnected

    def active_position_monitor(self, check_interval=15):
        """
        主动监控持仓，使用改进的跟踪止损策略和最优波动区间止盈
        - 修复止损位只上移不下移和重复激活的问题
        """
        print(f"🔄 启动主动持仓监控（每{check_interval}秒检查一次）")

        try:
            while True:
                # 如果没有持仓，等待一段时间后再检查
                if not self.open_positions:
                    time.sleep(check_interval)
                    continue

                # 加载最新持仓
                self.load_existing_positions()

                # 当前持仓列表的副本，用于检查
                positions = self.open_positions.copy()

                for pos in positions:
                    symbol = pos["symbol"]
                    position_side = pos.get("position_side", "LONG")
                    entry_price = pos["entry_price"]

                    # 获取当前价格
                    try:
                        ticker = self.client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])
                    except Exception as e:
                        print(f"⚠️ 获取{symbol}价格失败: {e}")
                        continue

                    # 获取历史数据用于反转检测和最优止盈检查
                    df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                    if df is not None:
                        df = calculate_optimized_indicators(df)

                        # 最优波动区间止盈检查
                        try:
                            from risk_management import calculate_optimal_take_profit

                            tp_data = calculate_optimal_take_profit(pos, df, current_price)

                            # 计算当前盈亏
                            if position_side == "LONG":
                                current_pnl = (current_price - entry_price) / entry_price
                            else:
                                current_pnl = (entry_price - current_price) / entry_price

                            # 检查是否达到最优止盈条件
                            optimal_tp = tp_data["current_optimal_tp"]
                            completion_pct = tp_data["current_completion_pct"]

                            # 基于完成度和最优止盈点执行止盈
                            take_profit = False
                            reason = ""

                            # 条件1: 完成80%以上的预期行程且达到最优止盈点
                            if completion_pct >= 0.8 and current_pnl >= optimal_tp.get("pnl_pct", 0):
                                take_profit = True
                                reason = f"达到80%以上完成度({completion_pct:.2%})并达到最优止盈点(级别{optimal_tp.get('level', 1)})"

                            # 条件2: 完成90%以上的预期行程
                            elif completion_pct >= 0.9:
                                take_profit = True
                                reason = f"达到90%以上完成度({completion_pct:.2%})"

                            # 条件3: 达到较高风险回报比的止盈点
                            elif optimal_tp.get("risk_reward", 0) >= 3.0 and current_pnl >= optimal_tp.get("pnl_pct",
                                                                                                           0):
                                take_profit = True
                                reason = f"达到风险回报比{optimal_tp.get('risk_reward', 0):.2f}的止盈点"

                            # 条件4: 完成75%以上并且在高波动市场中
                            if not take_profit and completion_pct >= 0.75:
                                volatility_pattern = tp_data.get("volatility_pattern", "normal")
                                if volatility_pattern == "expansion":
                                    take_profit = True
                                    reason = f"在高波动市场中达到75%以上完成度({completion_pct:.2%})"

                            # 检查并执行止盈
                            if take_profit:
                                print_colored(f"🔔 主动监控: {symbol} {position_side} 触发最优止盈: {reason}",
                                              Colors.YELLOW)
                                success, closed = self.close_position(symbol, position_side)
                                if success:
                                    print_colored(f"✅ {symbol} {position_side} 最优止盈成功! 利润: {current_pnl:.2%}",
                                                  Colors.GREEN)
                                    self.logger.info(f"{symbol} {position_side}主动监控最优止盈", extra={
                                        "profit_pct": current_pnl,
                                        "reason": reason,
                                        "completion_pct": completion_pct,
                                        "max_profit_pct": tp_data.get('max_profit_pct', 0),
                                        "volatility_pattern": tp_data.get("volatility_pattern", "normal"),
                                        "market_stage": tp_data.get("market_stage", "unknown")
                                    })
                                    continue  # 已平仓，跳过后续逻辑

                        except Exception as e:
                            print_colored(f"⚠️ {symbol} 主动监控计算最优止盈失败: {e}", Colors.WARNING)

                        # 反转检测止盈检查
                        try:
                            # 检测FVG
                            from fvg_module import detect_fair_value_gap
                            fvg_data = detect_fair_value_gap(df)

                            # 获取市场状态
                            from market_state_module import classify_market_state
                            market_state = classify_market_state(df)

                            # 获取趋势数据
                            trend_data = get_smc_trend_and_duration(df, None, self.logger)[2]

                            # 检查反转止盈条件
                            from risk_management import manage_take_profit
                            tp_result = manage_take_profit(pos, current_price, df, fvg_data, trend_data, market_state)

                            if tp_result['take_profit']:
                                print_colored(
                                    f"🔔 主动监控: {symbol} {position_side} 触发反转止盈: {tp_result['reason']}",
                                    Colors.YELLOW)
                                success, closed = self.close_position(symbol, position_side)
                                if success:
                                    print_colored(
                                        f"✅ {symbol} {position_side} 反转止盈成功! 利润: {tp_result['current_profit_pct']:.2%}",
                                        Colors.GREEN)
                                    self.logger.info(f"{symbol} {position_side}主动监控反转止盈", extra={
                                        "profit_pct": tp_result['current_profit_pct'],
                                        "reason": tp_result['reason'],
                                        "reversal_probability": tp_result['reversal_probability'],
                                        "current_reward_ratio": tp_result['current_reward_ratio'],
                                        "atr_value": tp_result['atr_value']
                                    })
                                    continue  # 已平仓，跳过后续止损逻辑
                        except Exception as e:
                            print_colored(f"⚠️ {symbol} 反转检测失败: {e}", Colors.WARNING)

                    # 获取跟踪止损参数
                    initial_stop_loss = pos.get("initial_stop_loss", -0.0175)
                    trailing_activation = pos.get("trailing_activation", 0.012)
                    trailing_distance = pos.get("trailing_distance", 0.003)
                    trailing_active = pos.get("trailing_active", False)
                    highest_price = pos.get("highest_price", entry_price if position_side == "LONG" else 0)
                    lowest_price = pos.get("lowest_price", entry_price if position_side == "SHORT" else float('inf'))
                    current_stop_level = pos.get("current_stop_level", entry_price * (
                            1 + initial_stop_loss) if position_side == "LONG" else entry_price * (
                                1 - initial_stop_loss))

                    # 根据持仓方向分别处理
                    if position_side == "LONG":
                        profit_pct = (current_price - entry_price) / entry_price

                        # ===== 修复部分 =====
                        # 1. 激活跟踪止损仅一次
                        if not trailing_active and profit_pct >= trailing_activation:
                            pos["trailing_active"] = True
                            trailing_active = True
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 首次激活跟踪止损 (利润: {profit_pct:.2%})",
                                Colors.GREEN)

                        # 2. 检查是否创新高，需要更新止损位
                        if current_price > highest_price:
                            # 更新最高价记录
                            pos["highest_price"] = current_price
                            highest_price = current_price

                            # 计算新止损位
                            new_stop_level = highest_price * (1 - trailing_distance)

                            # =====关键修复======
                            # 确保止损位只上移不下移，通过与现有止损位比较
                            if new_stop_level > current_stop_level:
                                # 保存新的止损位
                                pos["current_stop_level"] = new_stop_level
                                current_stop_level = new_stop_level
                                print_colored(
                                    f"🔄 主动监控: {symbol} {position_side} 上移止损位至 {current_stop_level:.6f}",
                                    Colors.CYAN)

                        # 3. 检查是否触发止损
                        if current_price <= current_stop_level:
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 触发{'跟踪' if trailing_active else '初始'}止损 (价格: {current_price:.6f} <= 止损: {current_stop_level:.6f})",
                                Colors.YELLOW)
                            success, closed = self.close_position(symbol, position_side)
                            if success:
                                print_colored(f"✅ {symbol} {position_side} 止损平仓成功: {profit_pct:.2%}",
                                              Colors.GREEN)
                                self.logger.info(f"{symbol} {position_side}主动监控止损平仓", extra={
                                    "profit_pct": profit_pct,
                                    "stop_type": "trailing" if trailing_active else "initial",
                                    "entry_price": entry_price,
                                    "exit_price": current_price,
                                    "highest_price": highest_price
                                })

                    else:  # SHORT
                        profit_pct = (entry_price - current_price) / entry_price

                        # ===== 修复部分 =====
                        # 1. 激活跟踪止损仅一次
                        if not trailing_active and profit_pct >= trailing_activation:
                            pos["trailing_active"] = True
                            trailing_active = True
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 首次激活跟踪止损 (利润: {profit_pct:.2%})",
                                Colors.GREEN)

                        # 2. 检查是否创新低，需要更新止损位
                        if current_price < lowest_price or lowest_price == 0:
                            # 更新最低价记录
                            pos["lowest_price"] = current_price
                            lowest_price = current_price

                            # 计算新止损位
                            new_stop_level = lowest_price * (1 + trailing_distance)

                            # =====关键修复======
                            # 确保止损位只下移不上移，通过与现有止损位比较
                            if new_stop_level < current_stop_level or current_stop_level == 0:
                                # 保存新的止损位
                                pos["current_stop_level"] = new_stop_level
                                current_stop_level = new_stop_level
                                print_colored(
                                    f"🔄 主动监控: {symbol} {position_side} 下移止损位至 {current_stop_level:.6f}",
                                    Colors.CYAN)

                        # 3. 检查是否触发止损
                        if current_price >= current_stop_level and current_stop_level > 0:
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 触发{'跟踪' if trailing_active else '初始'}止损 (价格: {current_price:.6f} >= 止损: {current_stop_level:.6f})",
                                Colors.YELLOW)
                            success, closed = self.close_position(symbol, position_side)
                            if success:
                                print_colored(f"✅ {symbol} {position_side} 止损平仓成功: {profit_pct:.2%}",
                                              Colors.GREEN)
                                self.logger.info(f"{symbol} {position_side}主动监控止损平仓", extra={
                                    "profit_pct": profit_pct,
                                    "stop_type": "trailing" if trailing_active else "initial",
                                    "entry_price": entry_price,
                                    "exit_price": current_price,
                                    "lowest_price": lowest_price
                                })

                    # 每20秒记录一次持仓状态 (只有在check_interval足够小时才能正常工作)
                    if time.time() % 20 < check_interval:
                        profit_color = Colors.GREEN if profit_pct > 0 else Colors.RED
                        print_colored(
                            f"📊 持仓状态: {symbol} {position_side}: 利润 {profit_color}{profit_pct:.2%}{Colors.RESET}, "
                            f"当前价 {current_price:.6f}, 止损位 {current_stop_level:.6f}",
                            Colors.INFO
                        )

                # 每次检查完所有持仓后，稍微休眠以减少资源占用
                time.sleep(check_interval)

        except Exception as e:
            print(f"主动持仓监控发生错误: {e}")
            self.logger.error(f"主动持仓监控错误", extra={"error": str(e)})

            # 尝试重启监控
            print("尝试重启主动持仓监控...")
            time.sleep(5)
            self.active_position_monitor(check_interval)


    def is_near_resistance(self, price, swing_highs, fib_levels, threshold=0.01):
        """检查价格是否接近阻力位"""
        # 检查摆动高点
        for high in swing_highs:
            if abs(price - high) / price < threshold:
                return True

        # 检查斐波那契阻力位
        if fib_levels and len(fib_levels) >= 3:
            for level in fib_levels:
                if abs(price - level) / price < threshold:
                    return True

        return False

    def calculate_expected_profit(self, symbol, side, current_price, df=None):
        """
        计算预期收益百分比，用于开仓决策

        参数:
            symbol: 交易对符号
            side: 交易方向 (BUY 或 SELL)
            current_price: 当前价格
            df: 可选的数据帧，如未提供则获取

        返回:
            预期收益百分比，无法计算则返回0
        """
        try:
            # 如果未提供数据，则获取
            if df is None:
                df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                if df is None:
                    return 0.0
                df = calculate_optimized_indicators(df)
                if df is None or df.empty:
                    return 0.0

            # 获取当前的市场状态和趋势
            trend_data = get_smc_trend_and_duration(df, None, self.logger)[2]

            # 从 risk_management 导入计算函数
            from risk_management import (
                calculate_max_movement_range,
                analyze_volatility_pattern,
                analyze_market_stage,
                estimate_support_resistance_range,
                estimate_structure_move
            )

            # 创建模拟持仓对象，用于计算最大波动区间
            position = {
                "position_side": "LONG" if side == "BUY" else "SHORT",
                "entry_price": current_price,
                "initial_stop_loss": 0.008  # 默认初始止损0.8%
            }

            # 计算波动区间收敛值
            volatility_pattern = analyze_volatility_pattern(df)
            market_stage = analyze_market_stage(df)

            # 1. 波动率方法估算最大盈利空间
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.005
            atr_ratio = atr / current_price  # ATR占价格的比例

            # 基于ATR估算合理的最大波动幅度 (根据市场阶段调整)
            if market_stage == "trending":
                max_atr_multiplier = 5.0
            elif market_stage == "ranging":
                max_atr_multiplier = 3.0
            else:  # transitioning
                max_atr_multiplier = 4.0

            max_expected_move_atr = atr_ratio * max_atr_multiplier

            # 2. 基于支撑/阻力位估算最大波动
            support_resist_move = estimate_support_resistance_range(df, position["position_side"], current_price)

            # 3. 基于价格结构估算最大波动
            structure_move = estimate_structure_move(df, position["position_side"], current_price, volatility_pattern)

            # 4. 加权合并三种方法的结果
            weights = {
                'atr': 0.4,
                'sr': 0.3,
                'structure': 0.3
            }

            max_expected_move = (
                    weights['atr'] * max_expected_move_atr +
                    weights['sr'] * support_resist_move +
                    weights['structure'] * structure_move
            )

            # 检查预期收益是否大于1%
            print_colored(
                f"{symbol} {side} 预期收益计算 - "
                f"ATR法: {max_expected_move_atr:.2%}, "
                f"支撑阻力法: {support_resist_move:.2%}, "
                f"结构法: {structure_move:.2%}",
                Colors.INFO
            )

            print_colored(
                f"{symbol} {side} 最终预期收益: {max_expected_move:.2%}, "
                f"波动模式: {volatility_pattern}, 市场阶段: {market_stage}",
                Colors.GREEN if max_expected_move >= 0.01 else Colors.YELLOW
            )

            return max_expected_move

        except Exception as e:
            print_colored(f"计算预期收益失败: {e}", Colors.ERROR)
            return 0.0

    def record_entry_reason(self, symbol, side, entry_price, expected_profit):
        """记录开仓原因和预期收益"""
        timestamp = time.time()
        entry_record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "expected_profit": expected_profit,
            "timestamp": timestamp,
            "date": datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存到文件
        try:
            if not hasattr(self, 'entry_records'):
                self.entry_records = []
                # 尝试从文件加载
                if os.path.exists("entry_records.json"):
                    with open("entry_records.json", "r") as f:
                        self.entry_records = json.load(f)

            self.entry_records.append(entry_record)

            # 保存到文件
            with open("entry_records.json", "w") as f:
                json.dump(self.entry_records, f, indent=4)

            print_colored(f"✅ 已记录 {symbol} {side} 开仓原因，预期收益: {expected_profit:.2%}", Colors.GREEN)

        except Exception as e:
            print_colored(f"❌ 记录开仓原因失败: {e}", Colors.ERROR)

    def adapt_to_market_conditions(self):
        """根据市场条件动态调整交易参数 - 改进版，支持跟踪止损系统"""
        print("\n===== 市场条件分析与参数适配 =====")

        # 分析当前市场波动性
        volatility_levels = {}
        trend_strengths = {}
        market_sentiment_score = 0.0
        sentiment_factors = 0
        btc_price_change = None

        # 尝试获取BTC数据
        btc_df = None
        try:
            # 首先尝试使用get_btc_data方法
            btc_df = self.get_btc_data()

            # 检查获取的数据是否有效
            if btc_df is not None and 'close' in btc_df.columns and len(btc_df) > 20:
                print("✅ 成功获取BTC数据")
                btc_current = btc_df['close'].iloc[-1]
                btc_prev = btc_df['close'].iloc[-13]  # 约1小时前
                btc_price_change = (btc_current - btc_prev) / btc_prev * 100
                print(f"📊 BTC 1小时变化率: {btc_price_change:.2f}%")
            else:
                print("⚠️ 获取的BTC数据无效或不完整")
                btc_df = None
        except Exception as e:
            print(f"⚠️ 获取BTC数据时出错: {e}")
            btc_df = None

        # 如果无法获取BTC数据，尝试使用ETH或其他替代方法
        if btc_df is None:
            print("🔄 尝试替代方法获取市场情绪...")

            # 尝试方法1: 直接使用futures_symbol_ticker获取BTC当前价格
            try:
                ticker_now = self.client.futures_symbol_ticker(symbol="BTCUSDT")
                current_price = float(ticker_now['price'])

                # 获取历史价格（通过klines获取单个数据点）
                klines = self.client.futures_klines(symbol="BTCUSDT", interval="1h", limit=2)
                if klines and len(klines) >= 2:
                    prev_price = float(klines[0][4])  # 1小时前的收盘价
                    btc_price_change = (current_price - prev_price) / prev_price * 100
                    print(f"📊 BTC 1小时变化率(替代方法): {btc_price_change:.2f}%")
                else:
                    print("⚠️ 无法获取BTC历史数据，无法计算价格变化")
            except Exception as e:
                print(f"⚠️ 替代方法获取BTC数据失败: {e}")

            # 尝试方法2: 使用ETH数据
            if btc_price_change is None:
                try:
                    eth_df = self.get_historical_data_with_cache("ETHUSDT", force_refresh=True)
                    if eth_df is not None and 'close' in eth_df.columns and len(eth_df) > 20:
                        eth_current = eth_df['close'].iloc[-1]
                        eth_prev = eth_df['close'].iloc[-13]  # 约1小时前
                        eth_price_change = (eth_current - eth_prev) / eth_prev * 100
                        print(f"📊 ETH 1小时变化率: {eth_price_change:.2f}% (BTC数据不可用，使用ETH替代)")
                        btc_price_change = eth_price_change  # 使用ETH的变化率代替BTC
                    else:
                        print(f"⚠️ ETH数据不可用，将使用其他指标分析市场情绪")
                except Exception as e:
                    print(f"⚠️ 获取ETH数据出错: {e}")

        # 分析各交易对的波动性和趋势强度
        for symbol in self.config["TRADE_PAIRS"]:
            df = self.get_historical_data_with_cache(symbol, force_refresh=True)
            if df is not None and 'close' in df.columns and len(df) > 20:
                # 计算波动性（当前ATR相对于历史的比率）
                if 'ATR' in df.columns:
                    current_atr = df['ATR'].iloc[-1]
                    avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
                    volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
                    volatility_levels[symbol] = volatility_ratio

                    # 检查趋势强度
                    if 'ADX' in df.columns:
                        adx = df['ADX'].iloc[-1]
                        trend_strengths[symbol] = adx

                # 计算1小时价格变化，用于市场情绪计算
                if len(df) >= 13:  # 确保有足够数据
                    recent_change = (df['close'].iloc[-1] - df['close'].iloc[-13]) / df['close'].iloc[-13] * 100
                    market_sentiment_score += recent_change
                    sentiment_factors += 1
                    print(f"📊 {symbol} 1小时变化率: {recent_change:.2f}%")

        # 如果BTC/ETH数据可用，给予更高权重
        if btc_price_change is not None:
            market_sentiment_score += btc_price_change * 3  # BTC变化的权重是普通交易对的3倍
            sentiment_factors += 3
            print(f"赋予BTC变化率 {btc_price_change:.2f}% 三倍权重")

        # 计算平均市场情绪分数
        if sentiment_factors > 0:
            avg_market_sentiment = market_sentiment_score / sentiment_factors
            print(f"📊 平均市场情绪得分: {avg_market_sentiment:.2f}%")

            # 根据得分确定市场情绪
            if avg_market_sentiment > 1.5:
                market_bias = "bullish"
                print(f"📊 市场情绪: 看涨 ({avg_market_sentiment:.2f}%)")
            elif avg_market_sentiment < -1.5:
                market_bias = "bearish"
                print(f"📊 市场情绪: 看跌 ({avg_market_sentiment:.2f}%)")
            else:
                market_bias = "neutral"
                print(f"📊 市场情绪: 中性 ({avg_market_sentiment:.2f}%)")
        else:
            # 极少情况下，无法获取任何有效数据
            market_bias = "neutral"
            print(f"⚠️ 无法收集足够市场数据，默认中性情绪")

        # 计算整体市场波动性
        if volatility_levels:
            avg_volatility = sum(volatility_levels.values()) / len(volatility_levels)
            print(f"📈 平均市场波动性: {avg_volatility:.2f}x (1.0为正常水平)")

            # 波动性高低排名
            high_vol_pairs = sorted(volatility_levels.items(), key=lambda x: x[1], reverse=True)[:3]
            low_vol_pairs = sorted(volatility_levels.items(), key=lambda x: x[1])[:3]

            print("📊 高波动交易对:")
            for sym, vol in high_vol_pairs:
                print(f"  - {sym}: {vol:.2f}x")

            print("📊 低波动交易对:")
            for sym, vol in low_vol_pairs:
                print(f"  - {sym}: {vol:.2f}x")
        else:
            avg_volatility = 1.0  # 默认值

        # 计算整体趋势强度
        if trend_strengths:
            avg_trend_strength = sum(trend_strengths.values()) / len(trend_strengths)
            print(f"📏 平均趋势强度(ADX): {avg_trend_strength:.2f} (>25为强趋势)")

            # 趋势强度排名
            strong_trend_pairs = sorted(trend_strengths.items(), key=lambda x: x[1], reverse=True)[:3]
            weak_trend_pairs = sorted(trend_strengths.items(), key=lambda x: x[1])[:3]

            print("📊 强趋势交易对:")
            for sym, adx in strong_trend_pairs:
                print(f"  - {sym}: ADX {adx:.2f}")
        else:
            avg_trend_strength = 20.0  # 默认值

        # 根据市场条件调整交易参数 - 适配跟踪止损系统
        # 1. 波动性调整
        if avg_volatility > 1.5:  # 市场波动性高于平均50%
            # 高波动环境
            initial_stop_loss = 0.020  # 加大初始止损到2.0%
            trailing_activation = 0.015  # 提高激活阈值到1.5%
            trailing_distance_min = 0.003  # 维持标准跟踪距离0.3%
            trailing_distance_max = 0.005  # 增加最大跟踪距离到0.5%

            print(f"⚠️ 市场波动性较高，调整初始止损至2.0%，跟踪激活阈值至1.5%，跟踪距离0.3-0.5%")

            # 记录调整
            self.logger.info("市场波动性高，调整交易参数", extra={
                "volatility": avg_volatility,
                "initial_stop_loss": initial_stop_loss,
                "trailing_activation": trailing_activation,
                "trailing_distance_range": f"{trailing_distance_min}-{trailing_distance_max}"
            })
        elif avg_volatility < 0.7:  # 市场波动性低于平均30%
            # 低波动环境
            initial_stop_loss = 0.006  # 缩小初始止损到0.6%
            trailing_activation = 0.010  # 降低激活阈值到1.0%
            trailing_distance_min = 0.001  # 降低最小跟踪距离到0.1%
            trailing_distance_max = 0.002  # 降低最大跟踪距离到0.2%

            print(f"ℹ️ 市场波动性较低，调整初始止损至0.6%，跟踪激活阈值至1.0%，跟踪距离0.1-0.2%")

            # 记录调整
            self.logger.info("市场波动性低，调整交易参数", extra={
                "volatility": avg_volatility,
                "initial_stop_loss": initial_stop_loss,
                "trailing_activation": trailing_activation,
                "trailing_distance_range": f"{trailing_distance_min}-{trailing_distance_max}"
            })
        else:
            # 正常波动环境，使用默认值
            initial_stop_loss = 0.008  # 默认初始止损0.8%
            trailing_activation = 0.012  # 默认激活阈值1.2%
            trailing_distance_min = 0.002  # 默认最小跟踪距离0.2%
            trailing_distance_max = 0.004  # 默认最大跟踪距离0.4%

            print(f"ℹ️ 市场波动性正常，使用默认跟踪止损参数 (初始止损0.8%，激活阈值1.2%，跟踪距离0.2-0.4%)")

            # 记录使用默认值
            self.logger.info("市场波动性正常，使用默认参数", extra={
                "volatility": avg_volatility,
                "initial_stop_loss": initial_stop_loss,
                "trailing_activation": trailing_activation,
                "trailing_distance_range": f"{trailing_distance_min}-{trailing_distance_max}"
            })

        # 更新参数
        self.dynamic_stop_loss = -initial_stop_loss  # 保持接口兼容性，但现在表示初始止损
        self.trailing_activation = trailing_activation
        self.trailing_min_distance = trailing_distance_min
        self.trailing_max_distance = trailing_distance_max

        # 2. 市场情绪调整
        self.market_bias = market_bias

        # 3. 趋势强度调整
        if avg_trend_strength > 30:  # 强趋势市场
            print(f"🔍 强趋势市场(ADX={avg_trend_strength:.2f})，优先选择趋势明确的交易对")
            self.trend_priority = True

            # 可以记录强趋势的交易对，优先考虑
            self.strong_trend_symbols = [sym for sym, adx in trend_strengths.items() if adx > 25]
            if self.strong_trend_symbols:
                print(f"💡 趋势明确的优先交易对: {', '.join(self.strong_trend_symbols)}")
        else:
            print(f"🔍 弱趋势或震荡市场(ADX={avg_trend_strength:.2f})，关注支撑阻力")
            self.trend_priority = False
            self.strong_trend_symbols = []

        return {
            "volatility": avg_volatility if 'avg_volatility' in locals() else 1.0,
            "trend_strength": avg_trend_strength if 'avg_trend_strength' in locals() else 20.0,
            "btc_change": btc_price_change,
            "initial_stop_loss": initial_stop_loss,
            "trailing_activation": trailing_activation,
            "trailing_distance_min": trailing_distance_min,
            "trailing_distance_max": trailing_distance_max,
            "market_bias": self.market_bias
        }


    def is_near_support(self, price, swing_lows, fib_levels, threshold=0.01):
        """检查价格是否接近支撑位"""
        # 检查摆动低点
        for low in swing_lows:
            if abs(price - low) / price < threshold:
                return True

        # 检查斐波那契支撑位
        if fib_levels and len(fib_levels) >= 3:
            for level in fib_levels:
                if abs(price - level) / price < threshold:
                    return True

        return False

    def place_hedge_orders(self, symbol, primary_side, quality_score):
        """
        根据质量评分和信号放置订单，支持双向持仓 - 修复版
        """
        account_balance = self.get_futures_balance()

        if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
            self.logger.warning(f"账户余额不足，无法交易: {account_balance} USDC")
            return False

        # 计算下单金额，确保不超过账户余额的5%
        order_amount = account_balance * 0.05
        print(f"📊 账户余额: {account_balance} USDC, 下单金额: {order_amount:.2f} USDC (5%)")

        # 双向持仓模式
        if primary_side == "BOTH":
            # 质量评分在中间区域时采用双向持仓
            if 4.0 <= quality_score <= 6.0:
                # 使用6:4比例分配多空仓位
                long_ratio = 0.6
                short_ratio = 0.4

                long_amount = order_amount * long_ratio
                short_amount = order_amount * short_ratio

                print(f"🔄 执行双向持仓 - 多头: {long_amount:.2f} USDC, 空头: {short_amount:.2f} USDC")

                # 计算每个方向的杠杆
                long_leverage = self.calculate_leverage_from_quality(quality_score)
                short_leverage = max(1, long_leverage - 2)  # 空头杠杆略低

                # 先执行多头订单
                long_success = self.place_futures_order_usdc(symbol, "BUY", long_amount, long_leverage)
                time.sleep(1)
                # 再执行空头订单
                short_success = self.place_futures_order_usdc(symbol, "SELL", short_amount, short_leverage)

                return long_success or short_success
            else:
                # 偏向某一方向
                side = "BUY" if quality_score > 5.0 else "SELL"
                leverage = self.calculate_leverage_from_quality(quality_score)
                return self.place_futures_order_usdc(symbol, side, order_amount, leverage)

        elif primary_side in ["BUY", "SELL"]:
            # 根据评分调整杠杆倍数
            leverage = self.calculate_leverage_from_quality(quality_score)
            return self.place_futures_order_usdc(symbol, primary_side, order_amount, leverage)
        else:
            self.logger.warning(f"{symbol}未知交易方向: {primary_side}")
            return False

    def get_futures_balance(self):
        """获取USDC期货账户余额"""
        try:
            assets = self.client.futures_account_balance()
            for asset in assets:
                if asset["asset"] == "USDC":
                    return float(asset["balance"])
            return 0.0
        except Exception as e:
            self.logger.error(f"获取期货余额失败: {e}")
            return 0.0

    def get_historical_data_with_cache(self, symbol, interval="15m", limit=200, force_refresh=False):
        """获取历史数据，使用缓存减少API调用 - 改进版"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()

        # 更频繁刷新缓存 - 减少到5分钟
        cache_ttl = 300  # 5分钟

        # 对于长时间运行的会话，每小时强制刷新一次
        hourly_force_refresh = self.trade_cycle % 12 == 0  # 假设每5分钟一个周期

        # 检查缓存是否存在且有效
        if not force_refresh and not hourly_force_refresh and cache_key in self.historical_data_cache:
            cache_item = self.historical_data_cache[cache_key]
            if current_time - cache_item['timestamp'] < cache_ttl:
                self.logger.info(f"使用缓存数据: {symbol}")
                return cache_item['data']

        # 获取新数据
        try:
            df = get_historical_data(self.client, symbol)
            if df is not None and not df.empty:
                # 缓存数据
                self.historical_data_cache[cache_key] = {
                    'data': df,
                    'timestamp': current_time
                }
                self.logger.info(f"获取并缓存新数据: {symbol}")
                return df
            else:
                self.logger.warning(f"无法获取{symbol}的数据")
                return None
        except Exception as e:
            self.logger.error(f"获取{symbol}历史数据失败: {e}")
            return None

    def predict_short_term_price(self, symbol, horizon_minutes=60):
        """预测短期价格走势"""
        df = self.get_historical_data_with_cache(symbol)
        if df is None or df.empty or len(df) < 20:
            self.logger.warning(f"{symbol}数据不足，无法预测价格")
            return None

        try:
            # 计算指标
            df = calculate_optimized_indicators(df)
            if df is None or df.empty:
                return None

            # 使用简单线性回归预测价格
            window_length = min(self.config.get("PREDICTION_WINDOW", 60), len(df))
            window = df['close'].tail(window_length)
            smoothed = window.rolling(window=3, min_periods=1).mean().bfill()

            x = np.arange(len(smoothed))
            slope, intercept = np.polyfit(x, smoothed, 1)

            current_price = smoothed.iloc[-1]
            candles_needed = horizon_minutes / 15.0  # 假设15分钟K线
            multiplier = self.config.get("PREDICTION_MULTIPLIER", 15)

            predicted_price = current_price + slope * candles_needed * multiplier

            # 确保预测有意义
            if slope > 0 and predicted_price < current_price:
                predicted_price = current_price * 1.01  # 至少上涨1%
            elif slope < 0 and predicted_price > current_price:
                predicted_price = current_price * 0.99  # 至少下跌1%

            # 限制在历史范围内
            hist_max = window.max() * 1.05  # 允许5%的超出
            hist_min = window.min() * 0.95  # 允许5%的超出
            predicted_price = min(max(predicted_price, hist_min), hist_max)

            self.logger.info(f"{symbol}价格预测: {predicted_price:.6f}", extra={
                "current_price": current_price,
                "predicted_price": predicted_price,
                "horizon_minutes": horizon_minutes,
                "slope": slope
            })

            return predicted_price
        except Exception as e:
            self.logger.error(f"{symbol}价格预测失败: {e}")
            return None

    def manage_resources(self):
        """定期管理和清理资源，防止内存泄漏"""
        # 启动时间
        if not hasattr(self, 'resource_management_start_time'):
            self.resource_management_start_time = time.time()
            return

        # 当前内存使用统计
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # 转换为MB

        # 日志记录内存使用
        print(f"ℹ️ 当前内存使用: {memory_usage:.2f} MB")
        self.logger.info(f"内存使用情况", extra={"memory_mb": memory_usage})

        # 限制缓存大小
        if len(self.historical_data_cache) > 50:
            # 删除最老的缓存
            oldest_keys = sorted(
                self.historical_data_cache.keys(),
                key=lambda k: self.historical_data_cache[k]['timestamp']
            )[:10]

            for key in oldest_keys:
                del self.historical_data_cache[key]

            print(f"🧹 清理了{len(oldest_keys)}个历史数据缓存项")
            self.logger.info(f"清理历史数据缓存", extra={"cleaned_items": len(oldest_keys)})

        # 限制持仓历史记录大小
        if hasattr(self, 'position_history') and len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
            self._save_position_history()
            print(f"🧹 持仓历史记录裁剪至1000条")
            self.logger.info(f"裁剪持仓历史记录", extra={"max_records": 1000})

        # 重置一些累积的统计数据
        if self.trade_cycle % 100 == 0:
            self.quality_score_history = {}
            self.similar_patterns_history = {}
            print(f"🔄 重置质量评分历史和相似模式历史")
            self.logger.info(f"重置累积统计数据")

        # 运行垃圾回收
        import gc
        collected = gc.collect()
        print(f"♻️ 垃圾回收完成，释放了{collected}个对象")

        # 计算运行时间
        run_hours = (time.time() - self.resource_management_start_time) / 3600
        print(f"⏱️ 机器人已运行: {run_hours:.2f}小时")

    def generate_trade_signal(self, df, symbol):
        """生成交易信号 - 添加RVI过滤"""

        if df is None or len(df) < 20:
            return "HOLD", 0

        try:
            # 计算指标
            df = self.calculate_simplified_indicators(df)
            if df is None or df.empty:
                return "HOLD", 0

            # 计算质量评分
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config, self.logger)
            print_colored(f"{symbol} 初始质量评分: {quality_score:.2f}", Colors.INFO)

            # 获取多时间框架信号
            signal, adjusted_score, details = self.mtf_coordinator.generate_signal(symbol, quality_score)
            print_colored(f"多时间框架信号: {signal}, 调整后评分: {adjusted_score:.2f}", Colors.INFO)

            # RVI过滤（如果启用）
            if signal != "HOLD" and self.config.get('USE_RVI_FILTER', True):
                from rvi_indicator import rvi_entry_filter
                position_side = 'LONG' if signal == 'BUY' else 'SHORT'
                rvi_allow, rvi_reason = rvi_entry_filter(df, position_side)

                if not rvi_allow:
                    print_colored(f"❌ RVI过滤: {rvi_reason}", Colors.YELLOW)
                    return "HOLD", 0
                else:
                    print_colored(f"✅ RVI确认: {rvi_reason}", Colors.GREEN)

            # 考虑市场偏向
            if hasattr(self, 'preferred_direction') and self.preferred_direction:
                if (self.preferred_direction == "LONG" and signal == "SELL") or \
                        (self.preferred_direction == "SHORT" and signal == "BUY"):
                    print_colored(f"信号与偏向冲突，降低评分", Colors.YELLOW)
                    adjusted_score *= 0.7

            return signal, adjusted_score

        except Exception as e:
            self.logger.error(f"生成{symbol}交易信号时出错: {e}")
            return "HOLD", 0

    def predict_price_movement(self, symbol, df, current_price, direction):
        """
        预测价格移动，预测未来收益能否达到1%

        参数:
            symbol: 交易对
            df: 价格数据
            current_price: 当前价格
            direction: 预期方向 ("UP" 或 "DOWN")

        返回:
            预期收益百分比, 是否达到1%阈值
        """
        try:
            # 获取ATR
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.005

            # 获取趋势强度
            trend, duration, trend_info = get_smc_trend_and_duration(df)
            trend_strength = 1.0

            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[-1]
                if adx > 30:
                    trend_strength = 1.5
                elif adx > 20:
                    trend_strength = 1.2
                elif adx < 15:
                    trend_strength = 0.7

            # 检测市场状态
            from market_state_module import classify_market_state
            market_state = classify_market_state(df)
            volatility_factor = 1.0

            if market_state["state"] == "RANGING":
                volatility_factor = 0.7  # 震荡市场收益预期降低
            elif "COMPRESSION" in market_state["state"]:
                volatility_factor = 1.3  # 压缩后可能有更大波动
            elif "VOLATILE" in market_state["state"]:
                volatility_factor = 1.2  # 波动市场可能有更大收益

            # 基于ATR和其他因素预测潜在移动
            price_movement_pct = (atr / current_price) * 3.0 * trend_strength * volatility_factor

            # 检查方向一致性
            direction_factor = 1.0
            if (direction == "UP" and trend == "UP") or (direction == "DOWN" and trend == "DOWN"):
                direction_factor = 1.2  # 方向与趋势一致，提高预期
            elif (direction == "UP" and trend == "DOWN") or (direction == "DOWN" and trend == "UP"):
                direction_factor = 0.7  # 方向与趋势相反，降低预期

            # 最终预期收益
            expected_profit = price_movement_pct * direction_factor

            # 检查是否达到1%
            meets_threshold = expected_profit >= 0.01  # 1%

            print_colored(
                f"{symbol} {direction}方向预期收益: {expected_profit:.2%}, "
                f"是否达到1%阈值: {meets_threshold}, "
                f"ATR因子: {atr / current_price:.2%}, 趋势强度: {trend_strength}, "
                f"波动因子: {volatility_factor}, 方向因子: {direction_factor}",
                Colors.GREEN if meets_threshold else Colors.YELLOW
            )

            return expected_profit, meets_threshold

        except Exception as e:
            print_colored(f"预测价格移动出错: {e}", Colors.ERROR)
            return 0.005, False  # 默认0.5%，不达标

    def place_hedge_orders(self, symbol, primary_side, quality_score):
        """根据质量评分和信号放置订单，支持双向持仓"""
        account_balance = self.get_futures_balance()

        if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
            self.logger.warning(f"账户余额不足，无法交易: {account_balance} USDC")
            return False

        # 检查当前持仓
        total_exposure, symbol_exposures = get_total_position_exposure(self.open_positions, account_balance)
        symbol_exposure = symbol_exposures.get(symbol, 0)

        # 计算下单金额
        order_amount, order_pct = calculate_order_amount(
            account_balance,
            symbol_exposure,
            max_total_exposure=85,
            max_symbol_exposure=15,
            default_order_pct=5
        )

        if order_amount <= 0:
            self.logger.warning(f"{symbol}下单金额过小或超出限额")
            return False

        # 双向持仓模式
        if primary_side == "BOTH":
            # 质量评分在中间区域时采用双向持仓
            if 4.0 <= quality_score <= 6.0:
                long_amount = order_amount * 0.6  # 60%做多
                short_amount = order_amount * 0.4  # 40%做空

                long_success = self.place_futures_order_usdc(symbol, "BUY", long_amount)
                time.sleep(1)  # 避免API请求过快
                short_success = self.place_futures_order_usdc(symbol, "SELL", short_amount)

                if long_success and short_success:
                    self.logger.info(f"{symbol}双向持仓成功", extra={
                        "long_amount": long_amount,
                        "short_amount": short_amount,
                        "quality_score": quality_score
                    })
                    return True
                else:
                    self.logger.warning(f"{symbol}双向持仓部分失败", extra={
                        "long_success": long_success,
                        "short_success": short_success
                    })
                    return long_success or short_success
            else:
                # 偏向某一方向
                side = "BUY" if quality_score > 5.0 else "SELL"
                return self.place_futures_order_usdc(symbol, side, order_amount)

        elif primary_side in ["BUY", "SELL"]:
            # 根据评分调整杠杆倍数
            leverage = self.calculate_leverage_from_quality(quality_score)
            return self.place_futures_order_usdc(symbol, primary_side, order_amount, leverage)
        else:
            self.logger.warning(f"{symbol}未知交易方向: {primary_side}")
            return False

    def calculate_leverage_from_quality(self, quality_score):
        """根据质量评分计算合适的杠杆水平"""
        if quality_score >= 9.0:
            return 20  # 最高质量，最高杠杆
        elif quality_score >= 8.0:
            return 15
        elif quality_score >= 7.0:
            return 10
        elif quality_score >= 6.0:
            return 8
        elif quality_score >= 5.0:
            return 5
        elif quality_score >= 4.0:
            return 3
        else:
            return 2  # 默认低杠杆

    def place_futures_order_usdc(self, symbol: str, side: str, amount: float, leverage: int = 5) -> bool:
        """
        执行期货市场订单 - 改进版本，添加预期收益检查

        参数:
            symbol: 交易对符号
            side: 交易方向 ('BUY' 或 'SELL')
            amount: 订单金额 (USDC)
            leverage: 杠杆倍数

        返回:
            下单是否成功
        """
        import math
        import time
        from logger_utils import Colors, print_colored

        try:
            # 获取当前账户余额
            account_balance = self.get_futures_balance()
            print(f"📊 当前账户余额: {account_balance:.2f} USDC")

            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 计算预期收益 - 新增部分 ⭐
            expected_profit = self.calculate_expected_profit(symbol, side, current_price)

            # 检查预期收益是否达到阈值
            min_profit_threshold = 0.01  # 1%最小预期收益
            if expected_profit < min_profit_threshold:
                print_colored(
                    f"⚠️ {symbol} {side} 预期收益 ({expected_profit:.2%}) 低于最小阈值 ({min_profit_threshold:.2%})，取消交易",
                    Colors.WARNING
                )
                self.logger.warning(f"{symbol} {side} 预期收益不足", extra={
                    "expected_profit": expected_profit,
                    "threshold": min_profit_threshold,
                    "current_price": current_price
                })
                return False

            # 基于ATR的止损计算
            df = self.get_historical_data_with_cache(symbol, force_refresh=True)
            if df is None:
                print_colored(f"⚠️ 无法获取{symbol}历史数据，使用默认止损比例", Colors.WARNING)
                initial_stop_loss = -0.008  # 默认0.8%
            else:
                df = calculate_optimized_indicators(df)
                if 'ATR' in df.columns:
                    # 使用ATR作为止损距离基础
                    atr = df['ATR'].iloc[-1]
                    # 计算ATR的价格百分比表示
                    atr_pct = atr / current_price

                    # 使用1.0-1.5倍ATR作为止损距离，根据波动性调整
                    if side == "BUY":
                        initial_stop_loss = -1.0 * atr_pct  # 1倍ATR
                    else:
                        initial_stop_loss = -1.0 * atr_pct  # 1倍ATR

                    print_colored(f"📊 {symbol} 基于ATR的止损距离: {abs(initial_stop_loss) * 100:.2f}% (ATR: {atr:.6f})",
                                  Colors.INFO)
                else:
                    print_colored(f"⚠️ {symbol} 未找到ATR指标，使用默认止损比例", Colors.WARNING)
                    initial_stop_loss = -0.008  # 默认0.8%

            # 确保最小止损距离
            min_stop_loss = -0.005  # 最小0.5%
            initial_stop_loss = min(initial_stop_loss, min_stop_loss)

            # 检测FVG和市场状态，优化入场
            try:
                from fvg_module import detect_fair_value_gap
                from market_state_module import classify_market_state
                from risk_management import optimize_entry_timing

                if df is not None:
                    # 检测FVG
                    fvg_data = detect_fair_value_gap(df)

                    # 分析市场状态
                    market_state = classify_market_state(df)

                    # 获取趋势数据
                    trend_data = get_smc_trend_and_duration(df, None, self.logger)[2]

                    # 优化入场时机 - 这里应该直接传递正确的参数
                    entry_data = optimize_entry_timing(
                        df,
                        fvg_data,
                        market_state,
                        side,  # 使用传入的side而不是可能不存在的quality_score
                        0.0 if not 'quality_score' in locals() else quality_score,  # 提供默认值
                        current_price
                    )

                    # 如果推荐等待，且不是强制市场订单
                    if entry_data["should_wait"] and 'order_type' in locals() and order_type != "MARKET":
                        print_colored(f"⚠️ {symbol} 建议等待更好入场点: {entry_data['expected_entry_price']:.6f}",
                                      Colors.WARNING)
                        print_colored(
                            f"原因: {entry_data['entry_conditions'][0] if entry_data['entry_conditions'] else '入场时机不佳'}",
                            Colors.WARNING)
                        return False
            except Exception as e:
                print_colored(f"⚠️ {symbol} 入场优化失败: {e}", Colors.WARNING)
                self.logger.warning(f"{symbol}入场优化失败", extra={"error": str(e)})

            # 严格限制订单金额不超过账户余额的5%
            max_allowed_amount = account_balance * 0.05

            if amount > max_allowed_amount:
                print(f"⚠️ 订单金额 {amount:.2f} USDC 超过账户余额5%限制，已调整为 {max_allowed_amount:.2f} USDC")
                amount = max_allowed_amount

            # 确保最低订单金额
            min_amount = self.config.get("MIN_NOTIONAL", 5)
            if amount < min_amount and account_balance >= min_amount:
                amount = min_amount
                print(f"⚠️ 订单金额已调整至最低限额: {min_amount} USDC")

            # 获取交易对信息，添加错误处理和默认值
            step_size = None
            min_qty = None
            max_qty = None
            notional_min = None

            try:
                # 获取交易对信息
                info = self.client.futures_exchange_info()

                # 查找该交易对的所有过滤器
                for item in info['symbols']:
                    if item['symbol'] == symbol:
                        for f in item['filters']:
                            # 数量精度
                            if f['filterType'] == 'LOT_SIZE':
                                step_size = float(f['stepSize'])
                                min_qty = float(f['minQty'])
                                max_qty = float(f['maxQty'])
                            # 最小订单价值
                            elif f['filterType'] == 'MIN_NOTIONAL':
                                notional_min = float(f.get('notional', 0))
                        break
            except Exception as e:
                print_colored(f"⚠️ 获取{symbol}交易信息失败: {e}，使用默认值", Colors.WARNING)
                self.logger.warning(f"获取交易信息失败: {e}", extra={"symbol": symbol})

            # 如果无法获取交易信息，使用安全的默认值
            if step_size is None:
                print_colored(f"⚠️ {symbol} 无法获取精度信息，使用默认值", Colors.WARNING)

                # 根据价格范围设置合理的默认值
                if current_price < 0.1:
                    step_size = 1  # 小币种通常可以买整数个
                    min_qty = 1
                    max_qty = 9000000
                elif current_price < 1:
                    step_size = 0.1
                    min_qty = 0.1
                    max_qty = 900000
                elif current_price < 10:
                    step_size = 0.01
                    min_qty = 0.01
                    max_qty = 90000
                elif current_price < 100:
                    step_size = 0.001
                    min_qty = 0.001
                    max_qty = 9000
                elif current_price < 1000:
                    step_size = 0.0001
                    min_qty = 0.0001
                    max_qty = 900
                else:
                    step_size = 0.00001
                    min_qty = 0.00001
                    max_qty = 90

                notional_min = 5  # 大多数交易所的最低订单价值是5 USDT/USDC

            # 计算数量并应用精度限制
            raw_qty = amount / current_price

            # 计算实际需要的保证金
            margin_required = amount / leverage
            if margin_required > account_balance:
                print(f"❌ 保证金不足: 需要 {margin_required:.2f} USDC, 账户余额 {account_balance:.2f} USDC")
                return False

            # 应用数量精度
            precision = int(round(-math.log(step_size, 10), 0)) if step_size < 1 else 0
            quantity = math.floor(raw_qty * 10 ** precision) / 10 ** precision

            # 确保数量>=最小数量
            if quantity < min_qty:
                print_colored(f"⚠️ {symbol} 数量 {quantity} 小于最小交易量 {min_qty}，已调整", Colors.WARNING)
                quantity = min_qty

            # 确保数量<=最大数量
            if max_qty and quantity > max_qty:
                print_colored(f"⚠️ {symbol} 数量 {quantity} 大于最大交易量 {max_qty}，已调整", Colors.WARNING)
                quantity = max_qty

            # 格式化为字符串(避免科学计数法问题)
            if precision > 0:
                qty_str = f"{quantity:.{precision}f}"
            else:
                qty_str = str(int(quantity))

            # 检查最小订单价值
            notional = quantity * current_price
            if notional_min and notional < notional_min:
                print_colored(f"⚠️ {symbol} 订单价值 ({notional:.2f}) 低于最小要求 ({notional_min})", Colors.WARNING)
                new_qty = math.ceil(notional_min / current_price * 10 ** precision) / 10 ** precision
                quantity = max(min_qty, new_qty)

                # 更新格式化后的数量字符串
                if precision > 0:
                    qty_str = f"{quantity:.{precision}f}"
                else:
                    qty_str = str(int(quantity))

                notional = quantity * current_price

            print_colored(f"🔢 {symbol} 计划交易: 金额={amount:.2f} USDC, 数量={quantity}, 价格={current_price}",
                          Colors.INFO)
            print_colored(f"🔢 杠杆: {leverage}倍, 实际保证金: {notional / leverage:.2f} USDC", Colors.INFO)
            print_colored(f"📈 预期收益: {expected_profit:.2%}", Colors.INFO)

            # 设置杠杆
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                print(f"✅ {symbol} 设置杠杆成功: {leverage}倍")
            except Exception as e:
                print(f"⚠️ {symbol} 设置杠杆失败: {e}，使用默认杠杆 1")
                leverage = 1

            # 执行交易
            try:
                if hasattr(self, 'hedge_mode_enabled') and self.hedge_mode_enabled:
                    # 双向持仓模式
                    pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=qty_str,
                        positionSide=pos_side
                    )
                else:
                    # 单向持仓模式
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=qty_str
                    )

                print_colored(f"✅ {side} {symbol} 成功, 数量={quantity}, 杠杆={leverage}倍", Colors.GREEN)
                self.logger.info(f"{symbol} {side} 订单成功", extra={
                    "order_id": order.get("orderId", "unknown"),
                    "quantity": quantity,
                    "notional": notional,
                    "leverage": leverage,
                    "expected_profit": expected_profit,
                    "initial_stop_loss": abs(initial_stop_loss) * 100,
                    "trailing_activation": 0.012 * 100,
                    "trailing_distance": 0.003 * 100
                })

                # 记录持仓信息 - 新的跟踪止损系统
                self.record_position_with_trailing_stop(
                    symbol=symbol,
                    side=side,
                    entry_price=current_price,
                    quantity=quantity,
                    initial_stop_loss=initial_stop_loss if side.upper() == "SELL" else -initial_stop_loss,  # 根据方向设置符号
                    trailing_activation=0.012,  # 激活跟踪止损的阈值 1.2%
                    trailing_distance=0.003,  # 跟踪距离 0.3%
                    expected_profit=expected_profit  # 新增：记录预期收益
                )

                # 记录开仓原因
                self.record_entry_reason(symbol, side, current_price, expected_profit)

                return True

            except Exception as e:
                order_error = str(e)
                print_colored(f"❌ {symbol} {side} 订单执行失败: {order_error}", Colors.ERROR)

                if "insufficient balance" in order_error.lower() or "margin is insufficient" in order_error.lower():
                    print_colored(f"  原因: 账户余额或保证金不足", Colors.WARNING)
                    print_colored(f"  当前余额: {account_balance} USDC, 需要保证金: {notional / leverage:.2f} USDC",
                                  Colors.WARNING)
                elif "precision" in order_error.lower():
                    print_colored(f"  原因: 价格或数量精度不正确", Colors.WARNING)
                elif "lot size" in order_error.lower():
                    print_colored(f"  原因: 订单大小不符合要求", Colors.WARNING)
                elif "min notional" in order_error.lower():
                    print_colored(f"  原因: 订单价值低于最小要求", Colors.WARNING)

                self.logger.error(f"{symbol} {side} 交易失败", extra={"error": order_error})
                return False

        except Exception as e:
            print_colored(f"❌ {symbol} {side} 交易过程中发生错误: {e}", Colors.ERROR)
            self.logger.error(f"{symbol} 交易错误", extra={"error": str(e)})
            return False

    def trade(self):
        """增强版多时框架集成交易循环，包含主动持仓监控"""
        import threading

        print("启动增强版多时间框架集成交易机器人...")
        self.logger.info("增强版多时间框架集成交易机器人启动", extra={"version": "Enhanced-MTF-" + VERSION})

        # 在单独的线程中启动主动持仓监控
        monitor_thread = threading.Thread(target=self.active_position_monitor, args=(15,), daemon=True)
        monitor_thread.start()
        print("✅ 主动持仓监控已在后台启动（每15秒检查一次）")

        # 初始化API连接
        self.check_and_reconnect_api()

        # 转换现有持仓到跟踪止损系统
        self.convert_positions_to_trailing_stop()

        # 最低质量评分要求 - 新增的参数设置
        min_quality_score = 6.80  # 只购买评分7.80及以上的交易对
        print(f"✅ 设置最低质量评分要求: {min_quality_score}")

        while True:
            try:
                self.trade_cycle += 1
                print(f"\n======== 交易循环 #{self.trade_cycle} ========")
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"当前时间: {current_time}")

                # 每10个周期运行资源管理和API检查
                if self.trade_cycle % 10 == 0:
                    self.manage_resources()
                    self.check_and_reconnect_api()

                # 每5个周期分析一次市场条件
                if self.trade_cycle % 5 == 0:
                    print("\n----- 分析市场条件 -----")
                    market_conditions = self.adapt_to_market_conditions()
                    market_bias = market_conditions['market_bias']
                    print(
                        f"市场分析完成: {'看涨' if market_bias == 'bullish' else '看跌' if market_bias == 'bearish' else '中性'} 偏向")

                # 获取账户余额
                account_balance = self.get_futures_balance()
                print(f"账户余额: {account_balance:.2f} USDC")
                self.logger.info("账户余额", extra={"balance": account_balance})

                if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
                    print(f"⚠️ 账户余额不足，最低要求: {self.config.get('MIN_MARGIN_BALANCE', 10)} USDC")
                    self.logger.warning("账户余额不足", extra={"balance": account_balance,
                                                               "min_required": self.config.get("MIN_MARGIN_BALANCE",
                                                                                               10)})
                    time.sleep(60)
                    continue

                # 管理现有持仓
                self.manage_open_positions()

                # 分析交易对并生成建议
                trade_candidates = []
                for symbol in self.config["TRADE_PAIRS"]:
                    try:
                        print(f"\n分析交易对: {symbol}")
                        # 获取基础数据
                        df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                        if df is None:
                            print(f"❌ 无法获取{symbol}数据")
                            continue

                        # 使用新的信号生成函数
                        signal, quality_score = self.generate_trade_signal(df, symbol)

                        # 跳过保持信号
                        if signal == "HOLD":
                            print(f"⏸️ {symbol} 保持观望")
                            continue

                        # 检查质量评分是否达到最低要求 - 新增的筛选条件
                        if quality_score < min_quality_score:
                            print_colored(
                                f"⚠️ {symbol} 质量评分 ({quality_score:.2f}) 低于最低要求 ({min_quality_score:.2f})，跳过交易",
                                Colors.YELLOW)
                            continue

                        # 检查原始信号是否为轻量级
                        is_light = False
                        # 临时获取原始信号
                        _, _, details = self.mtf_coordinator.generate_signal(symbol, quality_score)
                        raw_signal = details.get("coherence", {}).get("recommendation", "")
                        if raw_signal.startswith("LIGHT_"):
                            is_light = True
                            print_colored(f"{symbol} 检测到轻量级信号，将使用较小仓位", Colors.YELLOW)

                        # 获取当前价格
                        try:
                            ticker = self.client.futures_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                        except Exception as e:
                            print(f"❌ 获取{symbol}价格失败: {e}")
                            continue

                        # 预测未来价格
                        predicted = None
                        if "price_prediction" in details and details["price_prediction"].get("valid", False):
                            predicted = details["price_prediction"]["predicted_price"]
                        else:
                            predicted = self.predict_short_term_price(symbol, horizon_minutes=90)  # 使用90分钟预测

                        if predicted is None:
                            predicted = current_price * (1.05 if signal == "BUY" else 0.95)  # 默认5%变动

                        # 计算预期价格变动百分比
                        expected_movement = abs(predicted - current_price) / current_price * 100

                        # 使用固定的预期变动阈值: 1.35%
                        if expected_movement < 1.35:
                            print_colored(
                                f"⚠️ {symbol}的预期价格变动({expected_movement:.2f}%)小于最低要求(1.35%)，跳过交易",
                                Colors.WARNING)
                            continue

                        # 计算风险和交易金额
                        risk = expected_movement / 100  # 预期变动作为风险指标

                        # 计算交易金额时考虑轻量级信号
                        candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance)
                        if is_light:
                            candidate_amount *= 0.5  # 轻量级信号使用半仓
                            print_colored(f"{symbol} 轻量级信号，使用50%标准仓位: {candidate_amount:.2f} USDC",
                                          Colors.YELLOW)

                        # 添加到候选列表
                        candidate = {
                            "symbol": symbol,
                            "signal": signal,
                            "quality_score": quality_score,
                            "current_price": current_price,
                            "predicted_price": predicted,
                            "risk": risk,
                            "amount": candidate_amount,
                            "is_light": is_light,
                            "expected_movement": expected_movement
                        }

                        trade_candidates.append(candidate)

                        print_colored(
                            f"候选交易: {symbol} {signal}, "
                            f"质量评分: {quality_score:.2f}, "
                            f"预期波动: {expected_movement:.2f}%, "
                            f"下单金额: {candidate_amount:.2f} USDC",
                            Colors.GREEN if signal == "BUY" else Colors.RED
                        )

                    except Exception as e:
                        self.logger.error(f"处理{symbol}时出错: {e}")
                        print(f"❌ 处理{symbol}时出错: {e}")

                # 按质量评分排序候选交易
                trade_candidates.sort(key=lambda x: x["quality_score"], reverse=True)

                # 显示详细交易计划
                if trade_candidates:
                    print("\n==== 详细交易计划 ====")
                    for idx, candidate in enumerate(trade_candidates, 1):
                        symbol = candidate["symbol"]
                        signal = candidate["signal"]
                        quality = candidate["quality_score"]
                        current = candidate["current_price"]
                        predicted = candidate["predicted_price"]
                        amount = candidate["amount"]
                        is_light = candidate["is_light"]
                        expected_movement = candidate["expected_movement"]

                        side_color = Colors.GREEN if signal == "BUY" else Colors.RED
                        position_type = "轻仓位" if is_light else "标准仓位"

                        print(f"\n{idx}. {symbol} - {side_color}{signal}{Colors.RESET} ({position_type})")
                        print(f"   质量评分: {quality:.2f}")
                        print(f"   当前价格: {current:.6f}, 预测价格: {predicted:.6f}")
                        print(f"   预期波动: {expected_movement:.2f}%")
                        print(f"   下单金额: {amount:.2f} USDC")
                else:
                    print("\n本轮无交易候选")

                # 执行交易
                executed_count = 0
                max_trades = min(self.config.get("MAX_PURCHASES_PER_ROUND", 3), len(trade_candidates))

                for candidate in trade_candidates:
                    if executed_count >= max_trades:
                        break

                    symbol = candidate["symbol"]
                    signal = candidate["signal"]
                    amount = candidate["amount"]
                    quality_score = candidate["quality_score"]
                    is_light = candidate["is_light"]

                    print(f"\n🚀 执行交易: {symbol} {signal}, 金额: {amount:.2f} USDC{' (轻仓位)' if is_light else ''}")

                    # 计算适合的杠杆水平
                    leverage = self.calculate_leverage_from_quality(quality_score)
                    if is_light:
                        # 轻仓位降低杠杆
                        leverage = max(1, int(leverage * 0.7))
                        print_colored(f"轻仓位降低杠杆至 {leverage}倍", Colors.YELLOW)

                    # 执行交易
                    if self.place_futures_order_usdc(symbol, signal, amount, leverage):
                        executed_count += 1
                        print(f"✅ {symbol} {signal} 交易成功")
                    else:
                        print(f"❌ {symbol} {signal} 交易失败")

                # 显示持仓卖出预测
                self.display_position_sell_timing()

                # 打印交易循环总结
                print(f"\n==== 交易循环总结 ====")
                print(f"分析交易对: {len(self.config['TRADE_PAIRS'])}个")
                print(f"交易候选: {len(trade_candidates)}个")
                print(f"执行交易: {executed_count}个")
                print(f"最低质量评分要求: {min_quality_score:.2f}")

                # 循环间隔
                sleep_time = 60
                print(f"\n等待 {sleep_time} 秒进入下一轮...")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                self.logger.info("用户中断，程序结束")
                break
            except Exception as e:
                self.logger.error(f"交易循环异常: {e}")
                print(f"错误: {e}")
                time.sleep(30)

    def calculate_upside_potential(self, symbol, side, current_price):
        """
        计算价格上升空间，用于动态调整跟踪止损参数

        参数:
            symbol: 交易对符号
            side: 交易方向 ('BUY' 或 'SELL')
            current_price: 当前价格

        返回:
            upside_potential: 上升空间百分比 (0.0-1.0)
        """
        try:
            # 获取历史数据
            df = self.get_historical_data_with_cache(symbol)
            if df is None or len(df) < 20:
                return 0.03  # 默认上升空间3%

            # 计算指标
            df = calculate_optimized_indicators(df)
            if df is None or df.empty:
                return 0.03

            # 1. 使用多时间框架信号
            _, _, details = self.mtf_coordinator.generate_signal(symbol, 5.0)  # 使用中性评分
            coherence = details.get("coherence", {})

            # 一致性评分转换为上升空间
            coherence_score = coherence.get("coherence_score", 50) / 100

            # 根据一致性调整上升空间
            if side == "BUY" and coherence.get("dominant_trend") == "UP":
                coherence_factor = coherence_score * 0.03  # 最多贡献3%上升空间
            elif side == "SELL" and coherence.get("dominant_trend") == "DOWN":
                coherence_factor = coherence_score * 0.03
            else:
                coherence_factor = 0.01  # 无一致性时默认1%

            # 2. 分析RSI指标
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if side == "BUY" and rsi < 40:  # 买入且RSI低（超卖）
                    rsi_factor = 0.04  # 上升空间可能更大
                elif side == "SELL" and rsi > 60:  # 卖出且RSI高（超买）
                    rsi_factor = 0.04
                else:
                    rsi_factor = 0.02
            else:
                rsi_factor = 0.02

            # 3. 分析价格相对布林带位置
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Middle' in df.columns:
                bb_position = (current_price - df['BB_Lower'].iloc[-1]) / (
                            df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])

                if side == "BUY" and bb_position < 0.3:  # 靠近下轨，上升空间大
                    bb_factor = 0.05
                elif side == "SELL" and bb_position > 0.7:  # 靠近上轨，下跌空间大
                    bb_factor = 0.05
                else:
                    bb_factor = 0.02
            else:
                bb_factor = 0.02

            # 综合计算上升空间
            if side == "BUY":
                upside_potential = (coherence_factor + rsi_factor + bb_factor) / 2
            else:  # SELL - 下跌空间
                upside_potential = (coherence_factor + rsi_factor + bb_factor) / 2

            return min(upside_potential, 0.10)  # 限制在最大10%

        except Exception as e:
            self.logger.error(f"计算上升空间出错: {e}")
            return 0.03  # 默认上升空间3%

    def record_position_with_trailing_stop(self, symbol, side, entry_price, quantity,
                                           initial_stop_loss, trailing_activation, trailing_distance,
                                           expected_profit=0.0):
        """
        记录新开的持仓，使用跟踪止损系统，包含预期收益
        """
        position_side = "LONG" if side.upper() == "BUY" else "SHORT"

        # 检查是否已有同方向持仓
        for i, pos in enumerate(self.open_positions):
            if pos["symbol"] == symbol and pos.get("position_side", None) == position_side:
                # 合并持仓
                total_qty = pos["quantity"] + quantity
                new_entry = (pos["entry_price"] * pos["quantity"] + entry_price * quantity) / total_qty
                self.open_positions[i]["entry_price"] = new_entry
                self.open_positions[i]["quantity"] = total_qty
                self.open_positions[i]["last_update_time"] = time.time()

                # 更新止损设置
                self.open_positions[i]["initial_stop_loss"] = initial_stop_loss
                self.open_positions[i]["trailing_activation"] = trailing_activation
                self.open_positions[i]["trailing_distance"] = trailing_distance
                self.open_positions[i]["trailing_active"] = False
                self.open_positions[i]["highest_price"] = new_entry if position_side == "LONG" else 0
                self.open_positions[i]["lowest_price"] = new_entry if position_side == "SHORT" else float('inf')
                self.open_positions[i]["current_stop_level"] = new_entry * (
                        1 + initial_stop_loss) if position_side == "LONG" else new_entry * (1 - initial_stop_loss)

                # 保存预期收益
                self.open_positions[i]["expected_profit"] = expected_profit

                # 获取当前ATR并记录
                df = self.get_historical_data_with_cache(symbol)
                if df is not None and 'ATR' in df.columns:
                    self.open_positions[i]["entry_atr"] = df['ATR'].iloc[-1]
                else:
                    self.open_positions[i]["entry_atr"] = 0

                self.logger.info(f"更新{symbol} {position_side}持仓", extra={
                    "new_entry_price": new_entry,
                    "total_quantity": total_qty,
                    "initial_stop_loss": initial_stop_loss,
                    "trailing_activation": trailing_activation,
                    "trailing_distance": trailing_distance,
                    "entry_atr": self.open_positions[i]["entry_atr"],
                    "expected_profit": expected_profit
                })
                return

        # 计算初始止损价格
        initial_stop_price = entry_price * (1 + initial_stop_loss) if position_side == "LONG" else entry_price * (
                1 - initial_stop_loss)

        # 获取当前ATR
        entry_atr = 0
        df = self.get_historical_data_with_cache(symbol)
        if df is not None and 'ATR' in df.columns:
            entry_atr = df['ATR'].iloc[-1]

        # 添加新持仓，使用跟踪止损系统
        new_pos = {
            "symbol": symbol,
            "side": side,
            "position_side": position_side,
            "entry_price": entry_price,
            "quantity": quantity,
            "open_time": time.time(),
            "last_update_time": time.time(),
            "max_profit": 0.0,
            "initial_stop_loss": initial_stop_loss,
            "trailing_activation": trailing_activation,
            "trailing_distance": trailing_distance,
            "trailing_active": False,
            "highest_price": entry_price if position_side == "LONG" else 0,
            "lowest_price": entry_price if position_side == "SHORT" else float('inf'),
            "current_stop_level": initial_stop_price,
            "position_id": f"{symbol}_{position_side}_{int(time.time())}",
            "entry_atr": entry_atr,
            "expected_profit": expected_profit
        }

        self.open_positions.append(new_pos)
        self.logger.info(f"新增{symbol} {position_side}持仓", extra={
            **new_pos,
            "initial_stop_price": initial_stop_price,
            "expected_profit": expected_profit
        })

        print_colored(
            f"📝 新增{symbol} {position_side}持仓，初始止损: {abs(initial_stop_loss) * 100:.2f}%，" +
            f"跟踪激活阈值: {trailing_activation * 100:.2f}%，跟踪距离: {trailing_distance * 100:.2f}%，" +
            f"入场ATR: {entry_atr:.6f}，预期收益: {expected_profit:.2%}",
            Colors.GREEN + Colors.BOLD)

    def get_market_data_sync(self, symbol: str, interval: str = '5m', limit: int = 500) -> pd.DataFrame:
        """同步获取市场数据 - 完全修复版本"""
        try:
            # 使用 INFO 而不是 DEBUG
            print_colored(f"    📊 正在获取 {symbol} 的K线数据...", Colors.INFO)

            # 直接调用API
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if not klines:
                print_colored(f"    ⚠️ 未获取到数据", Colors.WARNING)
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])

            # 设置索引
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            print_colored(f"    ✅ 获取到 {len(df)} 条数据", Colors.GREEN)
            return df

        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            print_colored(f"    ❌ 错误: {str(e)}", Colors.ERROR)
            return pd.DataFrame()

    def calculate_indicators_safe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """安全计算技术指标"""
        try:
            # 检查 DataFrame
            if df.empty:
                print_colored(f"    ⚠️ DataFrame 为空，跳过指标计算", Colors.WARNING)
                return df

            # 检查必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print_colored(f"    ⚠️ 缺少必要列: {missing_columns}", Colors.WARNING)
                return df

            # 尝试导入指标模块
            try:
                from indicators_module import calculate_optimized_indicators
                df = calculate_optimized_indicators(df)
                print_colored(f"    ✅ 技术指标计算完成", Colors.SUCCESS)
            except ImportError:
                print_colored(f"    ⚠️ 指标模块不可用，使用基础计算", Colors.WARNING)
                # 基础指标计算
                df = self.calculate_basic_indicators(df)
            except Exception as e:
                print_colored(f"    ❌ 计算优化指标失败: {e}", Colors.ERROR)
                # 降级到基础指标
                df = self.calculate_basic_indicators(df)

            return df

        except Exception as e:
            print_colored(f"    ❌ 指标计算失败: {e}", Colors.ERROR)
            return df

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        try:
            # RSI
            df['RSI'] = self.calculate_rsi(df['close'], 14)

            # 移动平均线
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

            # 布林带
            df['BB_Middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()

            # ADX（简化版）
            df['ADX'] = 25  # 默认值

            print_colored(f"    ✅ 基础指标计算完成", Colors.SUCCESS)
            return df

        except Exception as e:
            print_colored(f"    ❌ 基础指标计算失败: {e}", Colors.ERROR)
            # 添加默认值
            df['RSI'] = 50
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['BB_Middle'] = df['close'].mean() if 'close' in df else 0
            df['BB_Upper'] = df['BB_Middle'] * 1.02
            df['BB_Lower'] = df['BB_Middle'] * 0.98
            return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # 填充 NaN 值
        return rsi

    def manage_open_positions(self):
        """管理现有持仓，使用改进的跟踪止损策略"""
        self.load_existing_positions()

        if not self.open_positions:
            self.logger.info("当前无持仓")
            return

        current_time = time.time()
        positions_to_remove = []  # 记录需要移除的持仓

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos.get("position_side", "LONG")
            entry_price = pos["entry_price"]

            # 获取跟踪止损参数
            initial_stop_loss = pos.get("initial_stop_loss", -0.0175)  # 默认-1.75%
            trailing_activation = pos.get("trailing_activation", 0.012)  # 默认1.2%
            trailing_distance = pos.get("trailing_distance", 0.003)  # 默认0.3%
            trailing_active = pos.get("trailing_active", False)
            highest_price = pos.get("highest_price", entry_price if position_side == "LONG" else 0)
            lowest_price = pos.get("lowest_price", entry_price if position_side == "SHORT" else float('inf'))
            current_stop_level = pos.get("current_stop_level", entry_price * (
                        1 + initial_stop_loss) if position_side == "LONG" else entry_price * (1 - initial_stop_loss))

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except Exception as e:
                print(f"⚠️ 无法获取 {symbol} 当前价格: {e}")
                continue

            # 计算盈亏百分比
            if position_side == "LONG":
                profit_pct = (current_price - entry_price) / entry_price

                # 更新最高价格
                if current_price > highest_price:
                    highest_price = current_price
                    pos["highest_price"] = highest_price

                    # 检查是否达到跟踪止损激活阈值
                    if not trailing_active and profit_pct >= trailing_activation:
                        pos["trailing_active"] = True
                        trailing_active = True
                        print_colored(
                            f"🔔 {symbol} {position_side} 激活跟踪止损 (利润: {profit_pct:.2%} >= {trailing_activation:.2%})",
                            Colors.GREEN)

                    # 更新跟踪止损价格
                    if trailing_active:
                        new_stop_level = highest_price * (1 - trailing_distance)
                        if new_stop_level > current_stop_level:
                            current_stop_level = new_stop_level
                            pos["current_stop_level"] = current_stop_level
                            print_colored(
                                f"🔄 {symbol} {position_side} 上移止损位至 {current_stop_level:.6f} (距离最高点 {trailing_distance * 100:.2f}%)",
                                Colors.CYAN)

                # 检查是否触发止损
                if current_price <= current_stop_level:
                    print_colored(
                        f"🔔 {symbol} {position_side} 触发{'跟踪' if trailing_active else '初始'}止损 ({current_price:.6f} <= {current_stop_level:.6f})",
                        Colors.YELLOW)
                    success, closed = self.close_position(symbol, position_side)
                    if success:
                        print_colored(f"✅ {symbol} {position_side} 止损平仓成功!", Colors.GREEN)
                        positions_to_remove.append(pos)
                        self.logger.info(f"{symbol} {position_side}止损平仓", extra={
                            "profit_pct": profit_pct,
                            "stop_type": "trailing" if trailing_active else "initial",
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "highest_price": highest_price
                        })
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price

                # 更新最低价格
                if current_price < lowest_price or lowest_price == 0:
                    lowest_price = current_price
                    pos["lowest_price"] = lowest_price

                    # 检查是否达到跟踪止损激活阈值
                    if not trailing_active and profit_pct >= trailing_activation:
                        pos["trailing_active"] = True
                        trailing_active = True
                        print_colored(
                            f"🔔 {symbol} {position_side} 激活跟踪止损 (利润: {profit_pct:.2%} >= {trailing_activation:.2%})",
                            Colors.GREEN)

                    # 更新跟踪止损价格
                    if trailing_active:
                        new_stop_level = lowest_price * (1 + trailing_distance)
                        if new_stop_level < current_stop_level or current_stop_level == 0:
                            current_stop_level = new_stop_level
                            pos["current_stop_level"] = current_stop_level
                            print_colored(
                                f"🔄 {symbol} {position_side} 下移止损位至 {current_stop_level:.6f} (距离最低点 {trailing_distance * 100:.2f}%)",
                                Colors.CYAN)

                # 检查是否触发止损
                if current_price >= current_stop_level and current_stop_level > 0:
                    print_colored(
                        f"🔔 {symbol} {position_side} 触发{'跟踪' if trailing_active else '初始'}止损 ({current_price:.6f} >= {current_stop_level:.6f})",
                        Colors.YELLOW)
                    success, closed = self.close_position(symbol, position_side)
                    if success:
                        print_colored(f"✅ {symbol} {position_side} 止损平仓成功!", Colors.GREEN)
                        positions_to_remove.append(pos)
                        self.logger.info(f"{symbol} {position_side}止损平仓", extra={
                            "profit_pct": profit_pct,
                            "stop_type": "trailing" if trailing_active else "initial",
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "lowest_price": lowest_price
                        })

            # 打印持仓状态
            profit_color = Colors.GREEN if profit_pct >= 0 else Colors.RED
            print_colored(
                f"{symbol} {position_side}: 当前盈亏 {profit_color}{profit_pct:.2%}{Colors.RESET}, " +
                f"{'跟踪' if trailing_active else '初始'}止损位 {current_stop_level:.6f}",
                Colors.INFO
            )

        # 从持仓列表中移除已平仓的持仓
        for pos in positions_to_remove:
            if pos in self.open_positions:
                self.open_positions.remove(pos)

        # 重新加载持仓以确保数据最新
        self.load_existing_positions()

    def active_position_monitor(self, check_interval=15):
        """
        主动监控持仓，使用改进的跟踪止损策略和最优波动区间止盈
        - 修复止损位重复激活和下降问题
        """
        print(f"🔄 启动主动持仓监控（每{check_interval}秒检查一次）")

        try:
            while True:
                # 如果没有持仓，等待一段时间后再检查
                if not self.open_positions:
                    time.sleep(check_interval)
                    continue

                # 加载最新持仓
                self.load_existing_positions()

                # 当前持仓列表的副本，用于检查
                positions = self.open_positions.copy()

                for pos in positions:
                    symbol = pos["symbol"]
                    position_side = pos.get("position_side", "LONG")
                    entry_price = pos["entry_price"]

                    # 获取当前价格
                    try:
                        ticker = self.client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])
                    except Exception as e:
                        print(f"⚠️ 获取{symbol}价格失败: {e}")
                        continue

                    # 获取跟踪止损参数
                    initial_stop_loss = pos.get("initial_stop_loss", -0.0175)
                    trailing_activation = pos.get("trailing_activation", 0.012)
                    trailing_distance = pos.get("trailing_distance", 0.003)
                    trailing_active = pos.get("trailing_active", False)  # 保持现有的激活状态
                    highest_price = pos.get("highest_price", entry_price if position_side == "LONG" else 0)
                    lowest_price = pos.get("lowest_price", entry_price if position_side == "SHORT" else float('inf'))
                    current_stop_level = pos.get("current_stop_level", entry_price * (
                            1 + initial_stop_loss) if position_side == "LONG" else entry_price * (
                                1 - initial_stop_loss))

                    # 根据持仓方向分别处理
                    if position_side == "LONG":
                        profit_pct = (current_price - entry_price) / entry_price

                        # 1. 只有在从未激活过的情况下才检查是否需要激活跟踪止损
                        if not trailing_active and profit_pct >= trailing_activation:
                            pos["trailing_active"] = True
                            trailing_active = True  # 更新局部变量
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 首次激活跟踪止损 (利润: {profit_pct:.2%})",
                                Colors.GREEN)

                        # 2. 检查是否创新高，需要更新止损位
                        if current_price > highest_price:
                            # 更新最高价记录
                            pos["highest_price"] = current_price
                            highest_price = current_price  # 更新局部变量

                            # 计算新止损位
                            new_stop_level = highest_price * (1 - trailing_distance)

                            # 止损位只上移不下移
                            if new_stop_level > current_stop_level:
                                # 保存新的止损位
                                pos["current_stop_level"] = new_stop_level
                                current_stop_level = new_stop_level  # 更新局部变量
                                print_colored(
                                    f"🔄 主动监控: {symbol} {position_side} 上移止损位至 {current_stop_level:.6f}",
                                    Colors.CYAN)
                        else:
                            # 未创新高，但要打印当前止损位
                            print_colored(f"ℹ️ {symbol} {position_side} 未创新高，维持止损位 {current_stop_level:.6f}",
                                          Colors.INFO)

                    else:  # SHORT
                        profit_pct = (entry_price - current_price) / entry_price

                        # 1. 只有在从未激活过的情况下才检查是否需要激活跟踪止损
                        if not trailing_active and profit_pct >= trailing_activation:
                            pos["trailing_active"] = True
                            trailing_active = True  # 更新局部变量
                            print_colored(
                                f"🔔 主动监控: {symbol} {position_side} 首次激活跟踪止损 (利润: {profit_pct:.2%})",
                                Colors.GREEN)

                        # 2. 检查是否创新低，需要更新止损位
                        if current_price < lowest_price or lowest_price == 0:
                            # 更新最低价记录
                            pos["lowest_price"] = current_price
                            lowest_price = current_price  # 更新局部变量

                            # 计算新止损位
                            new_stop_level = lowest_price * (1 + trailing_distance)

                            # 止损位只下移不上移
                            if new_stop_level < current_stop_level or current_stop_level == 0:
                                # 保存新的止损位
                                pos["current_stop_level"] = new_stop_level
                                current_stop_level = new_stop_level  # 更新局部变量
                                print_colored(
                                    f"🔄 主动监控: {symbol} {position_side} 下移止损位至 {current_stop_level:.6f}",
                                    Colors.CYAN)
                        else:
                            # 未创新低，但要打印当前止损位
                            print_colored(f"ℹ️ {symbol} {position_side} 未创新低，维持止损位 {current_stop_level:.6f}",
                                          Colors.INFO)

                    # 3. 检查是否触发止损
                    if (position_side == "LONG" and current_price <= current_stop_level) or \
                            (
                                    position_side == "SHORT" and current_price >= current_stop_level and current_stop_level > 0):
                        trigger_msg = f"价格: {current_price:.6f} {'<=' if position_side == 'LONG' else '>='} 止损: {current_stop_level:.6f}"
                        print_colored(
                            f"🔔 主动监控: {symbol} {position_side} 触发{'跟踪' if trailing_active else '初始'}止损 ({trigger_msg})",
                            Colors.YELLOW)
                        success, closed = self.close_position(symbol, position_side)
                        if success:
                            print_colored(f"✅ {symbol} {position_side} 止损平仓成功: {profit_pct:.2%}",
                                          Colors.GREEN)
                            self.logger.info(f"{symbol} {position_side}主动监控止损平仓", extra={
                                "profit_pct": profit_pct,
                                "stop_type": "trailing" if trailing_active else "initial",
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "price_extreme": highest_price if position_side == "LONG" else lowest_price
                            })

                # 每次检查完所有持仓后，稍微休眠以减少资源占用
                time.sleep(check_interval)

        except Exception as e:
            print(f"主动持仓监控发生错误: {e}")
            self.logger.error(f"主动持仓监控错误", extra={"error": str(e)})

            # 尝试重启监控
            print("尝试重启主动持仓监控...")
            time.sleep(5)
            self.active_position_monitor(check_interval)

    def record_open_position(self, symbol, side, entry_price, quantity, take_profit=0.025, stop_loss=-0.0175):
        """
        记录新开的持仓，转为使用跟踪止损系统替代固定止盈止损

        参数:
            symbol: 交易对符号
            side: 交易方向 ('BUY' 或 'SELL')
            entry_price: 入场价格
            quantity: 交易数量
            take_profit: 不再使用，保留参数兼容旧调用
            stop_loss: 初始止损百分比，默认-1.75%
        """
        position_side = "LONG" if side.upper() == "BUY" else "SHORT"

        # 设置跟踪止损参数
        initial_stop_loss = stop_loss  # 使用传入的止损比例
        trailing_activation = 0.012  # 默认1.2%激活阈值
        trailing_distance = 0.003  # 默认0.3%跟踪距离

        # 检查是否已有同方向持仓
        for i, pos in enumerate(self.open_positions):
            if pos["symbol"] == symbol and pos.get("position_side", None) == position_side:
                # 合并持仓
                total_qty = pos["quantity"] + quantity
                new_entry = (pos["entry_price"] * pos["quantity"] + entry_price * quantity) / total_qty
                self.open_positions[i]["entry_price"] = new_entry
                self.open_positions[i]["quantity"] = total_qty
                self.open_positions[i]["last_update_time"] = time.time()

                # 更新为跟踪止损参数（如果尚未使用）
                if "trailing_active" not in pos:
                    # 计算初始止损价格
                    if position_side == "LONG":
                        current_stop_level = new_entry * (1 + initial_stop_loss)
                        highest_price = new_entry
                    else:  # SHORT
                        current_stop_level = new_entry * (1 - initial_stop_loss)
                        lowest_price = new_entry

                    # 添加跟踪止损参数
                    self.open_positions[i]["initial_stop_loss"] = initial_stop_loss
                    self.open_positions[i]["trailing_activation"] = trailing_activation
                    self.open_positions[i]["trailing_distance"] = trailing_distance
                    self.open_positions[i]["trailing_active"] = False
                    self.open_positions[i]["highest_price"] = highest_price if position_side == "LONG" else 0
                    self.open_positions[i]["lowest_price"] = lowest_price if position_side == "SHORT" else float('inf')
                    self.open_positions[i]["current_stop_level"] = current_stop_level

                    # 移除旧的止盈止损参数
                    if "dynamic_take_profit" in self.open_positions[i]:
                        del self.open_positions[i]["dynamic_take_profit"]
                    if "stop_loss" in self.open_positions[i]:
                        del self.open_positions[i]["stop_loss"]

                    print_colored(
                        f"🔄 已将 {symbol} {position_side} 持仓转换为跟踪止损系统",
                        Colors.CYAN
                    )

                self.logger.info(f"更新{symbol} {position_side}持仓", extra={
                    "new_entry_price": new_entry,
                    "total_quantity": total_qty,
                    "initial_stop_loss": initial_stop_loss,
                    "trailing_activation": trailing_activation,
                    "trailing_distance": trailing_distance
                })
                return

        # 计算初始止损价格
        if position_side == "LONG":
            current_stop_level = entry_price * (1 + initial_stop_loss)
            highest_price = entry_price
        else:  # SHORT
            current_stop_level = entry_price * (1 - initial_stop_loss)
            lowest_price = entry_price

        # 添加新持仓，使用跟踪止损系统
        new_pos = {
            "symbol": symbol,
            "side": side,
            "position_side": position_side,
            "entry_price": entry_price,
            "quantity": quantity,
            "open_time": time.time(),
            "last_update_time": time.time(),
            "max_profit": 0.0,
            "initial_stop_loss": initial_stop_loss,
            "trailing_activation": trailing_activation,
            "trailing_distance": trailing_distance,
            "trailing_active": False,
            "highest_price": highest_price if position_side == "LONG" else 0,
            "lowest_price": lowest_price if position_side == "SHORT" else float('inf'),
            "current_stop_level": current_stop_level,
            "position_id": f"{symbol}_{position_side}_{int(time.time())}"
        }

        self.open_positions.append(new_pos)
        self.logger.info(f"新增{symbol} {position_side}持仓", extra={
            **new_pos,
            "initial_stop_loss": initial_stop_loss,
            "trailing_activation": trailing_activation,
            "trailing_distance": trailing_distance
        })

        print_colored(
            f"📝 新增{symbol} {position_side}持仓，初始止损: {abs(initial_stop_loss) * 100:.2f}%, "
            f"跟踪激活阈值: {trailing_activation * 100:.1f}%, 跟踪距离: {trailing_distance * 100:.1f}%",
            Colors.GREEN + Colors.BOLD
        )


    def close_position(self, symbol, position_side=None):
        """平仓指定货币对的持仓，并记录历史"""
        try:
            # 查找匹配的持仓
            positions_to_close = []
            for pos in self.open_positions:
                if pos["symbol"] == symbol:
                    if position_side is None or pos.get("position_side", "LONG") == position_side:
                        positions_to_close.append(pos)

            if not positions_to_close:
                print(f"⚠️ 未找到 {symbol} {position_side or '任意方向'} 的持仓")
                return False, []

            closed_positions = []
            success = False

            for pos in positions_to_close:
                pos_side = pos.get("position_side", "LONG")
                quantity = pos["quantity"]

                # 平仓方向
                close_side = "SELL" if pos_side == "LONG" else "BUY"

                print(f"📉 平仓 {symbol} {pos_side}, 数量: {quantity}")

                try:
                    # 获取精确数量
                    info = self.client.futures_exchange_info()
                    step_size = None

                    for item in info['symbols']:
                        if item['symbol'] == symbol:
                            for f in item['filters']:
                                if f['filterType'] == 'LOT_SIZE':
                                    step_size = float(f['stepSize'])
                                    break
                            break

                    if step_size:
                        precision = int(round(-math.log(step_size, 10), 0))
                        formatted_qty = f"{quantity:.{precision}f}"
                    else:
                        formatted_qty = str(quantity)

                    # 执行平仓订单
                    if hasattr(self, 'hedge_mode_enabled') and self.hedge_mode_enabled:
                        order = self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=formatted_qty,
                            positionSide=pos_side
                        )
                    else:
                        order = self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=formatted_qty,
                            reduceOnly=True
                        )

                    # 获取平仓价格
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    exit_price = float(ticker['price'])

                    # 计算盈亏
                    entry_price = pos["entry_price"]
                    if pos_side == "LONG":
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        profit_pct = (entry_price - exit_price) / entry_price * 100

                    # 记录平仓成功
                    closed_positions.append(pos)
                    success = True

                    print(f"✅ {symbol} {pos_side} 平仓成功，盈亏: {profit_pct:.2f}%")
                    self.logger.info(f"{symbol} {pos_side} 平仓成功", extra={
                        "profit_pct": profit_pct,
                        "entry_price": entry_price,
                        "exit_price": exit_price
                    })

                except Exception as e:
                    print(f"❌ {symbol} {pos_side} 平仓失败: {e}")
                    self.logger.error(f"{symbol} 平仓失败", extra={"error": str(e)})

            # 从本地持仓列表中移除已平仓的持仓
            for pos in closed_positions:
                if pos in self.open_positions:
                    self.open_positions.remove(pos)

            # 重新加载持仓以确保数据最新
            self.load_existing_positions()

            return success, closed_positions

        except Exception as e:
            print(f"❌ 平仓过程中发生错误: {e}")
            self.logger.error(f"平仓过程错误", extra={"symbol": symbol, "error": str(e)})
            return False, []

    def convert_positions_to_trailing_stop(self):
        """将现有持仓转换为使用跟踪止损策略"""
        for pos in self.open_positions:
            if "dynamic_take_profit" in pos or "stop_loss" in pos:
                # 获取旧参数
                old_take_profit = pos.get("dynamic_take_profit", 0.025)
                old_stop_loss = pos.get("stop_loss", -0.0175)

                # 设置新参数
                pos["initial_stop_loss"] = old_stop_loss
                pos["trailing_activation"] = 0.012  # 默认1.2%
                pos["trailing_distance"] = 0.003  # 默认0.3%
                pos["trailing_active"] = False
                pos["highest_price"] = pos["entry_price"] if pos["position_side"] == "LONG" else 0
                pos["lowest_price"] = pos["entry_price"] if pos["position_side"] == "SHORT" else float('inf')
                pos["current_stop_level"] = pos["entry_price"] * (1 + old_stop_loss) if pos[
                                                                                            "position_side"] == "LONG" else \
                pos["entry_price"] * (1 - abs(old_stop_loss))

                # 移除旧参数
                if "dynamic_take_profit" in pos:
                    del pos["dynamic_take_profit"]
                if "stop_loss" in pos:
                    del pos["stop_loss"]

                print(f"已将 {pos['symbol']} {pos['position_side']} 转换为跟踪止损策略")

    def display_positions_status(self):
        """显示所有持仓的状态，包括跟踪止损和最优波动区间信息"""
        if not self.open_positions:
            print("当前无持仓")
            return

        print("\n==== 当前持仓状态 ====")
        print(
            f"{'交易对':<10} {'方向':<6} {'持仓量':<10} {'开仓价':<10} {'当前价':<10} {'利润率':<8} {'止损价':<10} {'最优止盈':<10} {'完成度':<8}")
        print("-" * 110)

        current_time = time.time()

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos.get("position_side", "LONG")
            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price", 0)
            open_time = pos.get("open_time", current_time)

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except:
                current_price = 0.0

            # 计算利润率
            if position_side == "LONG":
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                profit_pct = ((entry_price - current_price) / entry_price) * 100

            # 计算持仓时间
            holding_hours = (current_time - open_time) / 3600

            # 获取止损信息
            current_stop_level = pos.get("current_stop_level", 0)

            # 获取最优波动区间信息
            max_tp_data = pos.get("max_tp_data", {})
            optimal_tp = max_tp_data.get("current_optimal_tp", {})
            completion_pct = max_tp_data.get("current_completion_pct", 0) * 100  # 转为百分比显示

            optimal_tp_price = optimal_tp.get("price", 0)

            # 根据利润率设置颜色
            profit_color = Colors.GREEN if profit_pct >= 0 else Colors.RED
            profit_str = f"{profit_color}{profit_pct:.2f}%{Colors.RESET}"

            # 完成度颜色
            comp_color = (
                Colors.GREEN + Colors.BOLD if completion_pct >= 90 else
                Colors.GREEN if completion_pct >= 70 else
                Colors.YELLOW if completion_pct >= 50 else
                Colors.RESET
            )
            comp_str = f"{comp_color}{completion_pct:.2f}%{Colors.RESET}"

            print(
                f"{symbol:<10} {position_side:<6} {quantity:<10.6f} {entry_price:<10.4f} {current_price:<10.4f} "
                f"{profit_str:<15} {current_stop_level:<10.6f} {optimal_tp_price:<10.6f} {comp_str:<8}")

        print("-" * 110)

    def get_btc_data(self):
        """专门获取BTC数据的方法"""
        try:
            # 直接从API获取最新数据，完全绕过缓存
            print("正在直接从API获取BTC数据...")

            # 尝试不同的交易对名称
            btc_symbols = ["BTCUSDT", "BTCUSDC"]

            for symbol in btc_symbols:
                try:
                    # 直接调用client.futures_klines而不是get_historical_data
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval="15m",
                        limit=30  # 获取足够多的数据点
                    )

                    if klines and len(klines) > 20:
                        print(f"✅ 成功获取{symbol}数据: {len(klines)}行")

                        # 转换为DataFrame
                        df = pd.DataFrame(klines, columns=[
                            'time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'trades',
                            'taker_base_vol', 'taker_quote_vol', 'ignore'
                        ])

                        # 转换数据类型
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                        # 转换时间
                        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')

                        print(f"BTC价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                        return df
                    else:
                        print(f"⚠️ {symbol}数据不足或为空")
                except Exception as e:
                    print(f"⚠️ 获取{symbol}数据失败: {e}")
                    continue

            # 如果所有交易对都失败，打印更多调试信息
            print("🔍 正在尝试获取可用的交易对列表...")
            try:
                # 获取可用的交易对列表
                exchange_info = self.client.futures_exchange_info()
                available_symbols = [info['symbol'] for info in exchange_info['symbols']]
                btc_symbols = [sym for sym in available_symbols if 'BTC' in sym]
                print(f"发现BTC相关交易对: {btc_symbols[:5]}...")
            except Exception as e:
                print(f"获取交易对列表失败: {e}")

            print("❌ 所有尝试获取BTC数据的方法都失败了")
            return None

        except Exception as e:
            print(f"❌ 获取BTC数据出错: {e}")
            return None


    def execute_with_retry(self, func, *args, max_retries=3, **kwargs):
        """执行函数并在失败时自动重试"""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  # 指数退避
                    print(f"操作失败，{sleep_time}秒后重试: {e}")
                    time.sleep(sleep_time)
                else:
                    print(f"操作失败，已达到最大重试次数: {e}")
                    raise

    def check_api_connection(self):
        """检查API连接状态"""
        try:
            account_info = self.client.futures_account()
            if "totalMarginBalance" in account_info:
                print("✅ API连接正常")
                return True
            else:
                print("❌ API连接异常: 返回数据格式不正确")
                return False
        except Exception as e:
            print(f"❌ API连接异常: {e}")
            return False

    def display_position_sell_timing(self):
        """显示持仓的预期卖出时机，包括止损价格"""
        if not self.open_positions:
            return

        print("\n==== 持仓卖出预测 ====")
        print(f"{'交易对':<10} {'方向':<6} {'当前价':<10} {'预测价':<10} {'止损价':<10} {'预计时间':<8}")
        print("-" * 70)

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos.get("position_side", "LONG")
            entry_price = pos.get("entry_price", 0)
            quantity = pos.get("quantity", 0)

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except:
                current_price = 0.0

            # 预测未来价格
            predicted_price = self.predict_short_term_price(symbol)
            if predicted_price is None:
                predicted_price = current_price

            # 获取止损信息
            trailing_active = pos.get("trailing_active", False)
            current_stop_level = pos.get("current_stop_level", 0)

            # 计算预计时间
            df = self.get_historical_data_with_cache(symbol)
            if df is not None and len(df) > 10:
                window = df['close'].tail(10)
                x = np.arange(len(window))
                slope, _ = np.polyfit(x, window, 1)

                if abs(slope) > 0.00001:
                    minutes_needed = abs((predicted_price - current_price) / slope) * 5
                else:
                    minutes_needed = 60
            else:
                minutes_needed = 60

            # 对非常大的时间进行限制
            if minutes_needed > 1440:  # 超过24小时
                minutes_str = ">24小时"
            else:
                minutes_str = f"{minutes_needed:.0f}分钟"

            print(
                f"{symbol:<10} {position_side:<6} {current_price:<10.4f} {predicted_price:<10.4f} "
                f"{current_stop_level:<10.4f} {minutes_str:<8}")

        print("-" * 70)


    def display_quality_scores(self):
        """显示所有交易对的质量评分"""
        print("\n==== 质量评分排名 ====")
        print(f"{'交易对':<10} {'评分':<6} {'趋势':<8} {'回测':<8} {'相似模式':<12}")
        print("-" * 50)

        scores = []
        for symbol in self.config["TRADE_PAIRS"]:
            df = self.get_historical_data_with_cache(symbol)
            if df is None:
                continue

            df = calculate_optimized_indicators(df)
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config,
                                                             self.logger)

            trend = metrics.get("trend", "NEUTRAL")

            # 获取相似度信息
            similarity_info = self.similar_patterns_history.get(symbol, {"max_similarity": 0, "is_similar": False})
            similarity_pct = round(similarity_info["max_similarity"] * 100, 1) if similarity_info[
                "is_similar"] else 0

            scores.append((symbol, quality_score, trend, similarity_pct))

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)

        for symbol, score, trend, similarity_pct in scores:
            backtest = "N/A"  # 回测暂未实现
            print(f"{symbol:<10} {score:<6.2f} {trend:<8} {backtest:<8} {similarity_pct:<12.1f}%")

        print("-" * 50)

    def _save_position_history(self):
        """保存历史持仓记录"""
        try:
            import json
            history_file = 'position_history.json'

            # 只保留最近1000条记录
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]

            with open(history_file, 'w') as f:
                json.dump(self.position_history, f, indent=2)

        except Exception as e:
            self.logger.error(f"保存持仓历史失败: {e}")

    def _test_connection(self):
        """测试与交易所的连接"""
        try:
            self.client.ping()
            server_time = self.client.get_server_time()
            print_colored("✅ 成功连接到 Binance", Colors.GREEN)
            self.logger.info(f"成功连接到Binance，服务器时间: {server_time}")
        except Exception as e:
            print_colored(f"❌ 连接失败: {e}", Colors.ERROR)
            self.logger.error(f"连接失败: {e}")
            raise


    def load_existing_positions(self):
        """加载现有持仓"""
        try:
            # 从交易所获取当前持仓
            positions = self.client.futures_position_information()

            self.open_positions = []

            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    position_data = {
                        'symbol': pos['symbol'],
                        'side': 'BUY' if float(pos['positionAmt']) > 0 else 'SELL',
                        'position_side': pos['positionSide'],
                        'quantity': abs(float(pos['positionAmt'])),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'open_time': time.time(),  # 这是近似值
                        'current_stop_level': 0,  # 需要从订单历史获取
                        'trailing_active': False,
                        'highest_price': float(pos['entryPrice']),
                        'lowest_price': float(pos['entryPrice']),
                        'entry_atr': 0  # 需要重新计算
                    }

                    self.open_positions.append(position_data)

            if self.open_positions:
                print_colored(f"📋 加载了 {len(self.open_positions)} 个现有持仓", Colors.INFO)
                for pos in self.open_positions:
                    pnl_color = Colors.GREEN if pos['unrealized_pnl'] >= 0 else Colors.RED
                    print_colored(
                        f"  - {pos['symbol']} {pos['position_side']}: "
                        f"数量={pos['quantity']}, PnL={pnl_color}{pos['unrealized_pnl']:.2f} USDT{Colors.RESET}",
                        Colors.INFO
                    )

        except Exception as e:
            self.logger.error(f"加载持仓失败: {e}")
            self.open_positions = []

    def _load_position_history(self):
        """加载历史持仓数据"""
        try:
            history_file = 'data/position_history.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.position_history = json.load(f)
                self.logger.info(f"加载了 {len(self.position_history)} 条历史记录")
        except Exception as e:
            self.logger.warning(f"加载历史记录失败: {e}")
            self.position_history = []

    def analyze_position_statistics(self):
        """分析并显示持仓统计数据，包括最优波动区间止盈效果"""
        # 基本统计
        stats = {
            "total_trades": len(self.position_history),
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "avg_holding_time": 0.0,
            "symbols": {},
            "hourly_distribution": [0] * 24,  # 24小时
            "daily_distribution": [0] * 7,  # 周一到周日
            "optimal_tp_exits": 0,  # 最优波动区间止盈次数
            "optimal_tp_profit": 0.0,  # 最优波动区间止盈盈利总和
            "reversal_exits": 0,  # 反转止盈次数
            "reversal_exit_profit": 0.0,  # 反转止盈盈利总和
            "trailing_sl_exits": 0,  # 跟踪止损次数
            "trailing_sl_profit": 0.0,  # 跟踪止损盈利总和
            "initial_sl_exits": 0,  # 初始止损次数
            "initial_sl_profit": 0.0,  # 初始止损盈利总和
            "completion_levels": {  # 完成度分布
                "0-25%": 0,
                "25-50%": 0,
                "50-75%": 0,
                "75-90%": 0,
                "90-100%": 0
            }
        }

        holding_times = []

        for pos in self.position_history:
            profit = pos.get("profit_pct", 0)
            symbol = pos.get("symbol", "unknown")
            holding_time = pos.get("holding_time", 0)  # 小时
            exit_reason = pos.get("exit_reason", "")
            completion_pct = pos.get("completion_pct", 0)

            # 按交易对统计
            if symbol not in stats["symbols"]:
                stats["symbols"][symbol] = {
                    "total": 0,
                    "wins": 0,
                    "losses": 0,
                    "profit": 0.0,
                    "loss": 0.0,
                    "optimal_tp_exits": 0,
                    "reversal_exits": 0,
                    "trailing_sl_exits": 0
                }

            stats["symbols"][symbol]["total"] += 1

            # 胜率与盈亏统计
            if profit > 0:
                stats["winning_trades"] += 1
                stats["total_profit"] += profit
                stats["symbols"][symbol]["wins"] += 1
                stats["symbols"][symbol]["profit"] += profit
            else:
                stats["losing_trades"] += 1
                stats["total_loss"] += abs(profit)
                stats["symbols"][symbol]["losses"] += 1
                stats["symbols"][symbol]["loss"] += abs(profit)

            # 时间统计
            if holding_time > 0:
                holding_times.append(holding_time)

            # 小时分布
            if "open_time" in pos:
                open_time = datetime.datetime.fromtimestamp(pos["open_time"])
                stats["hourly_distribution"][open_time.hour] += 1
                stats["daily_distribution"][open_time.weekday()] += 1

            # 出场策略统计
            if "最优止盈" in exit_reason or "最佳波动" in exit_reason:
                stats["optimal_tp_exits"] += 1
                stats["optimal_tp_profit"] += profit
                stats["symbols"][symbol]["optimal_tp_exits"] += 1
            elif "反转止盈" in exit_reason or "反转信号" in exit_reason:
                stats["reversal_exits"] += 1
                stats["reversal_exit_profit"] += profit
                stats["symbols"][symbol]["reversal_exits"] += 1
            elif "跟踪止损" in exit_reason:
                stats["trailing_sl_exits"] += 1
                stats["trailing_sl_profit"] += profit
                stats["symbols"][symbol]["trailing_sl_exits"] += 1
            elif "初始止损" in exit_reason or "止损平仓" in exit_reason:
                stats["initial_sl_exits"] += 1
                stats["initial_sl_profit"] += profit

            # 完成度分布统计
            if completion_pct < 0.25:
                stats["completion_levels"]["0-25%"] += 1
            elif completion_pct < 0.5:
                stats["completion_levels"]["25-50%"] += 1
            elif completion_pct < 0.75:
                stats["completion_levels"]["50-75%"] += 1
            elif completion_pct < 0.9:
                stats["completion_levels"]["75-90%"] += 1
            else:
                stats["completion_levels"]["90-100%"] += 1

        # 计算平均持仓时间
        if holding_times:
            stats["avg_holding_time"] = sum(holding_times) / len(holding_times)

        # 计算胜率
        if stats["total_trades"] > 0:
            stats["win_rate"] = stats["winning_trades"] / stats["total_trades"] * 100
        else:
            stats["win_rate"] = 0

        # 计算盈亏比
        if stats["total_loss"] > 0:
            stats["profit_loss_ratio"] = stats["total_profit"] / stats["total_loss"]
        else:
            stats["profit_loss_ratio"] = float('inf')  # 无亏损

        # 计算每个交易对的胜率和平均盈亏
        for symbol, data in stats["symbols"].items():
            if data["total"] > 0:
                data["win_rate"] = data["wins"] / data["total"] * 100
                data["avg_profit"] = data["profit"] / data["wins"] if data["wins"] > 0 else 0
                data["avg_loss"] = data["loss"] / data["losses"] if data["losses"] > 0 else 0
                data["net_profit"] = data["profit"] - data["loss"]

        # 计算出场策略的平均盈利
        if stats["optimal_tp_exits"] > 0:
            stats["avg_optimal_tp_profit"] = stats["optimal_tp_profit"] / stats["optimal_tp_exits"]
        else:
            stats["avg_optimal_tp_profit"] = 0.0

        if stats["reversal_exits"] > 0:
            stats["avg_reversal_profit"] = stats["reversal_exit_profit"] / stats["reversal_exits"]
        else:
            stats["avg_reversal_profit"] = 0.0

        if stats["trailing_sl_exits"] > 0:
            stats["avg_trailing_sl_profit"] = stats["trailing_sl_profit"] / stats["trailing_sl_exits"]
        else:
            stats["avg_trailing_sl_profit"] = 0.0

        if stats["initial_sl_exits"] > 0:
            stats["avg_initial_sl_profit"] = stats["initial_sl_profit"] / stats["initial_sl_exits"]
        else:
            stats["avg_initial_sl_profit"] = 0.0

        # 出场策略占比
        total_exits = (stats["optimal_tp_exits"] + stats["reversal_exits"] +
                       stats["trailing_sl_exits"] + stats["initial_sl_exits"])

        if total_exits > 0:
            stats["optimal_tp_percentage"] = (stats["optimal_tp_exits"] / total_exits) * 100
            stats["reversal_exits_percentage"] = (stats["reversal_exits"] / total_exits) * 100
            stats["trailing_sl_percentage"] = (stats["trailing_sl_exits"] / total_exits) * 100
            stats["initial_sl_percentage"] = (stats["initial_sl_exits"] / total_exits) * 100
        else:
            stats["optimal_tp_percentage"] = 0
            stats["reversal_exits_percentage"] = 0
            stats["trailing_sl_percentage"] = 0
            stats["initial_sl_percentage"] = 0

        # 输出止盈策略对比
        print("\n==== 止盈策略效果对比 ====")
        print(f"最优波动区间止盈: {stats['optimal_tp_exits']}次 ({stats['optimal_tp_percentage']:.1f}%), "
              f"平均盈利: {stats['avg_optimal_tp_profit']:.2%}")
        print(f"反转信号止盈: {stats['reversal_exits']}次 ({stats['reversal_exits_percentage']:.1f}%), "
              f"平均盈利: {stats['avg_reversal_profit']:.2%}")
        print(f"跟踪止损: {stats['trailing_sl_exits']}次 ({stats['trailing_sl_percentage']:.1f}%), "
              f"平均盈利: {stats['avg_trailing_sl_profit']:.2%}")
        print(f"初始止损: {stats['initial_sl_exits']}次 ({stats['initial_sl_percentage']:.1f}%), "
              f"平均盈利: {stats['avg_initial_sl_profit']:.2%}")

        # 输出完成度分布
        print("\n完成度分布:")
        for level, count in stats["completion_levels"].items():
            percentage = (count / stats["total_trades"]) * 100 if stats["total_trades"] > 0 else 0
            print(f"  {level}: {count}次 ({percentage:.1f}%)")

        return stats

    def generate_statistics_charts(self, stats):
        """生成统计图表"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.dates import DateFormatter

        # 确保目录存在
        charts_dir = "statistics_charts"
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # 设置样式
        plt.style.use('seaborn-v0_8-whitegrid')  # 使用兼容的样式

        # 1. 交易对胜率对比图
        plt.figure(figsize=(12, 6))
        symbols = list(stats["symbols"].keys())
        win_rates = [data["win_rate"] for data in stats["symbols"].values()]
        trades = [data["total"] for data in stats["symbols"].values()]

        # 按交易次数排序
        sorted_idx = sorted(range(len(trades)), key=lambda i: trades[i], reverse=True)
        symbols = [symbols[i] for i in sorted_idx]
        win_rates = [win_rates[i] for i in sorted_idx]
        trades = [trades[i] for i in sorted_idx]

        colors = ['green' if wr >= 50 else 'red' for wr in win_rates]

        if symbols:  # 确保有数据
            plt.bar(symbols, win_rates, color=colors)
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.7)
            plt.xlabel('交易对')
            plt.ylabel('胜率 (%)')
            plt.title('各交易对胜率对比')
            plt.xticks(rotation=45)

            # 添加交易次数标签
            for i, v in enumerate(win_rates):
                plt.text(i, v + 2, f"{trades[i]}次", ha='center')

            plt.tight_layout()
            plt.savefig(f"{charts_dir}/symbol_win_rates.png")
        plt.close()

        # 2. 日内交易分布
        plt.figure(figsize=(12, 6))
        plt.bar(range(24), stats["hourly_distribution"])
        plt.xlabel('小时')
        plt.ylabel('交易次数')
        plt.title('日内交易时间分布')
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/hourly_distribution.png")
        plt.close()

        # 3. 每周交易分布
        plt.figure(figsize=(10, 6))
        days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        plt.bar(days, stats["daily_distribution"])
        plt.xlabel('星期')
        plt.ylabel('交易次数')
        plt.title('每周交易日分布')
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/daily_distribution.png")
        plt.close()

        # 4. 交易对净利润对比
        plt.figure(figsize=(12, 6))
        sorted_symbols = sorted(stats["symbols"].items(), key=lambda x: x[1]["total"], reverse=True)
        net_profits = [data["net_profit"] for _, data in sorted_symbols]
        symbols_sorted = [s for s, _ in sorted_symbols]

        if symbols_sorted:  # 确保有数据
            colors = ['green' if np >= 0 else 'red' for np in net_profits]
            plt.bar(symbols_sorted, net_profits, color=colors)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('交易对')
            plt.ylabel('净利润 (%)')
            plt.title('各交易对净利润对比')
            plt.xticks(rotation=45)
            plt.tight_layout()
        plt.savefig(f"{charts_dir}/symbol_net_profits.png")
        plt.close()

        # 5. 盈亏分布图
        if self.position_history:
            profits = [pos.get("profit_pct", 0) for pos in self.position_history]
            plt.figure(figsize=(12, 6))
            sns.histplot(profits, bins=20, kde=True)
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('盈亏百分比 (%)')
            plt.ylabel('次数')
            plt.title('交易盈亏分布')
            plt.tight_layout()
            plt.savefig(f"{charts_dir}/profit_distribution.png")
        plt.close()

    def generate_statistics_report(self, stats):
        """生成HTML统计报告"""
        report_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>交易统计报告 - {report_time}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .stat-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .green {{ color: green; }}
                .red {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ width: 48%; margin-bottom: 20px; }}
                @media (max-width: 768px) {{ .chart {{ width: 100%; }} }}
            </style>
        </head>
        <body>
            <h1>交易统计报告</h1>
            <p>生成时间: {report_time}</p>

            <div class="stat-card">
                <h2>总体概览</h2>
                <table>
                    <tr><th>指标</th><th>数值</th></tr>
                    <tr><td>总交易次数</td><td>{stats['total_trades']}</td></tr>
                    <tr><td>盈利交易</td><td>{stats['winning_trades']} ({stats['win_rate']:.2f}%)</td></tr>
                    <tr><td>亏损交易</td><td>{stats['losing_trades']}</td></tr>
                    <tr><td>总盈利</td><td class="green">{stats['total_profit']:.2f}%</td></tr>
                    <tr><td>总亏损</td><td class="red">{stats['total_loss']:.2f}%</td></tr>
                    <tr><td>净盈亏</td><td class="{('green' if stats['total_profit'] > stats['total_loss'] else 'red')}">{stats['total_profit'] - stats['total_loss']:.2f}%</td></tr>
                    <tr><td>盈亏比</td><td>{stats['profit_loss_ratio']:.2f}</td></tr>
                    <tr><td>平均持仓时间</td><td>{stats['avg_holding_time']:.2f} 小时</td></tr>
                </table>
            </div>

            <div class="stat-card">
                <h2>交易对分析</h2>
                <table>
                    <tr>
                        <th>交易对</th>
                        <th>交易次数</th>
                        <th>胜率</th>
                        <th>平均盈利</th>
                        <th>平均亏损</th>
                        <th>净盈亏</th>
                    </tr>
        """

        # 按交易次数排序
        sorted_symbols = sorted(stats["symbols"].items(), key=lambda x: x[1]["total"], reverse=True)

        for symbol, data in sorted_symbols:
            html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{data['total']}</td>
                        <td>{data['win_rate']:.2f}%</td>
                        <td class="green">{data['avg_profit']:.2f}%</td>
                        <td class="red">{data['avg_loss']:.2f}%</td>
                        <td class="{('green' if data['net_profit'] >= 0 else 'red')}">{data['net_profit']:.2f}%</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="chart-container">
                <div class="chart">
                    <h3>交易对胜率对比</h3>
                    <img src="statistics_charts/symbol_win_rates.png" width="100%">
                </div>
                <div class="chart">
                    <h3>交易对净利润对比</h3>
                    <img src="statistics_charts/symbol_net_profits.png" width="100%">
                </div>
                <div class="chart">
                    <h3>日内交易时间分布</h3>
                    <img src="statistics_charts/hourly_distribution.png" width="100%">
                </div>
                <div class="chart">
                    <h3>每周交易日分布</h3>
                    <img src="statistics_charts/daily_distribution.png" width="100%">
                </div>
                <div class="chart">
                    <h3>交易盈亏分布</h3>
                    <img src="statistics_charts/profit_distribution.png" width="100%">
                </div>
            </div>
        </body>
        </html>
        """

        # 写入HTML文件
        with open("trading_statistics_report.html", "w") as f:
            f.write(html)

        print(f"✅ 统计报告已生成: trading_statistics_report.html")
        return "trading_statistics_report.html"

    def show_statistics(self):
        """显示交易统计信息"""
        # 加载持仓历史
        self._load_position_history()

        if not self.position_history:
            print("⚠️ 没有交易历史记录，无法生成统计")
            return

        print(f"📊 生成交易统计，共 {len(self.position_history)} 条记录")

        # 分析数据
        stats = self.analyze_position_statistics()

        # 生成图表
        self.generate_statistics_charts(stats)

        # 生成报告
        report_file = self.generate_statistics_report(stats)

        # 显示简要统计
        print("\n===== 交易统计摘要 =====")
        print(f"总交易: {stats['total_trades']} 次")
        print(f"盈利交易: {stats['winning_trades']} 次 ({stats['win_rate']:.2f}%)")
        print(f"亏损交易: {stats['losing_trades']} 次")
        print(f"总盈利: {stats['total_profit']:.2f}%")
        print(f"总亏损: {stats['total_loss']:.2f}%")
        print(f"净盈亏: {stats['total_profit'] - stats['total_loss']:.2f}%")
        print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
        print(f"平均持仓时间: {stats['avg_holding_time']:.2f} 小时")
        print(f"详细报告: {report_file}")

    def check_all_positions_status(self):
        """检查所有持仓状态，确认是否有任何持仓达到止盈止损条件，支持反转检测"""
        self.load_existing_positions()

        if not self.open_positions:
            print("当前无持仓，状态检查完成")
            return

        print("\n===== 持仓状态检查 =====")
        positions_requiring_action = []

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos.get("position_side", "LONG")
            entry_price = pos["entry_price"]
            open_time = datetime.datetime.fromtimestamp(pos["open_time"]).strftime("%Y-%m-%d %H:%M:%S")

            try:
                # 获取当前价格
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # 计算盈亏
                if position_side == "LONG":
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price

                # 获取历史数据用于反转检测
                df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                if df is not None:
                    df = calculate_optimized_indicators(df)

                    # 检测FVG
                    from fvg_module import detect_fair_value_gap
                    fvg_data = detect_fair_value_gap(df)

                    # 获取市场状态
                    from market_state_module import classify_market_state
                    market_state = classify_market_state(df)

                    # 获取趋势数据
                    trend_data = get_smc_trend_and_duration(df, None, self.logger)[2]

                    # 检查反转止盈条件
                    from risk_management import manage_take_profit
                    tp_result = manage_take_profit(pos, current_price, df, fvg_data, trend_data, market_state)

                    status = "正常"
                    action_needed = False

                    # 检查反转止盈
                    if tp_result['take_profit']:
                        status = f"⚠️ 达到反转止盈条件: {tp_result['reason']}"
                        action_needed = True
                    # 检查止损
                    elif position_side == "LONG" and current_price <= pos.get("current_stop_level", 0):
                        status = f"⚠️ 达到止损条件 ({current_price:.6f} <= {pos.get('current_stop_level', 0):.6f})"
                        action_needed = True
                    elif position_side == "SHORT" and current_price >= pos.get("current_stop_level", 0):
                        status = f"⚠️ 达到止损条件 ({current_price:.6f} >= {pos.get('current_stop_level', 0):.6f})"
                        action_needed = True

                    holding_time = (time.time() - pos["open_time"]) / 3600

                    print(f"{symbol} {position_side}: 开仓于 {open_time}, 持仓 {holding_time:.2f}小时")
                    print(f"  入场价: {entry_price:.6f}, 当前价: {current_price:.6f}, 盈亏: {profit_pct:.2%}")
                    print(
                        f"  止损: {pos.get('current_stop_level', 0):.6f}, 反转概率: {tp_result.get('reversal_probability', 0):.2f}")
                    print(f"  状态: {status}")

                    if action_needed:
                        positions_requiring_action.append((symbol, position_side, status))
                else:
                    print(f"⚠️ 无法获取 {symbol} 历史数据，无法进行反转检测")

            except Exception as e:
                print(f"检查 {symbol} 状态时出错: {e}")

        if positions_requiring_action:
            print("\n需要处理的持仓:")
            for symbol, side, status in positions_requiring_action:
                print(f"- {symbol} {side}: {status}")
        else:
            print("\n所有持仓状态正常，没有达到止盈止损条件")

        # 定义获取IP函数
        def get_public_ip():
            """获取公网IP地址"""
            try:
                response = requests.get('https://api.ipify.org?format=json', timeout=5)
                return response.json()['ip']
            except:
                try:
                    response = requests.get('https://checkip.amazonaws.com', timeout=5)
                    return response.text.strip()
                except:
                    return "无法获取IP"

        try:
            # 打印启动信息
            print_colored(f"""
            {'=' * 50}
            加密货币自动交易机器人 v{VERSION}
            模式: {'博弈论增强' if USE_GAME_THEORY else '传统技术分析'}
            时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            {'=' * 50}
            """, Colors.CYAN)

            # 检查API密钥
            if API_KEY == 'your_binance_api_key_here' or API_SECRET == 'your_binance_api_secret_here':
                print_colored("❌ 错误：请在 config.py 中设置您的 Binance API 密钥！", Colors.ERROR)
                sys.exit(1)

            # 创建客户端
            print_colored("正在连接到 Binance...", Colors.INFO)

            try:
                client = Client(API_KEY, API_SECRET)
                # 测试连接
                account = client.get_account()
                print_colored("✅ 成功连接到 Binance", Colors.GREEN)

            except Exception as e:
                # 立即显示错误和IP
                print_colored(f"❌ 连接 Binance 失败: {e}", Colors.ERROR)

                # 获取并显示IP
                current_ip = get_public_ip()
                print_colored(f"\n📍 当前机器人IP地址: {current_ip}", Colors.YELLOW)
                print_colored("\n请执行以下步骤：", Colors.CYAN)
                print_colored(f"1. 登录 Binance", Colors.WHITE)
                print_colored(f"2. 进入 API Management", Colors.WHITE)
                print_colored(f"3. 找到您的API密钥", Colors.WHITE)
                print_colored(f"4. 点击 'Edit restrictions'", Colors.WHITE)
                print_colored(f"5. 在 'Restrict access to trusted IPs only' 中添加: {current_ip}", Colors.WHITE)
                print_colored(f"6. 确保启用了 'Enable Spot & Margin Trading' 和 'Enable Futures'", Colors.WHITE)
                print_colored(f"7. 保存设置", Colors.WHITE)

                # 如果是-2015错误，给出更多信息
                if "-2015" in str(e):
                    print_colored(f"\n错误代码 -2015 通常表示：", Colors.YELLOW)
                    print_colored(f"- IP地址未在白名单中", Colors.WHITE)
                    print_colored(f"- API密钥已被删除或禁用", Colors.WHITE)
                    print_colored(f"- API权限设置不正确", Colors.WHITE)

                sys.exit(1)

            # 创建交易机器人
            print_colored("正在初始化交易机器人...", Colors.INFO)
            bot = SimpleTradingBot(client, CONFIG)  # ← 添加这一行！
            print_colored("✅ 交易机器人初始化成功", Colors.GREEN)
            # 显示配置信息
            print_colored(f"""
            配置信息:
            - 交易对: {len(TRADE_PAIRS)} 个
            - 每单金额: {ORDER_AMOUNT_PERCENT}%
            - 最大持仓: {MAX_POSITIONS} 个
            - 止盈: {TAKE_PROFIT_PERCENT}%
            - 止损: {STOP_LOSS_PERCENT}%
            - 扫描间隔: {SCAN_INTERVAL} 秒
            """, Colors.INFO)

            # 主循环
            print_colored("\n开始运行交易机器人...\n", Colors.GREEN)

            while True:
                try:
                    bot.run_trading_cycle()

                    # 等待下一个循环
                    print_colored(f"\n⏰ 等待 {SCAN_INTERVAL} 秒后进行下一次扫描...", Colors.GRAY)
                    time.sleep(SCAN_INTERVAL)

                except Exception as e:
                    print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)

                    # 再次检查是否是IP错误
                    if "-2015" in str(e):
                        current_ip = get_public_ip()
                        print_colored(f"📍 当前IP: {current_ip} (请添加到API白名单)", Colors.YELLOW)

                    logging.error(f"交易循环错误: {e}", exc_info=True)

                    # 错误后等待一段时间再继续
                    print_colored("等待60秒后重试...", Colors.YELLOW)
                    time.sleep(60)

        except KeyboardInterrupt:
            print_colored("\n\n⚠️ 接收到停止信号", Colors.YELLOW)
            print_colored("正在安全关闭交易机器人...", Colors.INFO)
            print_colored("✅ 交易机器人已安全停止", Colors.GREEN)

        except Exception as e:
            print_colored(f"\n❌ 严重错误: {e}", Colors.ERROR)

            # 最后再检查一次是否是IP问题
            if "-2015" in str(e):
                current_ip = get_public_ip()
                print_colored(f"📍 当前IP: {current_ip}", Colors.YELLOW)

            logging.error(f"主程序错误: {e}", exc_info=True)
            sys.exit(1)


class EnhancedTradingBot(SimpleTradingBot):
    """增强版交易机器人 - 集成流动性分析"""

    def __init__(self, client=None, config=None):
        super().__init__(client, config)

        # 初始化流动性猎手系统
        try:
            self.liquidity_hunter = LiquidityHunterSystem(self.client, self.logger)
            print_colored("✅ 流动性分析系统初始化成功", Colors.GREEN)
        except Exception as e:
            print_colored(f"⚠️ 流动性分析系统初始化失败: {e}", Colors.WARNING)
            self.liquidity_hunter = None

    def analyze_with_liquidity(self, symbol: str) -> Dict:
        """使用流动性分析增强交易决策"""
        if not self.liquidity_hunter:
            return {}

        try:
            # 获取流动性分析信号
            liquidity_signal = self.liquidity_hunter.generate_trading_signal(symbol)

            # 与现有系统整合
            if liquidity_signal['action'] != 'HOLD' and liquidity_signal['confidence'] > 0.6:
                return {
                    'use_liquidity_signal': True,
                    'signal': liquidity_signal,
                    'weight': 0.4  # 给流动性信号40%权重
                }

            return {
                'use_liquidity_signal': False,
                'signal': liquidity_signal,
                'weight': 0.2  # 仅作参考
            }

        except Exception as e:
            self.logger.error(f"流动性分析失败: {e}")
            return {}


def run_bot():
    """运行机器人的函数"""
    print_colored(f"{'=' * 60}", Colors.BLUE)
    print_colored(f"🚀 交易机器人 v{VERSION}", Colors.BLUE + Colors.BOLD)
    print_colored(f"{'=' * 60}", Colors.BLUE)

    try:
        # 创建Binance客户端
        print("正在连接到 Binance...")
        client = Client(API_KEY, API_SECRET)

        # 测试连接
        client.ping()
        print_colored("✅ 成功连接到 Binance", Colors.GREEN)

        # 创建机器人实例 - 这里定义bot变量！
        print("正在初始化交易机器人...")
        bot = SimpleTradingBot(client, CONFIG)
        print_colored("✅ 交易机器人初始化成功", Colors.GREEN)

        # 打印配置信息
        print_colored(f"""
            配置信息:
            - 交易对: {len(CONFIG['TRADE_PAIRS'])} 个
            - 每单金额: {CONFIG['ORDER_AMOUNT_PERCENT']}%
            - 最大持仓: {CONFIG['MAX_POSITIONS']} 个
            - 止盈: {CONFIG['TAKE_PROFIT_PERCENT']}%
            - 止损: {CONFIG['STOP_LOSS_PERCENT']}%
            - 扫描间隔: {CONFIG['SCAN_INTERVAL']} 秒
            """, Colors.INFO)

        # 运行主循环
        print_colored("开始运行交易机器人...\n", Colors.CYAN)

        # 主循环
        while True:
            try:
                # 运行一个交易循环
                bot.run_trading_cycle()

                # 等待下一个循环
                scan_interval = CONFIG.get('SCAN_INTERVAL', 300)
                print_colored(f"\n⏳ 等待 {scan_interval} 秒后进行下一轮扫描...", Colors.INFO)
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print_colored("\n⚠️ 收到中断信号，正在安全退出...", Colors.WARNING)
                break
            except Exception as e:
                print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)
                import traceback
                traceback.print_exc()
                print_colored("⏳ 30秒后重试...", Colors.WARNING)
                time.sleep(30)

    except Exception as e:
        print_colored(f"❌ 严重错误: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()
        return 1

    print_colored("👋 交易机器人已停止", Colors.INFO)

# ==================== 主程序入口 ====================
def main():
    """主程序入口"""
    print_colored(f"{'=' * 60}", Colors.BLUE)
    print_colored(f"🚀 交易机器人 v{VERSION}", Colors.BLUE + Colors.BOLD)
    print_colored(f"{'=' * 60}", Colors.BLUE)

    try:
        # 创建Binance客户端
        print("正在连接到 Binance...")
        client = Client(API_KEY, API_SECRET)

        # 测试连接
        client.ping()
        print_colored("✅ 成功连接到 Binance", Colors.GREEN)

        # 创建机器人实例 - 注意是小写的 bot
        print("正在初始化交易机器人...")
        bot = SimpleTradingBot(client, CONFIG)  # 小写 bot！
        print_colored("✅ 交易机器人初始化成功", Colors.GREEN)

        # 打印配置信息
        print_colored(f"""
            配置信息:
            - 交易对: {len(CONFIG['TRADE_PAIRS'])} 个
            - 每单金额: {CONFIG['ORDER_AMOUNT_PERCENT']}%
            - 最大持仓: {CONFIG['MAX_POSITIONS']} 个
            - 止盈: {CONFIG['TAKE_PROFIT_PERCENT']}%
            - 止损: {CONFIG['STOP_LOSS_PERCENT']}%
            - 扫描间隔: {CONFIG['SCAN_INTERVAL']} 秒
            """, Colors.INFO)

        print_colored("开始运行交易机器人...\n", Colors.CYAN)

        # 主循环
        while True:
            try:
                # 确保使用小写的 bot
                bot.run_trading_cycle()  # 小写 bot！

                # 等待下一个循环
                print_colored(f"\n⏳ 等待 {CONFIG['SCAN_INTERVAL']} 秒后进行下一轮扫描...", Colors.INFO)
                time.sleep(CONFIG['SCAN_INTERVAL'])

            except KeyboardInterrupt:
                print_colored("\n⚠️ 收到中断信号，正在安全退出...", Colors.WARNING)
                break
            except Exception as e:
                print_colored(f"❌ 交易循环错误: {e}", Colors.ERROR)
                import traceback
                traceback.print_exc()
                print_colored("⏳ 60秒后重试...", Colors.WARNING)
                time.sleep(60)

    except Exception as e:
        print_colored(f"❌ 严重错误: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()
        return 1

    print_colored("👋 交易机器人已停止", Colors.INFO)
    return 0


# 程序入口
if __name__ == "__main__":
    # 确保异步支持
    import nest_asyncio

    nest_asyncio.apply()

    # 运行主程序
    sys.exit(main())