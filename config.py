"""
交易机器人配置文件
"""
import os
from datetime import datetime
import os
from dotenv import load_dotenv
# ==================== 版本信息 ====================
VERSION = "2.0.0"
RELEASE_DATE = "2025-06-19"
# ==================== API配置 ====================
# 在这里填写您的Binance API密钥
API_KEY = 'm5tGGRn3rRFIfeGkLCCitenWpkHDHOa5MIKZoTJVbrkmvSH6x9Q24ieBEYAVgKGT'
API_SECRET = '8MYXtWWBNGPLC5UJLeLkCb3a0ZeV8L5euaQUUDgfgmLvIUC7w5gnL9gYZiCp4pa1'

# ==================== 交易配置 ====================
# 交易对列表
TRADE_PAIRS = ["XRPUSDT","BBUSDT","DOGEUSDT","SUSDT","BNBUSDT","VINEUSDT","XPRUSDT","SUIUSDT","BNBUSDT","DOGEUSDT","ETHUSDC","1000PEPEUSDT","IPUSDT","SOLUSDT","AIXBTUSDT"]

# 交易参数
MIN_MARGIN_BALANCE = 50  # 最小保证金余额 (USDT)
ORDER_AMOUNT_PERCENT = 5  # 每笔订单占总余额的百分比
MAX_POSITIONS = 5  # 最大同时持仓数
MAX_DAILY_TRADES = 100000  # 每日最大交易次数
TAKE_PROFIT_PERCENT = 3  # 止盈百分比
STOP_LOSS_PERCENT = 2  # 止损百分比

# ==================== 博弈论配置 ====================
USE_GAME_THEORY = True  # 是否启用博弈论系统
MIN_GAME_THEORY_CONFIDENCE = 0.5  # 博弈论最小置信度

GAME_THEORY_CONFIG = {
    # 数据源配置
    "USE_LONG_SHORT_RATIO": True,
    "USE_FUNDING_RATE": True,
    "USE_LIQUIDATION_DATA": True,
    "USE_ORDER_BOOK": True,

    # SMC配置
    "FVG_MIN_GAP": 0.001,  # 最小FVG缺口 (0.1%)
    "ORDER_BLOCK_STRENGTH": 2.0,  # 订单块强度阈值
    "LIQUIDITY_ZONE_THRESHOLD": 0.02,  # 流动性区域阈值 (2%)

    # 多空比阈值
    "EXTREME_LONG_RATIO": 2.0,  # 极度看多阈值
    "EXTREME_SHORT_RATIO": 0.5,  # 极度看空阈值
    "SMART_RETAIL_DIVERGENCE": 0.3,  # 聪明钱与散户分歧阈值

    # 操纵检测阈值
    "MANIPULATION_THRESHOLD": 0.7,  # 操纵概率阈值
    "SPOOFING_DETECTION": True,  # 是否检测幌骗
    "ICEBERG_DETECTION": True,  # 是否检测冰山订单

    # 风险参数
    "MAX_MANIPULATION_TRADE": 0.7,  # 高操纵环境最大仓位系数
    "STOP_HUNT_BUFFER": 0.005,  # 止损猎杀缓冲区 (0.5%)
    "TIGHT_STOP_LOSS": 0.015,  # 收紧的止损 (1.5%)
    "NORMAL_STOP_LOSS": 0.02,  # 正常止损 (2%)

    # 执行参数
    "USE_ICEBERG_ORDERS": False,  # 使用冰山订单（需要交易所支持）
    "RANDOMIZE_ENTRY": True,  # 随机化入场时机
    "SPLIT_ORDERS": True,  # 分批下单
    "MAX_ORDER_SPLITS": 3,  # 最大分单数
}

# 添加到CONFIG字典

# ==================== 持仓可视化配置 ====================
POSITION_VISUALIZATION = {
    'enabled': True,
    'refresh_interval': 5,     # 刷新间隔（秒）
    'show_liquidity': True,    # 显示流动性预测
    'show_charts': False,      # 生成图表（需要matplotlib）
    'price_decimals': 4,       # 价格显示小数位
    'pct_decimals': 2,         # 百分比显示小数位
}


# ==================== 增强评分系统配置 ====================
ENHANCED_SCORING_CONFIG = {
    'use_enhanced_scoring': True,
    'trade_thresholds': {
        'strong_buy': 2.5,
        'buy': 1.5,
        'strong_sell': -2.5,
        'sell': -1.5
    },
    'volume_spike_config': {
        'timeframe': '15m',
        'lookback_periods': 20,
        'spike_threshold': 1.5,
    }
}

# 修改现有的USE_TREND_PRIORITY为True
USE_TREND_PRIORITY = True  # 确保使用趋势优先的整合方法
# ==================== 流动性感知止损配置 ====================
LIQUIDITY_STOP_LOSS = {
    'enabled': True,
    'activation_threshold': 0.618,  # 0.618%激活移动止损
    'check_interval': 60,           # 60秒检查一次流动性
    'high_liquidity_multiplier': 1.5,  # 高流动性倍数
    'max_adjustment': 0.3,          # 最大调整30%
}


# ==================== 风险管理 ====================
MAX_POSITION_SIZE_PERCENT = 30  # 单个仓位最大占比
MAX_DAILY_LOSS_PERCENT = 5  # 每日最大亏损百分比
MAX_DRAWDOWN_PERCENT = 10  # 最大回撤百分比

# ==================== 其他配置 ====================
SCAN_INTERVAL = 120  # 扫描间隔（秒）
MIN_SCORE = 6.0  # 传统模式最小评分
MAX_CONCURRENT_TRADES = 5  # 最大并发交易数

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE_PATH = "logs/trading_bot.log"
# ==================== 统一配置字典（为了兼容性） ====================
CONFIG = {
    # API
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,

    # 交易
    "TRADE_PAIRS": TRADE_PAIRS,
    "MIN_MARGIN_BALANCE": MIN_MARGIN_BALANCE,
    "ORDER_AMOUNT_PERCENT": ORDER_AMOUNT_PERCENT,
    "MAX_POSITIONS": MAX_POSITIONS,
    "MAX_DAILY_TRADES": MAX_DAILY_TRADES,
    "TAKE_PROFIT_PERCENT": TAKE_PROFIT_PERCENT,
    "STOP_LOSS_PERCENT": STOP_LOSS_PERCENT,
    "LIQUIDITY_STOP_LOSS": LIQUIDITY_STOP_LOSS,

    # 博弈论
    "USE_GAME_THEORY": USE_GAME_THEORY,
    "MIN_GAME_THEORY_CONFIDENCE": MIN_GAME_THEORY_CONFIDENCE,
    "GAME_THEORY_CONFIG": GAME_THEORY_CONFIG,

    # 风险管理
    "MAX_POSITION_SIZE_PERCENT": MAX_POSITION_SIZE_PERCENT,
    "MAX_DAILY_LOSS_PERCENT": MAX_DAILY_LOSS_PERCENT,
    "MAX_DRAWDOWN_PERCENT": MAX_DRAWDOWN_PERCENT,

    # 其他
    "SCAN_INTERVAL": SCAN_INTERVAL,
    "MIN_SCORE": MIN_SCORE,
    "MAX_CONCURRENT_TRADES": MAX_CONCURRENT_TRADES,
    "POSITION_VISUALIZATION":    POSITION_VISUALIZATION,

    # 日志
    "LOG_LEVEL": LOG_LEVEL,
    "LOG_TO_FILE": LOG_TO_FILE,
    "LOG_FILE_PATH": LOG_FILE_PATH,

}