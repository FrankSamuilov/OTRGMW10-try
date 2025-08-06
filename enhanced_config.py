
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
