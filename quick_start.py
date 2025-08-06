#!/usr/bin/env python
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
