#!/usr/bin/env python3
# run_bot_sync.py
# 使用安全的同步方式运行交易机器人

import sys
import time
from datetime import datetime
from logger_utils import Colors, print_colored

# 导入模块
from simple_trading_bot import SimpleTradingBot
from safe_analyzer import patch_bot_instance


def main():
    """主函数"""
    print_colored(f"{'=' * 60}", Colors.BLUE)
    print_colored(f"🚀 启动交易机器人 (安全同步版本)", Colors.BLUE + Colors.BOLD)
    print_colored(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.INFO)
    print_colored(f"{'=' * 60}", Colors.BLUE)

    try:
        # 创建并打补丁
        print_colored("初始化交易机器人...", Colors.INFO)
        bot = SimpleTradingBot()
        bot = patch_bot_instance(bot)
        print_colored("✅ 机器人初始化成功（已应用安全补丁）", Colors.GREEN)

        # 运行主循环
        print_colored("\n📊 开始运行交易循环...", Colors.INFO)

        while True:
            try:
                bot.run_trading_cycle()

                # 等待下一轮
                wait_time = 300  # 5分钟
                print_colored(f"\n⏳ 等待 {wait_time} 秒后进行下一轮扫描...", Colors.INFO)
                time.sleep(wait_time)

            except KeyboardInterrupt:
                print_colored("\n⚠️ 收到中断信号，正在安全退出...", Colors.WARNING)
                break
            except Exception as e:
                print_colored(f"\n❌ 交易循环错误: {e}", Colors.ERROR)
                print_colored("⏳ 30秒后重试...", Colors.WARNING)
                time.sleep(30)

    except Exception as e:
        print_colored(f"\n❌ 严重错误: {e}", Colors.ERROR)
        import traceback
        traceback.print_exc()
        return 1

    print_colored("\n👋 交易机器人已停止", Colors.INFO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
