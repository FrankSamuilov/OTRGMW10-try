# main_sync.py
# 同步版本的主程序入口

import sys
import time
from datetime import datetime
from logger_utils import Colors, print_colored

# 导入必要的模块
from simple_trading_bot import SimpleTradingBot
from sync_wrapper import ensure_sync_execution


def main():
    """主函数 - 完全同步版本"""
    print_colored(f"{'=' * 60}", Colors.BLUE)
    print_colored(f"🚀 启动交易机器人 (同步版本)", Colors.BLUE + Colors.BOLD)
    print_colored(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.INFO)
    print_colored(f"{'=' * 60}", Colors.BLUE)

    try:
        # 创建机器人实例
        print_colored("初始化交易机器人...", Colors.INFO)
        bot = SimpleTradingBot()

        # 确保同步执行
        sync_bot = ensure_sync_execution(bot)

        print_colored("✅ 机器人初始化成功", Colors.GREEN)

        # 运行主循环
        print_colored("\n📊 开始运行交易循环...", Colors.INFO)

        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                print_colored(f"\n🔄 第 {cycle_count} 轮扫描开始", Colors.CYAN)

                # 运行一个交易循环
                sync_bot.run_trading_cycle()

                # 等待下一轮
                wait_time = 300  # 5分钟
                print_colored(f"\n⏳ 等待 {wait_time} 秒后进行下一轮扫描...", Colors.INFO)

                # 显示倒计时
                for remaining in range(wait_time, 0, -30):
                    print(f"\r剩余时间: {remaining} 秒", end='', flush=True)
                    time.sleep(min(30, remaining))

            except KeyboardInterrupt:
                print_colored("\n\n⚠️ 收到中断信号，正在安全退出...", Colors.WARNING)
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
