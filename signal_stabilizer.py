import numpy as np
import pandas as pd
import talib
from logger_utils import Colors, print_colored

class SignalStabilizer:
    """信号稳定器 - 减少虚假信号"""

    def __init__(self, smoothing_window=5, confirmation_periods=3, min_holding_periods=5):
        self.smoothing_window = smoothing_window
        self.confirmation_periods = confirmation_periods
        self.min_holding_periods = min_holding_periods
        self.signal_history = []
        self.position_state = 0  # -1: 做空, 0: 中性, 1: 做多
        self.last_position_time = None
        self.position_hold_count = 0
        print_colored("✅ 信号稳定系统初始化完成", Colors.GREEN)

    def apply_kalman_filter(self, signal, Q=0.0001, R=1.0):
        """卡尔曼滤波器用于信号平滑"""
        if not hasattr(self, 'kalman_state'):
            self.kalman_state = signal
            self.kalman_covariance = 1.0

        # 预测步骤
        predicted_state = self.kalman_state
        predicted_covariance = self.kalman_covariance + Q

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + R)
        self.kalman_state = predicted_state + kalman_gain * (signal - predicted_state)
        self.kalman_covariance = (1 - kalman_gain) * predicted_covariance

        return self.kalman_state

    def apply_hysteresis(self, signal, upper_threshold=50, lower_threshold=-50):
        """使用迟滞带防止信号震荡"""
        old_state = self.position_state

        if self.position_state == 0:  # 中性
            if signal > upper_threshold:
                self.position_state = 1
            elif signal < lower_threshold:
                self.position_state = -1
        elif self.position_state == 1:  # 做多
            if signal < lower_threshold * 0.8:  # 需要更强的信号才能反转
                self.position_state = -1
        elif self.position_state == -1:  # 做空
            if signal > upper_threshold * 0.8:
                self.position_state = 1

        # 如果状态改变，重置持仓计数
        if old_state != self.position_state and self.position_state != 0:
            self.position_hold_count = 0
            print_colored(f"    ⚡ 信号状态变化: {'做多' if self.position_state == 1 else '做空'}", Colors.YELLOW)

        return self.position_state

    def check_position_protection(self):
        """检查是否在持仓保护期内"""
        self.position_hold_count += 1

        if self.position_state != 0 and self.position_hold_count < self.min_holding_periods:
            print_colored(f"    🛡️ 持仓保护期: {self.position_hold_count}/{self.min_holding_periods}", Colors.YELLOW)
            return True
        return False

    def confirm_signal(self, raw_signal):
        """多阶段信号确认"""
        # 阶段1: 卡尔曼滤波
        smoothed_signal = self.apply_kalman_filter(raw_signal)

        # 阶段2: 添加到历史并检查持续性
        self.signal_history.append(smoothed_signal)
        if len(self.signal_history) > self.confirmation_periods:
            self.signal_history.pop(0)

        # 阶段3: 要求方向一致
        if len(self.signal_history) >= self.confirmation_periods:
            signs = [np.sign(s) for s in self.signal_history]
            if all(s == signs[0] for s in signs):
                # 所有信号方向一致
                avg_strength = np.mean([abs(s) for s in self.signal_history])
                confirmed_signal = signs[0] * avg_strength
                print_colored(f"    ✅ 信号确认: 方向一致 (强度: {avg_strength:.1f})", Colors.GREEN)
            else:
                confirmed_signal = 0  # 无共识
                print_colored(f"    ❌ 信号不一致，保持观望", Colors.YELLOW)
        else:
            confirmed_signal = 0
            print_colored(f"    ⏳ 等待更多确认 ({len(self.signal_history)}/{self.confirmation_periods})", Colors.GRAY)

        # 检查持仓保护
        if self.check_position_protection():
            return self.position_state, 0  # 保持当前仓位，不产生新信号

        # 阶段4: 应用迟滞
        final_position = self.apply_hysteresis(confirmed_signal)

        return final_position, confirmed_signal
