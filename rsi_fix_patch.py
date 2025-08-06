
# RSI 传递修复补丁
def fix_rsi_in_technical_analysis(df):
    """确保 RSI 正确传递"""
    # 获取实际的 RSI 值
    actual_rsi = 50  # 默认值

    if 'RSI' in df.columns and not df['RSI'].empty:
        rsi_series = df['RSI'].dropna()
        if len(rsi_series) > 0:
            actual_rsi = float(rsi_series.iloc[-1])
            print(f"[DEBUG] 获取到实际 RSI: {actual_rsi:.2f}")
        else:
            print("[DEBUG] RSI 列全是 NaN")
    else:
        print("[DEBUG] RSI 列不存在")

    return actual_rsi
