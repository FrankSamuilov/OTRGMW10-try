import pandas as pd


def get_historical_data(client, symbol, interval='15m', limit=100):
    """
    获取历史K线数据

    参数:
        client: Binance客户端
        symbol: 交易对
        interval: K线间隔，默认15m
        limit: 获取数量，默认100
    """
    try:
        # 获取K线数据
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"获取历史数据失败 {symbol}: {e}")
        return None

def get_spot_balance(client):
    try:
        info = client.get_asset_balance(asset="USDC")
        return float(info["free"])
    except Exception as e:
        print(f"错误：获取现货余额失败 - {e}")
        return 0.0

def get_futures_balance(client):
    try:
        assets = client.futures_account_balance()
        for asset in assets:
            if asset["asset"] == "USDC":
                return float(asset["balance"])
        return 0.0
    except Exception as e:
        print(f"错误：获取期货余额失败 - {e}")
        return 0.0