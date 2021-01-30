# import libraries
import concurrent.futures
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta

from chart import chart


class RsiOptimizer(object):
    """
    Note:
    RSI売買を仮想で行い、パラメータをグリッドサーチで最適化
    パラメータ
        :span       RSIのspan
        :buy_thres  購入する閾値
        :sell_thres 売る閾値
    """
    def __init__(self):
        self._buy_thres_low = None
        self._buy_thres_high = None
        self._sell_thres_low = None
        self._sell_thres_high = None
        self._span_low = None
        self._span_high = None

        self.df = None

        self._best_parms = {
            "span": 0,
            "buy_thres": 0,
            "sell_thres": 0,
            "profit": 0
        }

    def set_params(self,
                   spans: List[int],
                   buy_thres: List[int],
                   sell_thres: List[int]):
        """
        set param
        Args:
            buy_thres(List)  : RSI buy limit[low, high]
            sell_thres(List) : RSI sell limit[low, high]
            span_thres(List) : RSI span range[low, high]
        """
        self._span_low = spans[0]
        self._span_high = spans[1]
        self._buy_thres_low = buy_thres[0]
        self._buy_thres_high = buy_thres[1]
        self._sell_thres_low = sell_thres[0]
        self._sell_thres_high = sell_thres[1]

    def _calculate_rsi_profit(self, df, span: int, low_thres: int, high_thres: int) -> float:
        """
        inner function
        RSIで売買し、その結果利損益を返す
        Args:
            df(pandas Dataframe)   : stock data
            span(int)              : RSI span
        Return:
            pl_amount_retio(float) : profit and loss from RSI
        """

        _df = df.copy()

        # set RSI cal data
        _df["RSI"] = ta.RSI(_df["Close"], span)
        _df["transaction"] = None

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset
        last_close_price = 0

        for i in range(len(_df) - 1):

            # buy
            if _df["RSI"][i] > low_thres and _df["RSI"][i - 1] < low_thres:
                if asset != 0:
                    last_close_price = asset
                    stock = asset / _df["Open"][i + 1]
                    asset = 0
                    _df["transaction"][i + 1] = "buy"
                    continue

            # sell
            if _df["RSI"][i] < high_thres and _df["RSI"][i - 1] > high_thres:
                if stock != 0:
                    last_close_price = 0
                    asset = stock * _df["Open"][i + 1]
                    stock = 0
                    _df["transaction"][i + 1] = "sell"
                    continue

        # If you still have stock, sell it
        last_asset = asset + last_close_price
        pl_amount_retio = last_asset / start_asset * 100
        return pl_amount_retio, _df

    def run(self, df):
        """
        Note: 解析開始コマンド
        Args:
            df(pandas dataframe)     : stock data
        Return:
            result(pandas Dataframe) : 解析結果
        """

        result = []
        _df = df.copy()

        def inner_thread(span):

            sys.stdout.write(f"Thread START RSI Span={span}\n")

            for buy in list(range(self._buy_thres_low, self._buy_thres_high + 1, 1)):
                for sell in list(range(self._sell_thres_low, self._sell_thres_high + 1, 1)):

                    profit, r_df = self._calculate_rsi_profit(_df, span, buy, sell)

                    result.append([span, buy, sell, profit])

                    if profit > self._best_parms["profit"]:
                        self.df = r_df.copy()
                        self._best_parms["span"] = span
                        self._best_parms["buy_thres"] = buy
                        self._best_parms["sell_thres"] = sell
                        self._best_parms["profit"] = round(profit, 2)
                        self.df["buy_thres"] = buy
                        self.df["sell_thres"] = sell
            return

        # concurrent.futuresでマルチスレッド(並列計算)処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            res = executor.map(
                inner_thread,
                [j for j in range(self._span_low, self._span_high + 1, 1)]
            )

        print("\n計算終了")

        return pd.DataFrame(result, columns=["span", "buy", "sell", "profit"]).sort_values("profit", ascending=False)

    def result_params(self):
        """
        Note: 解析結果パラメタを表示
        """
        return self._best_parms

    def result_graph(self):
        """
        Note: matplotlibで解析結果グラフを表示
        """

        chart.candlestick(self.df)

        fig = plt.figure(figsize=(12, 4))
        plt.subplot(111)
        plt.plot(ta.RSI(self.df["Close"], self._best_parms["span"]), marker="o")
        plt.plot(self.df["buy_thres"], linestyle="dashed", color="red")
        plt.plot(self.df["sell_thres"], linestyle="dashed", color="blue")
        plt.ylim([0, 100])
        plt.ylabel("RSI[%]")
        plt.legend([f"RSI {self._best_parms['span']}", f"Buy signal {self._best_parms['buy_thres']}", f"Sell signal {self._best_parms['sell_thres']}"])
        plt.grid()

        plt.tight_layout()
        fig.show()
