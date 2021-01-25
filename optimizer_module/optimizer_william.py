# import libraries
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta


class WilliamrOptimizer(object):
    """
    Note:
    William %R売買を仮想で行い、パラメータをグリッドサーチで最適化
    パラメータ
        :span       William %Rのspan
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
            buy_thres(List)  : William %R buy limit[low, high]
            sell_thres(List) : William %R sell limit[low, high]
            span_thres(List) : William %R span range[low, high]
        """
        self._span_low = spans[0]
        self._span_high = spans[1]
        self._buy_thres_low = buy_thres[0]
        self._buy_thres_high = buy_thres[1]
        self._sell_thres_low = sell_thres[0]
        self._sell_thres_high = sell_thres[1]

    def _calculate_willr_profit(self, df, span: int, low_thres: int, high_thres: int) -> float:
        """
        inner function
        William %Rで売買し、その結果利損益を返す
        Args:
            df(pandas Dataframe)   : stock data
            span(int)              : William %R span
        Return:
            pl_amount_retio(float) : profit and loss from RSI
        """

        # set William cal data
        df["Willr"] = ta.WILLR(df["High"], df["Low"], df["Close"], span)

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset

        for i in range(len(df)):

            # buy
            if df["Willr"][i] > low_thres and df["Willr"][i - 1] < low_thres:
                if asset != 0:
                    stock = asset / df["Close"][i]
                    asset = 0
                # print(f"buy : {round(df['Willr'][i], 1)} -> {round(df['Willr'][i-1], 1): 5} | stock: {round(stock, 2)}")

            # sell
            if df["Willr"][i] < high_thres and df["Willr"][i - 1] > high_thres:
                if stock != 0:
                    asset = stock * df["Close"][i]
                    stock = 0
                # print(f"sell: {round(df['Willr'][i], 1)} -> {round(df['Willr'][i-1], 1): 5} | asset: {round(asset, 2)}")

        # If you still have stock, sell it
        last_asset = asset + stock * df["Close"][-1]
        pl_amount_retio = last_asset / start_asset * 100
        # print(f"PL Amount: {round(pl_amount_retio, 2)}")
        return pl_amount_retio

    def run(self, df):
        """
        Note: 解析開始コマンド
        Args:
            df(pandas dataframe)     : stock data
        Return:
            result(pandas Dataframe) : 解析結果
        """

        self.df = df

        result = []

        # 全パラメータをグリッドサーチ(計算量: Ο[n^3])
        for i, span in enumerate(list(range(self._span_low, self._span_high, 1))):
            for buy in list(range(self._buy_thres_low, self._buy_thres_high, 1)):
                for sell in list(range(self._sell_thres_low, self._sell_thres_high, 1)):

                    sys.stdout.write(f"\rWilliam %R [{i}/{len(list(range(self._span_low, self._span_high, 1)))-1}] 計算中...")
                    profit = self._calculate_willr_profit(df, span, buy, sell)

                    result.append([span, buy, sell, profit])

                    if profit > self._best_parms["profit"]:
                        self._best_parms["span"] = span
                        self._best_parms["buy_thres"] = buy
                        self._best_parms["sell_thres"] = sell
                        self._best_parms["profit"] = round(profit, 2)

        print("\n計算終了")

        return pd.DataFrame(result, columns=["span", "buy", "sell", "profit"])

    def result_params(self):
        """
        Note: 解析結果パラメタを表示
        """
        return self._best_parms

    def result_graph(self):
        """
        Note: matplotlibで解析結果グラフを表示
        """

        self.df["buy_thres"] = self._best_parms['buy_thres']
        self.df["sell_thres"] = self._best_parms['sell_thres']

        fig = plt.figure(figsize=(20, 8))

        plt.subplot(211)
        plt.title(f"William %R trade (profit: {self._best_parms['profit']})")
        plt.plot(self.df["Close"], marker="o")
        plt.ylabel("Stock Price[Yen]")
        plt.grid()

        plt.subplot(212)
        plt.plot(ta.WILLR(self.df["High"], self.df["Low"], self.df["Close"], self._best_parms["span"]), marker="o", color="orange")
        plt.plot(self.df["buy_thres"], linestyle="dashed", color="red")
        plt.plot(self.df["sell_thres"], linestyle="dashed", color="blue")
        plt.ylabel(f"William %R[%]")
        plt.legend([f"WilliamR {self._best_parms['span']}", f"Buy signal {self._best_parms['buy_thres']}", f"Sell signal {self._best_parms['sell_thres']}"])
        plt.grid()

        plt.tight_layout()
        fig.show()
