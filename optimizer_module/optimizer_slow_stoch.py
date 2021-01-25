# import libraries
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta


class SlowStochOptimizer(object):
    """
    Note:
    スローストキャスティクス売買を仮想で行い、パラメータをグリッドサーチで最適化
    パラメータ
        :fastK
        :slowK
        :slowD
    """
    def __init__(self):

        self.df = None

        self._fastk_low = None
        self._fastk_high = None
        self._slowk_low = None
        self._slowk_high = None
        self._slowd_low = None
        self._slowd_high = None

        self._best_parms = {
            "fastk": 0,
            "slowk": 0,
            "slowd": 0,
            "profit": 0
        }

    def _calculate_slow_stoch_profit(self, df, fastk_period: int, slowk_period: int, slowd_period: int) -> float:
        stoch = ta.STOCH(df["High"], df["Low"], df["Close"],
                         fastk_period=fastk_period,
                         slowk_period=slowd_period,
                         slowd_period=slowk_period)

        df = pd.DataFrame([df["Close"].to_list(), stoch[0].to_list(), stoch[1].to_list()],
                          columns=stoch[0].index,
                          index=["Close", "slowK", "slowD"]).T

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset

        for i in range(len(df)):
            # buy
            if df["slowD"][i] > df["slowK"][i] and df["slowD"][i - 1] < df["slowK"][i - 1]:
                if asset != 0:
                    stock = asset / df["Close"][i]
                    asset = 0
                # print(f"buy ({df.index[i]}) : {round(df['slowK'][i-1], 1)}/{round(df['slowD'][i-1], 1)} -> {round(df['slowK'][i], 1)}/{round(df['slowD'][i], 1)} | stock: {round(stock, 2)}")
            # sell
            if df["slowD"][i] < df["slowK"][i] and df["slowD"][i - 1] > df["slowK"][i - 1]:
                if stock != 0:
                    asset = stock * df["Close"][i]
                    stock = 0
                # print(f"sell({df.index[i]}) : {round(df['slowK'][i-1], 1)}/{round(df['slowD'][i-1], 1)} -> {round(df['slowK'][i], 1)}/{round(df['slowD'][i], 1)} | asset: {round(asset, 2)}")

        # If you still have stock, sell it
        last_asset = asset + stock * df["Close"][-1]
        pl_amount_retio = last_asset / start_asset * 100
        # print(f"PL Amount: {round(pl_amount_retio, 2)}")

        return pl_amount_retio

    def set_params(self,
                   fastk: List[int],
                   slowk: List[int],
                   slowd: List[int]):
        """
        set param
        Args:
            fastk(List) : fast K range[low, high]
            slowk(List) : slow K range[low, high]
            slowd(List) : slow D range[low, high]
        """
        self._fastk_low = fastk[0]
        self._fastk_high = fastk[1]
        self._slowk_low = slowk[0]
        self._slowk_high = slowk[1]
        self._slowd_low = slowd[0]
        self._slowd_high = slowd[1]

    def run(self, df):

        self.df = df

        result = []

        # 全パラメータをグリッドサーチ(計算量: Ο[n^3])
        for i, fastk in enumerate(list(range(self._fastk_low, self._fastk_high, 1))):
            for slowk in list(range(self._slowk_low, self._slowk_high, 1)):
                for slowd in list(range(self._slowd_low, self._slowd_high, 1)):

                    sys.stdout.write(f"\rSlow Stoch [{i}/{len(list(range(self._fastk_low, self._fastk_high, 1)))-1}] 計算中...")
                    profit = self._calculate_slow_stoch_profit(df, fastk, slowk, slowd)

                    result.append([fastk, slowk, slowd, profit])

                    if profit > self._best_parms["profit"]:
                        self._best_parms["fastk"] = fastk
                        self._best_parms["slowk"] = slowk
                        self._best_parms["slowd"] = slowd
                        self._best_parms["profit"] = round(profit, 2)

        print("\n計算終了")

        return pd.DataFrame(result, columns=["fastk", "slowk", "slowd", "profit"])

    def result_graph(self):
        stoch = ta.STOCH(self.df["High"], self.df["Low"], self.df["Close"],
                         fastk_period=self._best_parms["fastk"],
                         slowk_period=self._best_parms["slowk"],
                         slowd_period=self._best_parms["slowd"])

        fig = plt.figure(figsize=(20, 8))

        plt.subplot(211)
        plt.title(f"Slow Stoch trade (profit: {self._best_parms['profit']})")
        plt.plot(self.df["Close"][(self._best_parms["slowk"] + self._best_parms["slowk"]):], marker="o")
        plt.grid()
        plt.ylabel("Stock Price[Yen]")

        plt.subplot(212)
        plt.plot(stoch[0], marker="o")
        plt.plot(stoch[1], marker="o")
        plt.legend([f"Slow-K {self._best_parms['fastk']}", f"Slow-D {self._best_parms['slowk']}/{self._best_parms['slowd']}"])
        plt.ylabel("Slow-K/D[%]")
        plt.grid()

        plt.tight_layout()
        fig.show()

    def result_params(self):
        return self._best_parms
