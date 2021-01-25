# import libraries
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta


class FastStochOptimizer(object):
    """
    Note:
    ファストストキャスティクス売買を仮想で行い、パラメータをグリッドサーチで最適化
    パラメータ
        :fastK
        :fastD
    """
    def __init__(self):

        self.df = None

        self._fastk_low = None
        self._fastk_high = None
        self._fastd_low = None
        self._fastd_high = None

        self._best_parms = {
            "fastk": 0,
            "fastd": 0,
            "profit": 0
        }

    def _calculate_fast_stoch_profit(self, df, fastk_period: int, fastd_period: int) -> float:
        stoch = ta.STOCHF(df["High"], df["Low"], df["Close"],
                          fastk_period=fastk_period,
                          fastd_period=fastd_period)

        df = pd.DataFrame([df["Close"].to_list(), stoch[0].to_list(), stoch[1].to_list()],
                          columns=stoch[0].index,
                          index=["Close", "fastK", "fastD"]).T

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset

        for i in range(len(df)):
            # buy
            if df["fastD"][i] > df["fastK"][i] and df["fastD"][i - 1] < df["fastK"][i - 1]:
                if asset != 0:
                    stock = asset / df["Close"][i]
                    asset = 0
                # print(f"buy ({df.index[i]}) : {round(df['slowK'][i-1], 1)}/{round(df['slowD'][i-1], 1)} -> {round(df['slowK'][i], 1)}/{round(df['slowD'][i], 1)} | stock: {round(stock, 2)}")
            # sell
            if df["fastD"][i] < df["fastK"][i] and df["fastD"][i - 1] > df["fastK"][i - 1]:
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
                   fastd: List[int]):
        """
        set param
        Args:
            fastk(List) : fast K range[low, high]
            fastd(List) : slow D range[low, high]
        """
        self._fastk_low = fastk[0]
        self._fastk_high = fastk[1]
        self._fastd_low = fastd[0]
        self._fastd_high = fastd[1]

    def run(self, df):

        self.df = df

        result = []

        # 全パラメータをグリッドサーチ(計算量: Ο[n^3])
        for i, fastk in enumerate(list(range(self._fastk_low, self._fastk_high, 1))):
            for fastd in list(range(self._fastd_low, self._fastd_high, 1)):

                sys.stdout.write(f"\rFast Stoch [{i}/{len(list(range(self._fastk_low, self._fastk_high, 1)))-1}] 計算中...")
                profit = self._calculate_fast_stoch_profit(df, fastk, fastd)

                result.append([fastk, fastd, profit])

                if profit > self._best_parms["profit"]:
                    self._best_parms["fastk"] = fastk
                    self._best_parms["fastd"] = fastd
                    self._best_parms["profit"] = round(profit, 2)

        print("\n計算終了")

        return pd.DataFrame(result, columns=["fastk", "fastd", "profit"])

    def result_graph(self):
        stochf = ta.STOCHF(self.df["High"], self.df["Low"], self.df["Close"],
                           fastk_period=self._best_parms["fastk"],
                           fastd_period=self._best_parms["fastd"])

        fig = plt.figure(figsize=(20, 8))

        plt.subplot(211)
        plt.title(f"Fast Stoch trade (profit: {self._best_parms['profit']})")
        plt.plot(self.df["Close"], marker="o")
        plt.grid()
        plt.ylabel("Stock Price[Yen]")

        plt.subplot(212)
        plt.plot(stochf[0], marker="o")
        plt.plot(stochf[1], marker="o")
        plt.legend([f"Fast-K {self._best_parms['fastk']}", f"Fast-D {self._best_parms['fastd']}"])
        plt.ylabel("Fast-K/D[%]")
        plt.grid()

        plt.tight_layout()
        fig.show()

    def result_params(self):
        return self._best_parms
