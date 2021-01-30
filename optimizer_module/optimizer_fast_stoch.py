# import libraries
import concurrent.futures
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta

from chart import chart


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

        _df = df.copy()

        stoch = ta.STOCHF(_df["High"], _df["Low"], _df["Close"],
                          fastk_period=fastk_period,
                          fastd_period=fastd_period)

        _df["fastK"] = stoch[0]
        _df["fastD"] = stoch[1]
        _df["transaction"] = None

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset
        last_close_price = 0

        for i in range(len(_df) - 1):
            # buy
            if _df["fastD"][i] > _df["fastK"][i] and _df["fastD"][i - 1] < _df["fastK"][i - 1]:
                if asset != 0:
                    last_close_price = asset
                    stock = asset / _df["Open"][i + 1]
                    asset = 0
                    _df["transaction"][i + 1] = "buy"
                    continue
            # sell
            if _df["fastD"][i] < _df["fastK"][i] and _df["fastD"][i - 1] > _df["fastK"][i - 1]:
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

        result = []

        _df = df.copy()

        def inner_thread(fastk):

            sys.stdout.write(f"Thread START FastStoch Span={fastk}\n")

            for fastd in list(range(self._fastd_low, self._fastd_high + 1, 1)):

                profit, r_df = self._calculate_fast_stoch_profit(_df, fastk, fastd)

                result.append([fastk, fastd, profit])

                if profit > self._best_parms["profit"]:
                    self.df = r_df
                    self._best_parms["fastk"] = fastk
                    self._best_parms["fastd"] = fastd
                    self._best_parms["profit"] = round(profit, 2)
            return

        # concurrent.futuresでマルチスレッド(並列計算)処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            res = executor.map(
                inner_thread,
                [j for j in range(self._fastk_low, self._fastk_high + 1, 1)]
            )

        print("\n計算終了")

        return pd.DataFrame(result, columns=["fastk", "fastd", "profit"]).sort_values("profit", ascending=False)

    def result_graph(self):
        stochf = ta.STOCHF(self.df["High"], self.df["Low"], self.df["Close"],
                           fastk_period=self._best_parms["fastk"],
                           fastd_period=self._best_parms["fastd"])

        chart.candlestick(self.df)

        fig = plt.figure(figsize=(12, 4))
        plt.subplot(111)
        plt.plot(stochf[0], marker="o")
        plt.plot(stochf[1], marker="o")
        plt.legend([f"Fast-K {self._best_parms['fastk']}", f"Fast-D {self._best_parms['fastd']}"])
        plt.ylabel("Fast-K/D[%]")
        plt.ylim([0, 100])
        plt.grid()

        plt.tight_layout()
        fig.show()

    def result_params(self):
        return self._best_parms
