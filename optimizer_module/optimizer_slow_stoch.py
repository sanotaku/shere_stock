# import libraries
import concurrent.futures
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta

from chart import chart


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

        _df = df.copy()

        stoch = ta.STOCH(_df["High"], _df["Low"], _df["Close"],
                         fastk_period=fastk_period,
                         slowk_period=slowd_period,
                         slowd_period=slowk_period)

        _df["slowK"] = stoch[0]
        _df["slowD"] = stoch[1]
        _df["transaction"] = None

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset
        last_close_price = 0

        for i in range(len(_df) - 1):
            # buy
            if _df["slowD"][i] > _df["slowK"][i] and _df["slowD"][i - 1] < _df["slowK"][i - 1]:
                if asset != 0:
                    last_close_price = asset
                    stock = asset / _df["Open"][i + 1]
                    asset = 0
                    _df["transaction"][i + 1] = "buy"
                    continue
            # sell
            if _df["slowD"][i] < _df["slowK"][i] and _df["slowD"][i - 1] > _df["slowK"][i - 1]:
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

        _df = df.copy()
        result = []

        def inner_thread(fastk):

            sys.stdout.write(f"Thread START SlowStoch Fast-K={fastk}\n")

            for slowk in list(range(self._slowk_low, self._slowk_high + 1, 1)):
                for slowd in list(range(self._slowd_low, self._slowd_high + 1, 1)):

                    # sys.stdout.write(f"\rSlow Stoch [{i+1}/{len(list(range(self._fastk_low, self._fastk_high, 1)))}] 計算中...")
                    profit, r_df = self._calculate_slow_stoch_profit(df=_df, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)

                    result.append([fastk, slowk, slowd, profit])

                    if profit > self._best_parms["profit"]:
                        self.df = r_df
                        self._best_parms["fastk"] = fastk
                        self._best_parms["slowk"] = slowk
                        self._best_parms["slowd"] = slowd
                        self._best_parms["profit"] = round(profit, 2)
            return

        # concurrent.futuresでマルチスレッド(並列計算)処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            res = executor.map(
                inner_thread,
                [j for j in range(self._fastk_low, self._fastk_high + 1, 1)]
            )

        print("\n計算終了")

        return pd.DataFrame(result, columns=["fastk", "slowk", "slowd", "profit"]).sort_values("profit", ascending=False)

    def result_graph(self):
        stoch = ta.STOCH(self.df["High"], self.df["Low"], self.df["Close"],
                         fastk_period=self._best_parms["fastk"],
                         slowk_period=self._best_parms["slowk"],
                         slowd_period=self._best_parms["slowd"])

        chart.candlestick(self.df)

        fig = plt.figure(figsize=(12, 4))
        plt.subplot(111)
        plt.plot(stoch[0], marker="o")
        plt.plot(stoch[1], marker="o")
        plt.legend([f"Slow-K {self._best_parms['fastk']}", f"Slow-D {self._best_parms['slowk']}/{self._best_parms['slowd']}"])
        plt.ylabel("Slow-K/D[%]")
        plt.ylim([0, 100])
        plt.grid()

        plt.tight_layout()
        fig.show()

    def result_params(self):
        return self._best_parms
