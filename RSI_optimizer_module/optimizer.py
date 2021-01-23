# import libraries
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import talib as ta


class RsiOptimizer(object):
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
                   buy_thres: List[int],
                   sell_thres: List[int],
                   spans: List[int]):
        """
        set param
        Args:
            buy_thres(List)  : RSI buy limit[low, high]
            sell_thres(List) : RSI sell limit[low, high]
            span_thres(List) : RSI span range[low, high]
        """
        self._buy_thres_low = buy_thres[0]
        self._buy_thres_high = buy_thres[1]
        self._sell_thres_low = sell_thres[0]
        self._sell_thres_high = sell_thres[1]
        self._span_low = spans[0]
        self._span_high = spans[1]

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
        # set RSI cal data
        df["RSI"] = ta.RSI(df["Close"], span)

        # Args
        stock = 0
        start_asset = 10000
        asset = start_asset

        for i in range(len(df)):

            # buy
            if df["RSI"][i] > low_thres and df["RSI"][i - 1] < low_thres:
                if asset != 0:
                    stock = asset / df["Close"][i]
                    asset = 0
                # print(f"buy : {round(df['RSI'][i], 1)} -> {round(df['RSI'][i-1], 1): 5} | stock: {round(stock, 2)}")

            # sell
            if df["RSI"][i] < high_thres and df["RSI"][i - 1] > high_thres:
                if stock != 0:
                    asset = stock * df["Close"][i]
                    stock = 0
                # print(f"sell: {round(df['RSI'][i], 1)} -> {round(df['RSI'][i-1], 1): 5} | asset: {round(asset, 2)}")

        # If you still have stock, sell it
        last_asset = asset + stock * df["Close"][-1]
        pl_amount_retio = last_asset / start_asset * 100
        # print(f"PL Amount: {round(pl_amount_retio, 2)}")
        return pl_amount_retio

    def run(self, df):
        """
        Note: 解析開始コマンド
        concurrent.futuresで並列計算を行う
        Args:
            df(pandas dataframe): stock data
        Return:
            result(dict)        : 解析結果
        """

        self.df = df

        def _thread_loop(span):
            """
            Thread用inner function
            """
            max_profit = {
                "span": span,
                "buy_thres": 0,
                "sell_thres": 0,
                "profit": 0
            }

            # loop low and high thres
            for buy_thres in list(range(self._buy_thres_low, self._buy_thres_high)):
                for sell_thres in list(range(self._sell_thres_low, self._sell_thres_high)):
                    result = self._calculate_rsi_profit(df, span, buy_thres, sell_thres)
                    if result > max_profit["profit"]:
                        max_profit["buy_thres"] = buy_thres
                        max_profit["sell_thres"] = sell_thres
                        max_profit["profit"] = result

            return max_profit

        result = []

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="thread") as executor:
            futures = []
            spans = list(range(self._span_low, self._span_high, 1))
            for i, span in enumerate(spans):
                sys.stdout.write(f"\r[{i}/{len(spans)-1}] 計算中...")
                futures.append(executor.submit(_thread_loop, span))
                result = [thread.result() for thread in futures]

            print("\n計算終了")

        # best parmsの登録
        for r in result:
            if self._best_parms["profit"] < r["profit"]:
                self._best_parms["span"] = r["span"]
                self._best_parms["buy_thres"] = r["buy_thres"]
                self._best_parms["sell_thres"] = r["sell_thres"]
                self._best_parms["profit"] = round(r["profit"], 3)
                self.df["RSI"] = ta.RSI(df["Close"], self._best_parms["span"])
                self.df["buy_thres"] = r["buy_thres"]
                self.df["sell_thres"] = r["sell_thres"]

        return result

    def result_params(self):
        """
        Note: 解析結果パラメタを表示
        """
        return self._best_parms

    def result_graph(self):
        """
        Note: matplotlibで解析結果グラフを表示
        """
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(self.df["Close"])
        plt.subplot(212)
        plt.plot(self.df["RSI"])
        plt.plot(self.df["buy_thres"])
        plt.plot(self.df["sell_thres"])
        plt.tight_layout()
        plt.show()
