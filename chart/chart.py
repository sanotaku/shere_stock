import matplotlib.pyplot as plt


def candlestick(_df):

    df = _df.copy().head(len(_df) - 1)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(111)
    bp = ax1.boxplot(
        df[["Open", "High", "Low", "Close"]],
        patch_artist=True,
        sym="",
        labels=[str(d)[: 10] if i % 10 == 0 else "" for i, d in enumerate(df.index)]
    )
    ax1.set_ylabel("Stock Price[Yen]")

    colors = []
    for i in range(len(df)):
        if df.iloc[i]["Close"] < df.iloc[i]["Open"]:
            colors.append("red")
        else:
            colors.append("green")

    #  boxの色の設定
    for b, c in zip(bp['boxes'], colors):
        b.set(color="black", linewidth=1)  # boxの外枠の色
        b.set_facecolor(c)  # boxの色
    for b in bp['medians']:
        b.set(linewidth=0)

    # シグナルの矢印
    for i in range(len(df)):

        # 買いシグナル矢印
        if df["transaction"][i] == "buy":
            plt.annotate(
                "Buy",
                xy=(i + 1, max(df["High"][i], df["Open"][i], df["Close"][i]) + 50),
                xytext=(0, 50),
                textcoords='offset points',
                arrowprops=dict(color='blue', headwidth=10, width=2, headlength=10)
            )

        # 売りシグナル矢印
        if df["transaction"][i] == "sell":
            plt.annotate(
                "Sell",
                xy=(i + 1, max(df["High"][i], df["Open"][i], df["Close"][i]) + 50),
                xytext=(0, 50),
                textcoords='offset points',
                arrowprops=dict(color='red', headwidth=10, width=2, headlength=10)
            )
    fig.show()
