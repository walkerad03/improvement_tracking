import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_PERCENT_IMPROVEMENTS = False

# Importing data
df = pl.scan_csv("data/davis_50fr.csv")

# Formatting dates
df = df.select(pl.col("Time", "Date")).with_columns(
    pl.col("Date")
    .str.strptime(pl.Date, "%b %d, %Y")
    .dt.strftime("%Y-%m-%d")
    .str.to_datetime()
)

# Calculating current best time for each swim
df = df.sort(["Date", "Time"])

# Only keep the fastest swim per day
df = df.unique(subset=["Date"], keep="first")

df = df.with_columns(
    pl.col("Time")
    .cum_min()
    .shift(1, fill_value=float("inf"))
    .alias("current_best")
)

sns.lineplot(data=df.collect(), x="Date", y="Time")
plt.show()

print(df.collect())

# Computing time between swims
df = df.with_columns(pl.col("Date").diff().alias("time_diff").dt.total_days())

# Calculating time from best
df = df.with_columns(
    pl.col("Time").sub(pl.col("current_best")).alias("time_from_best")
)

# Calculating percent improvement
df = df.with_columns(
    (
        (pl.col("Time") - pl.col("current_best"))
        / pl.col("current_best")
        * (-100)
    ).alias("percent_improvement")
)

# Scale improvement rate by time since swim
df = df.with_columns(
    pl.when(pl.col("time_diff") > 0)
    .then(pl.col("percent_improvement") / pl.col("time_diff"))
    .otherwise(0)
    .alias("scaled_percent_improvement")
)


if PLOT_PERCENT_IMPROVEMENTS:
    sns.histplot(data=df.collect(), x="scaled_percent_improvement", bins=75)
    plt.show()

median_pct_imp = (
    df.select(pl.col("scaled_percent_improvement")).median().collect().item()
)

df = df.with_columns(
    (pl.col("scaled_percent_improvement") - median_pct_imp)
    .abs()
    .alias("abs_dev")
)


mad = df.select(pl.col("abs_dev")).median().collect().item()

df = df.with_columns(
    (
        0.6745 * (pl.col("scaled_percent_improvement") - median_pct_imp) / mad
    ).alias("modified_z_score")
)

print(
    df.select(
        [
            "Time",
            "Date",
            "time_from_best",
            "time_diff",
            "scaled_percent_improvement",
            "modified_z_score",
        ]
    )
    .filter(pl.col("modified_z_score") > 3.5)
    .collect()
)
