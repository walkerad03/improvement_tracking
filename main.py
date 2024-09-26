import argparse

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

# Plot histogram of performance improvement percentages
PLOT_PERCENT_IMPROVEMENTS = True

# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Relative filepath for times")

filepath = parser.parse_args().file

assert filepath is not None, "Must supply file (--file <filepath>)"

# Importing data
df = pl.scan_csv(filepath)

# Format times from MM:SS.MS to SS.MS
times_vec = df.select(pl.col("Time")).collect().to_series().to_list()

for idx, time in enumerate(times_vec):
    time_str = str(time)
    if ":" in time_str:
        mins = int(time_str.split(":")[0])
        secs = int(time_str.split(":")[1].split(".")[0])
        ms = float(time_str.split(".")[1])

        time_sec_ms = round(mins * 60 + secs + ms / 100, 2)
    else:
        secs = int(time_str.split(".")[0])
        ms = int(time_str.split(".")[1])
        time_sec_ms = round(secs + ms / 100, 2)

    times_vec[idx] = time_sec_ms

df = df.with_columns(pl.Series("Time", times_vec))

# Formatting dates
df = df.select(pl.col("Time", "Date")).with_columns(
    pl.col("Date")
    .str.strptime(pl.Date, "%b %d, %Y")
    .dt.strftime("%Y-%m-%d")
    .str.to_datetime()
)

# Plot swim times as a function of date swam
sns.lineplot(data=df.collect(), x="Date", y="Time")
plt.xlabel("Date")
plt.ylabel("Time (s)")
plt.title("Swimming results over time")
plt.show()

# Calculating current best time for each swim
df = df.sort(["Date", "Time"])

# Only keep the fastest swim per day
df = df.unique(subset=["Date"], keep="first")

# Find personal best time at time of swim
df = df.with_columns(
    pl.col("Time")
    .cum_min()
    .shift(1, fill_value=float("inf"))
    .alias("current_best")
)


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
    sns.histplot(data=df.collect(), x="scaled_percent_improvement")
    plt.xlabel("Scaled Percent Improvements")
    plt.ylabel("Count")
    plt.title("Distribution of Scaled Percent Improvements")
    plt.show()

# Compute Adjusted Z-Scores
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

# Display flagged times and output full details to disk
print("Flagged Swims")
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

df.collect().write_csv("all_swims.csv")
