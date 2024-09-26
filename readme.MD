# Swimming Improvement Tracking
This script is a proof of concept script to attempt an alternate fraudulent time filter. Currently, the filter is too aggressive, and will flag times where a swimmer is rapidly improving, as percent improvements are scaled by the number of days since the last swim.

## Installation and Runnning
```bash
python -m pip install -r requirements.txt
python main.py --file "filepath.csv"
```

> [!note]
> Input files have been copied and pasted directly from Swimcloud's event time progression table.

The script will output the full data for all flagged swims to `flagged_swims.csv`.

## Procedure
1. Convert any times over 1 minute to seconds and hundredths of seconds. Format dates into ISO Timecodes.
2. When an event is swam multiple times in a single day, drop all but the fastest time. This is due to scaling swims by the number of days since the last swim, so ignoring this would result in the scalar being 0.
3. Compute the swimmer's best time at the time of each swim. This will be used as the comparison
4. Compute the amount of time between each swim. This time difference is used as a scalar for the percentage improvements.
5. For every swim, compute the number of seconds off of the current best time.
6. Calculate the percent improvement from each time off best value.
7. Scale improvement rate by the time since the last swim. This is to account for swimmers who go multiple months or years without competing.
8. Compute adjusted z-scores for each scaled percent improvement value. Any value above 3.5 is marked as a flagged time.

## Example
Below is an example of the script running on my own 100 Yard Backstroke times. Even though times are not actually fraudulent, the script still flags a few times.
```text
 python main.py --file data/davis_100bk.csv
Flagged Swims
shape: (3, 6)
┌───────┬─────────────────┬────────────────┬───────────┬────────────────┬────────────────┐
│ Time  ┆ Date            ┆ time_from_best ┆ time_diff ┆ scaled_percent ┆ modified_z_sco │
│ ---   ┆ ---             ┆ ---            ┆ ---       ┆ _improvement   ┆ re             │
│ f64   ┆ datetime[μs]    ┆ f64            ┆ i64       ┆ ---            ┆ ---            │
│       ┆                 ┆                ┆           ┆ f64            ┆ f64            │
╞═══════╪═════════════════╪════════════════╪═══════════╪════════════════╪════════════════╡
│ 55.38 ┆ 2017-04-07      ┆ -0.97          ┆ 1         ┆ 1.721384       ┆ 15.168498      │
│       ┆ 00:00:00        ┆                ┆           ┆                ┆                │
│ 51.74 ┆ 2018-03-10      ┆ -1.24          ┆ 1         ┆ 2.340506       ┆ 20.534223      │
│       ┆ 00:00:00        ┆                ┆           ┆                ┆                │
│ 51.37 ┆ 2018-03-30      ┆ -0.37          ┆ 1         ┆ 0.715114       ┆ 6.447482       │
│       ┆ 00:00:00        ┆                ┆           ┆                ┆                │
└───────┴─────────────────┴────────────────┴───────────┴────────────────┴────────────────┘
```