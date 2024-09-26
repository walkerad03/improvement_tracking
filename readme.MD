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