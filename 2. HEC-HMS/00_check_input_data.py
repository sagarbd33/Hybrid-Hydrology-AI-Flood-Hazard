# -*- coding: utf-8 -*-
"""
Step 0 - Quality check of the station rainfall used to force HEC-HMS.

The IWRD station records contain long periods recorded as 0 mm (station not
reporting). HEC-HMS treats these as 'no rain', so they are reported here.
A station-year is flagged when its total is below 25 % of the median annual
total of all station-years in the basin (likely missing data stored as zero),
or when it contains missing (NaN) values.

Input  : Data/Observed/ObservedPr.csv  (Date + 12 stations)
Output : Results/Station_rainfall_annual_totals.csv
         Results/Station_rainfall_flagged_years.csv

Run    : python 00_check_input_data.py
"""

import pandas as pd

from common import DATA_DIR, RESULTS_DIR

FLAG_FRACTION = 0.25


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    pr = pd.read_csv(DATA_DIR / "Observed" / "ObservedPr.csv", parse_dates=["Date"]).set_index("Date")
    annual = pr.resample("YE").sum(min_count=1)
    annual.index = annual.index.year
    annual.index.name = "Year"
    annual.round(1).to_csv(RESULTS_DIR / "Station_rainfall_annual_totals.csv")

    ref = float(annual.stack().median())     # basin-wide median station-year total
    flags = []
    for st in annual.columns:
        for yr, tot in annual[st].items():
            n_nan = int(pr.loc[str(yr), st].isna().sum())
            if pd.isna(tot) or tot < FLAG_FRACTION * ref or n_nan > 0:
                flags.append({"Station": st, "Year": yr, "Annual_total_mm": tot,
                              "Basin_median_mm": round(ref, 1), "Missing_values": n_nan})
    flags = pd.DataFrame(flags)
    flags.to_csv(RESULTS_DIR / "Station_rainfall_flagged_years.csv", index=False)

    print(f"Basin-wide median station-year total: {ref:.0f} mm; flag threshold {FLAG_FRACTION * ref:.0f} mm")
    print("Station-years with (near-)zero totals or missing values:")
    if flags.empty:
        print("  none")
    else:
        summary = flags.groupby("Station")["Year"].agg(lambda y: f"{len(y)} yrs: " + ", ".join(map(str, y)))
        for st, txt in summary.items():
            print(f"  {st:<16} {txt}")
    print(f"\nTables -> {RESULTS_DIR.name}/Station_rainfall_*.csv")


if __name__ == "__main__":
    main()
