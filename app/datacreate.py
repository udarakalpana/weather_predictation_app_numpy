import logging
from os.path import join

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(data: dict):
    daily = data["daily"]

    dates = np.array(daily["time"], dtype="datetime64[D]")

    values = np.column_stack([
        np.array(daily['temperature_2m_max'], dtype="float"),
        np.array(daily['temperature_2m_min'], dtype="float"),
        np.array(daily['precipitation_sum'], dtype="float"),
        np.array(daily['wind_speed_10m_max'], dtype="float"),
    ])

    return dates[:-1], values[:-1]


def build_summary(dates, values) -> str:
    temp_max = values[:,0]
    rain = values[:,2]

    avg_max = np.nanmean(temp_max)
    total_rain = np.nansum(rain)
    trend = temp_max[-1] - temp_max[0]

    lines = []

    for related_date, rows in zip(dates, values):

        lines.append(
            f"{related_date}: max {rows[0]:.1f}°C, min {rows[1]:.1f}°C, "
            f"rain {rows[2]:.1f} mm, wind {rows[3]:.1f} km/h"
        )

    text = "\n".join(lines)


    return (
        f"{text}\n\n"
        f"Summary: average max temp {avg_max:.1f}°C, total rain {total_rain:.1f} mm, "
        f"temperature trend {trend:.1f} °C over the week."
    )



