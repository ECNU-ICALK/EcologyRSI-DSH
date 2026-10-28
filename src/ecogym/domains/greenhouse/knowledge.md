# Greenhouse climate: background for forecasting

- The indoor climate is controlled. Setpoints (role `action`) are each grower's strategy; the controller drives heating
  pipes, vents, lamps, screens and CO2 dosing to follow them. Schedules repeat almost daily, so yesterday's setpoint at
  the same clock time is usually today's.
- Weather does not repeat day to day and is hard to extrapolate beyond a few hours; solar radiation is the main
  disturbance (heat gain, transpiration, photosynthetic CO2 uptake).
- Air temperature follows heating and ventilation setpoints closely; deviations come from solar gain and vent limits.
- Relative humidity depends on temperature (saturation) and on water vapour from crop transpiration, removed by
  ventilation and dehumidification; predicting it directly works better than recombining separate temperature and
  absolute-humidity forecasts, whose errors add up.
- CO2 follows a mass balance: dosing toward the setpoint, exchange with outside air (~400 ppm) through open vents, and
  uptake by photosynthesis driven by light (sun and lamps). Most CO2 error falls in daylight hours.
- Seasons change the regime (day length, lamp use, vent openings, heating demand), so relationships learned in one
  period transfer imperfectly to the next; level-preserving, anomaly-style models are more robust to such drift.
