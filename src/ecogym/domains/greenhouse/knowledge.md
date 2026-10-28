# Greenhouse climate: what is known (AGC 2018 cucumber, team AiCU)

## System
- The indoor climate is controlled. Setpoints (role `action`) are each team's strategy; the controller drives heating
  pipes, vents, lamps and CO2 dosing to follow them. Controller schedules repeat almost daily: the CO2 setpoint equals
  yesterday's value at the same clock time 87% of the time, the humidity-deficit setpoint 94%, lamps 85%.
- Weather does not repeat: outside radiation anomalies correlate only 0.13 day to day; morning and afternoon
  radiation of the same day correlate 0.27-0.37. Weather is hard to extrapolate beyond a few hours.
- Indoor humidity quantities are physically consistent (RH reconstructed from air temperature and humidity deficit
  matches within 0.07 %RH), but only 18% of a 1 h RH change is explained by temperature change at constant absolute
  humidity; transpiration, ventilation and dehumidification drive the rest.
- CO2 follows a mass balance: dosing toward the setpoint, ventilation exchange with outside air (~400 ppm), and
  photosynthetic uptake driven by light (sun and lamps). 82-90% of CO2 squared error falls in daylight hours.

## Regime drift between partitions
From calibration_fit (16 Aug - 9 Sep) to model_selection (20 Sep - 23 Oct): lamps on 15% -> 37% of hours, leeward
vent mean opening 51.6% -> 18.4%, heating pipe 3.95 -> 7.90 degC, outside temperature 17.4 -> 14.4 degC, indoor CO2
mean 571 -> 636 ppm, RH mean 78.4% -> 84.8%. Relationships learned in August (hour-of-day effects, weather and
actuator interactions, implicit long-run levels) transfer poorly to October.

## Findings of the October 2026 study (fit on calibration_fit, choices on model_selection)
- The target's own recent days at the same clock time are the main signal: a two-day profile beats one day; three or
  more days add nothing reliable within a 48 h history.
- A free ridge on profile reads has coefficients summing below one plus an intercept, i.e. it pulls toward the
  August level; a constrained sum-to-one, no-intercept profile-anomaly member keeps the current level. Their equal
  blend beat both members in every period checked; blend weights 0.25-0.75 are all close.
- Mechanism terms that transferred: the last 1 h change for air temperature and RH; yesterday's heating and
  ventilation setpoint steps over the same clock window for air temperature; for CO2 the vent exchange proxy
  (opening x (CO2 - 400)), outside radiation, lamp power, dosing gap (setpoint - CO2) and yesterday's CO2 setpoint
  step.
- Did not help: hour-of-day harmonics or hour-varying coefficients, weather anomalies, rolling 7-14 day refits,
  clear-sky radiation geometry, other targets' anomalies, and predicting RH through absolute humidity and
  temperature (the two errors add up).
- Per cell on model_selection the expert's skill intervals exclude zero for T@1, RH@1 and all CO2 cells; T@6, T@24,
  RH@6 and RH@24 are within noise of the naive baseline.
