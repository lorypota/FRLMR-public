-- Base query for ODiN potential bike-suitable Den Haag trips.
--
-- Purpose:
--   Select the minimum ODiN fields needed to estimate category-period
--   arrival and departure rates for a Donkey-only Den Haag CMDP.
--
-- Template fields:
--   {year}         -> ODiN year, for example 2023 for odin.odin2023.
--   {limit_clause} -> empty for a full run, or LIMIT n.
--
-- Selected columns:
--   opid     respondent id, used for unique-person reliability counts.
--   verplid  trip id, used for unique-trip counts.
--   verpl    trip-record type. The filter below keeps regular trips only.
--   vertpc   origin PC4, used to map departures to service categories.
--   aankpc   destination PC4, used to map arrivals to service categories.
--   vertgem  origin municipality. Code 518 is 's-Gravenhage / Den Haag.
--   aankgem  destination municipality. Code 518 is 's-Gravenhage / Den Haag.
--   afstv    trip distance in hectometers. 5..100 means 0.5..10 km.
--   khvm     grouped main mode, used for all-modes, car-driver, and bike scenarios.
--   vertuur  departure hour, mapped to morning/evening periods.
--   factorv  ODiN trip expansion weight, used for weighted demand totals.
--
-- Filters:
--   verpl = 1
--     Keeps regular A-to-B trips. This excludes person-only rows, follow-up
--     leg rows, series trips, and professional freight records.
--
--   afstv BETWEEN 5 AND 100
--     Keeps trips from 0.5 km to 10 km. ODiN stores this distance in
--     hectometers, so 5 = 0.5 km and 100 = 10 km.
--
--   vertgem = 518 OR aankgem = 518
--     Keeps trips with a Den Haag origin or destination.
--
-- The Python script applies the scenario filters after this base query:
--   bike_suitable_all_modes     no extra khvm filter
--   bike_suitable_car_driver    khvm = 1
--   bike_suitable_current_bike  khvm = 5
--
-- Note:
--   The Python script dynamically handles 2024 DANS-style geography names
--   such as vertpc_pram/aankpc_pram and vertgem_dans24/aankgem_dans24.

SELECT
    opid,
    verplid,
    verpl,
    vertpc,
    aankpc,
    vertgem,
    aankgem,
    afstv,
    khvm,
    vertuur,
    factorv
FROM odin.odin{year}
WHERE verpl = 1
  AND afstv BETWEEN 5 AND 100
  AND (vertgem = 518 OR aankgem = 518)
{limit_clause};
