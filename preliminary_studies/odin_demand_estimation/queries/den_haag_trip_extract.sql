-- Base query for ODiN Den Haag trip extraction.
--
-- Purpose:
--   Select the minimum ODiN fields needed to build PC4 OD demand, map those
--   PC4s to service zones, and estimate category-period arrival/departure
--   rates for the CMDP.
--
-- Template fields:
--   {year}         -> ODiN year, for example 2023 for odin.odin2023.
--   {limit_clause} -> empty for a full run, or LIMIT n.
--
-- Selected columns:
--   opid     respondent id, used for unique-person reliability counts.
--   verplid  trip id, used for unique-trip counts.
--   verpl    trip-record type. The filter below keeps regular trips only.
--   vertpc   origin PC4, used as the intermediate origin geography.
--   aankpc   destination PC4, used as the intermediate destination geography.
--   vertgem  origin municipality. Code 518 is 's-Gravenhage / Den Haag.
--   aankgem  destination municipality. Code 518 is 's-Gravenhage / Den Haag.
--   afstv    trip distance in hectometers. 5..100 means 0.5..10 km.
--   khvm     grouped main mode, kept for later diagnostics or sensitivity checks.
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
