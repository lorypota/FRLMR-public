-- Light ODiN database smoke test.
--
-- Purpose:
--   Check that the database connection works and that the selected yearly ODiN
--   table can be read before running the full demand-estimation query.
--
-- Template fields:
--   {year}  -> ODiN year, for example 2023 for odin.odin2023.
--   {limit} -> maximum number of rows to return.
--
-- Filter:
--   verpl = 1 keeps regular A-to-B trips. This avoids person-only records,
--   follow-up leg rows, series trips, and professional freight records.
--
-- This query intentionally selects all columns. It is for inspection only,
-- not for the final demand-rate aggregation.

SELECT *
FROM odin.odin{year}
WHERE verpl = 1
LIMIT {limit};
