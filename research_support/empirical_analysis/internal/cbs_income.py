"""Fetch income data per buurt from CBS OpenData.

Uses the cbsodata package to fetch table 85618NED (Kerncijfers wijken en
buurten 2023) which contains income, WOZ, and socioeconomic indicators.

If income data is suppressed or unavailable, returns None and logs a warning.

Usage:
    Import-only helper module used by analysis_*.py scripts.
"""

import logging
from pathlib import Path

import cbsodata
import pandas as pd

logger = logging.getLogger(__name__)

# Tables to try, most recent first
CBS_TABLES = [
    "85618NED",  # Kerncijfers wijken en buurten 2023
    "85163NED",  # 2022
    "84983NED",  # 2021
]

DEN_HAAG_BUURT_PREFIX = "BU0518"

CACHE_FILENAME = "cbs_income_buurten.csv"

# Column prefixes we want, mapped to clean names
INCOME_COLUMN_PREFIXES = {
    "GemiddeldInkomenPerInwoner": "gemiddeld_inkomen_per_inwoner",
    "GemiddeldInkomenPerInkomensontvanger": "gemiddeld_inkomen_per_ontvanger",
    "GemiddeldeWOZWaardeVanWoningen": "gemiddelde_woz_waarde",
    "HuishoudensMetEenLaagInkomen": "pct_huishoudens_laag_inkomen",
    "k_40PersonenMetLaagsteInkomen": "pct_personen_laagste_40",
    "k_20PersonenMetHoogsteInkomen": "pct_personen_hoogste_20",
}


def fetch_income_data(cache_dir=None, force_refresh=False):
    """Fetch income/WOZ data per buurt for Den Haag from CBS.

    Args:
        cache_dir: Directory to cache results. If None, no caching.
        force_refresh: If True, ignore cache and re-fetch.

    Returns:
        DataFrame with buurtcode and income columns, or None if all suppressed.
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / CACHE_FILENAME
        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached CBS income data from %s", cache_path)
            df = pd.read_csv(cache_path)
            if len(df) > 0:
                return df
            logger.warning("Cached file is empty, re-fetching")

    for table_id in CBS_TABLES:
        logger.info("Trying CBS table %s...", table_id)
        df = _try_fetch_table(table_id)
        if df is not None:
            if cache_dir is not None:
                cache_path = Path(cache_dir) / CACHE_FILENAME
                df.to_csv(cache_path, index=False)
                logger.info("Cached CBS income data to %s", cache_path)
            return df

    logger.warning("Could not fetch income data from any CBS table.")
    return None


def _try_fetch_table(table_id):
    """Fetch buurt-level income data from a CBS table via cbsodata."""
    try:
        data = cbsodata.get_data(table_id)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", table_id, e)
        return None

    df = pd.DataFrame(data)
    logger.info("Table %s: %d total rows", table_id, len(df))

    # Find the coding column
    coding_col = None
    for col in df.columns:
        if col.startswith("Codering"):
            coding_col = col
            break
    if coding_col is None:
        logger.warning("No Codering column in %s", table_id)
        return None

    df["buurtcode"] = df[coding_col].str.strip()
    df = df[df["buurtcode"].str.startswith(DEN_HAAG_BUURT_PREFIX)].copy()
    logger.info("  %d Den Haag buurt rows", len(df))

    if len(df) == 0:
        return None

    # Match columns by prefix
    result = {"buurtcode": df["buurtcode"].values}
    found_any = False

    for prefix, clean_name in INCOME_COLUMN_PREFIXES.items():
        matches = [c for c in df.columns if c.startswith(prefix)]
        if matches:
            col = matches[0]
            values = pd.to_numeric(df[col], errors="coerce")
            non_null = values.notna().sum()
            logger.info(
                "  %s -> %s: %d/%d non-null", col, clean_name, non_null, len(df)
            )
            if non_null > 0:
                result[clean_name] = values.values
                found_any = True

    if not found_any:
        logger.info("All income/WOZ values suppressed in %s", table_id)
        return None

    return pd.DataFrame(result)
