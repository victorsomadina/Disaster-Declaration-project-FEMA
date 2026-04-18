import pandas as pd
import numpy as np
import requests
from src.utils.helpers import get_logger
from pathlib import Path

BASE_URL = "https://www.fema.gov/api/open"

log = get_logger(__name__)

ENDPOINTS = {
    "declarations": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

FIELDS = {
    "declarations": [
        "disasterNumber", "state", "declarationType", "incidentType",
        "declarationDate", "incidentBeginDate", "incidentEndDate",
        "fyDeclared", "designatedArea", "declarationRequestNumber",
    ],
    "public_assistance": [
        "disasterNumber", "stateAbbreviation", "incidentType",
        "projectAmount", "federalShareObligated", "totalObligated",
        "projectSize", "damageCategoryCode", "applicationTitle",
    ],
    "disaster_summaries": [
        "disasterNumber", "totalAmountIhpApproved", "totalAmountHaApproved",
        "totalAmountOnaApproved", "totalObligatedAmountPa",
        "totalObligatedAmountCatAb", "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp", "paLoadDate", "iaLoadDate",
    ],
}

def _fetch_paginated_data(endpoint: str, fields: list[str], max_records: int|None = None):
    params = {
        "$format": "json",
        "$top": 1000,
        "$select": ",".join(fields),
        "$inlinecount": "allpages"
    }
    skip = 0

    records = []

    while True:
        resp = requests.get(endpoint, params={**params, "$skip": skip}, timeout=60)
        resp.raise_for_status()

        payload = resp.json()
        # log.info(f"Fetched {len(payload['value'])} records from {endpoint}")

        batch = payload[[k for k in payload if k not in ("metadata", "count")][0]]

        if not batch:
            break

        records.extend(batch)
        skip = skip + 1000

        return pd.DataFrame(records)
    

def fetch_decl():
    return _fetch_paginated_data(ENDPOINTS["declarations"], FIELDS["declarations"])

def fetch_PA():
    return _fetch_paginated_data(ENDPOINTS["public_assistance"], FIELDS["public_assistance"])

def fetch_DS():
    return _fetch_paginated_data(ENDPOINTS["disaster_summaries"], FIELDS["disaster_summaries"])

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def save_data(df: pd.DataFrame, filename: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{filename}.csv", index=False)
    log.info(f"Saved {len(df)} records to src/data/{filename}.csv")


def run_ingestion():
    log.info("Starting data ingestion")
    decl = fetch_decl()
    PA = fetch_PA()
    DS = fetch_DS()

    save_data(decl, "declarations")
    save_data(PA, "PA")
    save_data(DS, "DS")

    log.info("Data ingestion complete")

if __name__ == "__main__":
    run_ingestion()

 