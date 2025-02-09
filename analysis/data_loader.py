# analysis/data_loader.py

"""Data loading and validation utilities."""

import logging
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def read_excel_data(excel_path: str) -> Dict[str, pd.DataFrame]:
    """Read all required sheets from the Excel file."""
    data = {}
    sheet_names = {
        "overall_performance": "Overall Performance",
        "layer_metrics": "Layer Metrics",
        "energy_analysis": "Energy Analysis",
    }

    for key, sheet_name in sheet_names.items():
        try:
            data[key] = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.debug(f"Successfully read sheet '{sheet_name}'")
        except Exception as e:
            logger.warning(f"Could not read sheet '{sheet_name}': {e}")
            data[key] = None

    return data


def validate_dataframe(
    df: pd.DataFrame, required_cols: List[str], sheet_name: str
) -> None:
    """Validate that DataFrame contains required columns."""
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Excel sheet '{sheet_name}' must contain columns: {required_cols}"
        )
