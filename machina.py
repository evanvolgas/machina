import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Callable, Dict, List, Optional, TypedDict, Union

import pandas as pd
import pytz
import yaml
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformConfig(TypedDict):
    Input: str
    Output: Union[str, List[str]]
    Transform: str
    Parameters: Optional[Dict]


@dataclass
class Transform:
    """Handles data transformations based on YAML configuration."""

    config: Dict[str, TransformConfig]
    strict_mode: bool = False

    def __init__(self, config_path: str, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.config = self._load_and_validate_config(config_path)

    def _load_and_validate_config(self, config_path: str) -> Dict[str, TransformConfig]:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        for name, conf in config.items():
            if not all(key in conf for key in ("Input", "Output", "Transform")):
                raise ValueError(f"Missing required keys in transform {name}")

            if "Parameters" in conf and not isinstance(conf["Parameters"], dict):
                raise ValueError(f"Parameters in {name} must be a dictionary")

        return config

    @staticmethod
    @lru_cache(maxsize=128)
    def _infer_date_format(sample_date: str) -> Optional[str]:
        """Cache-optimized date format inference."""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",  # Most common formats first
            "%m/%d/%y %H:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d/%y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                datetime.strptime(sample_date.strip(), fmt)
                return fmt
            except ValueError:
                continue
        return None

    def _get_transform_func(self, transform_str: str) -> Callable:
        """Get the appropriate transformation function."""
        builtin_transforms = {
            "datetime": self._to_datetime,
            "date": self._to_date,
            "split_name": self._split_name,
            "copy": lambda x, **_: x.copy(),
            "int": lambda x, **_: pd.to_numeric(x, errors="coerce").astype("Int64"),
            "float": lambda x, **_: pd.to_numeric(x, errors="coerce"),
            "string": lambda x, **_: x.astype(str),
            "proper": lambda x, **_: x.str.strip().str.title(),
        }

        if transform_str in builtin_transforms:
            return builtin_transforms[transform_str]

        if "." in transform_str:
            return lambda x, **_: eval(f"x.{transform_str}", {"x": x})

        raise ValueError(f"Unknown transformation: {transform_str}")

    def _to_datetime(
        self, data: pd.Series, format: Optional[str] = None, timezone: Optional[str] = None
    ) -> pd.Series:
        if not format and not data.empty:
            sample = str(data[data.notna()].iloc[0])
            format = self._infer_date_format(sample)

        dt = pd.to_datetime(data, format=format, errors="coerce")

        if timezone:
            dt = dt.dt.tz_localize(pytz.UTC).dt.tz_convert(pytz.timezone(timezone))
        return dt

    def _to_date(self, data: pd.Series, format: Optional[str] = None) -> pd.Series:
        return self._to_datetime(data, format=format).dt.date

    @staticmethod
    def _split_name(data: pd.Series) -> pd.Series:
        def split(name: str) -> tuple:
            if pd.isna(name):
                return ("", "", "")
            parts = str(name).strip().split()
            return (
                parts[0],
                " ".join(parts[1:-1]) if len(parts) > 2 else "",
                parts[-1] if len(parts) > 1 else "",
            )

        return data.apply(split)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        processed_inputs = set()

        for name, config in self.config.items():
            input_col = config["Input"]

            if input_col not in df.columns:
                msg = f"Input column {input_col} not found"
                if self.strict_mode:
                    raise ValueError(msg)
                logger.warning(f"{msg}. Skipping {name}")
                continue

            try:
                transform_func = self._get_transform_func(config["Transform"])
                transformed = transform_func(df[input_col], **(config.get("Parameters", {})))

                output_cols = config["Output"]
                if isinstance(output_cols, list):
                    if not isinstance(transformed[0], (tuple, list)):
                        raise ValueError(f"{name} didn't return sequence for multiple outputs")
                    for idx, col in enumerate(output_cols):
                        result[col] = [row[idx] for row in transformed]
                else:
                    result[output_cols] = transformed

                processed_inputs.add(input_col)

            except Exception as e:
                if self.strict_mode:
                    raise
                logger.warning(f"Error in {name}: {e}")
                continue

        # Drop processed inputs
        result.drop(columns=list(processed_inputs), inplace=True)
        return result


@dataclass
class BigQueryLoad:
    """Handles loading data into BigQuery."""

    project_id: str
    dataset_id: str
    table_id: str
    if_exists: str = "append"

    def __post_init__(self):
        self.client = bigquery.Client(project=self.project_id)

    def load(self, df: pd.DataFrame):
        table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition=(
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if self.if_exists == "replace"
                else bigquery.WriteDisposition.WRITE_APPEND
            ),
        )

        try:
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()
            logger.info(f"Loaded {len(df)} rows into {table_ref}")
        except Exception as e:
            logger.error(f"BigQuery load error: {e}")
            raise
