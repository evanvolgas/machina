import pandas as pd
import yaml
from google.cloud import bigquery
from typing import List, Dict, Any, Union, Optional
import logging
from abc import ABC, abstractmethod
import importlib.util
from datetime import datetime
import pytz


# Check for required dependencies
def check_dependency(package: str) -> Optional[ModuleNotFoundError]:
    """Check if a package is installed and return error if not."""
    if importlib.util.find_spec(package) is None:
        return ModuleNotFoundError(
            f"{package} is required but not installed. "
            f"Please install it using: pip install {package}"
        )
    return None


# Check for required packages
for package in ["pyarrow", "pandas", "yaml", "google.cloud.bigquery"]:
    if error := check_dependency(package):
        raise error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transform:
    """Class that handles data transformations based on YAML configuration."""

    def __init__(self, config_path: str, strict_mode: bool = False):
        """
        Initialize Transform with a YAML configuration file.

        Args:
            config_path: Path to YAML configuration file
            strict_mode: If True, raises errors for missing columns. If False, skips missing transformations.
        """
        self.strict_mode = strict_mode
        self.input_columns = set()

        # Enhanced built-in transformations while preserving originals
        self.builtin_transforms = {
            "to_datetime": self._to_datetime,
            "to_date": self._to_date,
            "format_date": self._format_date,
            "split_name": self._split_name,
            # Basic transformations
            "copy": lambda x, **kwargs: x.copy(),
            "int": lambda x, **kwargs: pd.to_numeric(x, errors="coerce").astype("Int64"),
            "float": lambda x, **kwargs: pd.to_numeric(x, errors="coerce"),
            "string": lambda x, **kwargs: x.astype(str),
            # Enhanced transformations
            "clean_string": lambda x, **kwargs: (
                x.str.strip().str.upper() if kwargs.get("upper", False) else x.str.strip()
            ),
        }

        # Load and validate config
        self.config = self._load_config(config_path)
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict:
        """Load and parse YAML configuration file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def _validate_config(self):
        """Validate the configuration structure."""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary of transformations")

        for transform_name, transform_config in self.config.items():
            if not isinstance(transform_config, dict):
                raise ValueError(f"Transform {transform_name} must be a dictionary")

            # Check required keys
            required_keys = {"Input", "Output", "Transform"}
            missing_keys = required_keys - set(transform_config.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing required keys {missing_keys} in transform {transform_name}"
                )

            # Validate Output format
            output = transform_config["Output"]
            if not isinstance(output, (str, list)):
                raise ValueError(
                    f"Output in transform {transform_name} must be either a string or list"
                )

            # If using built-in transform, validate Parameters if present
            transform_str = transform_config["Transform"]
            if transform_str in self.builtin_transforms and "Parameters" in transform_config:
                if not isinstance(transform_config["Parameters"], dict):
                    raise ValueError(
                        f"Parameters in transform {transform_name} must be a dictionary"
                    )

    def _to_datetime(
        self, data: pd.Series, format: Optional[str] = None, timezone: Optional[str] = None
    ) -> pd.Series:
        """Convert string to datetime with format inference and timezone handling."""
        try:
            if format:
                dt = pd.to_datetime(data, format=format, errors="coerce")
            else:
                # Try to infer format from first non-null value
                sample_date = data[data.notna()].iloc[0] if not data.empty else None
                if sample_date:
                    inferred_format = self._infer_date_format(str(sample_date))
                    if inferred_format:
                        dt = pd.to_datetime(data, format=inferred_format, errors="coerce")
                    else:
                        dt = pd.to_datetime(data, errors="coerce")
                else:
                    dt = pd.to_datetime(data, errors="coerce")

            if timezone:
                tz = pytz.timezone(timezone)
                dt = dt.dt.tz_localize(pytz.UTC).dt.tz_convert(tz)

            return dt
        except Exception as e:
            logger.error(f"Error converting to datetime: {e}")
            raise

    def _infer_date_format(self, sample_date: str) -> Optional[str]:
        """Infer date format from a sample date string."""
        common_formats = [
            "%m/%d/%y %H:%M",  # 12/1/10 8:26
            "%m/%d/%Y %H:%M",  # 12/1/2010 8:26
            "%Y-%m-%d %H:%M:%S",  # 2010-12-01 08:26:00
            "%Y-%m-%d",  # 2010-12-01
            "%m/%d/%y",  # 12/1/10
            "%m/%d/%Y",  # 12/1/2010
            "%d-%m-%Y",  # 01-12-2010
            "%d/%m/%Y",  # 01/12/2010
        ]

        for fmt in common_formats:
            try:
                datetime.strptime(sample_date.strip(), fmt)
                return fmt
            except ValueError:
                continue
        return None

    def _to_date(self, data: pd.Series, format: Optional[str] = None) -> pd.Series:
        """Convert string to date only with smart format inference."""
        try:
            if format:
                return pd.to_datetime(data, format=format, errors="coerce").dt.date

            # Try to infer format from first non-null value
            sample_date = data[data.notna()].iloc[0] if not data.empty else None
            if sample_date:
                inferred_format = self._infer_date_format(str(sample_date))
                if inferred_format:
                    return pd.to_datetime(data, format=inferred_format, errors="coerce").dt.date

            # Fallback to letting pandas guess
            return pd.to_datetime(data, errors="coerce").dt.date

        except Exception as e:
            logger.error(f"Error converting to date: {e}")
            raise

    def _format_date(self, data: pd.Series, output_format: str) -> pd.Series:
        """Format datetime/date to string."""
        try:
            if not pd.api.types.is_datetime64_any_dtype(data):
                data = pd.to_datetime(data, errors="coerce")
            return data.dt.strftime(output_format)
        except Exception as e:
            logger.error(f"Error formatting date: {e}")
            raise

    def _split_name(self, data: pd.Series) -> pd.Series:
        """Enhanced name splitting that handles various formats."""

        def split_single_name(name: str) -> tuple:
            if pd.isna(name):
                return ("", "", "")
            parts = str(name).strip().split()
            if len(parts) == 1:
                return (parts[0], "", "")
            elif len(parts) == 2:
                return (parts[0], "", parts[1])
            else:
                return (parts[0], " ".join(parts[1:-1]), parts[-1])

        return data.apply(split_single_name)

    def _apply_transformation(
        self, data: pd.Series, transform_config: Dict
    ) -> Union[pd.Series, List[pd.Series]]:
        """
        Apply transformation specified in the config.

        Args:
            data: Input data series
            transform_config: Dictionary containing transformation details

        Returns:
            Transformed data
        """
        transform_str = transform_config["Transform"]

        try:
            # Check if it's a built-in transformation
            if transform_str in self.builtin_transforms:
                params = transform_config.get("Parameters", {})
                return self.builtin_transforms[transform_str](data, **params)

            # Handle custom transformations
            if "." in transform_str:
                # Create a safe namespace for eval
                namespace = {
                    "pd": pd,
                    "data": data,
                    "str": str,
                    "int": lambda x: pd.to_numeric(x, errors="coerce").astype("Int64"),
                    "float": lambda x: pd.to_numeric(x, errors="coerce"),
                }
                return eval(f"data.{transform_str}", namespace)

            raise ValueError(f"Unknown transformation: {transform_str}")

        except Exception as e:
            logger.error(f"Error applying transformation {transform_str}: {e}")
            if self.strict_mode:
                raise
            return data  # Return original data on error if not in strict mode

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations specified in the config to the input DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with transformed columns
        """
        result_df = df.copy()
        applied_transforms = []

        for transform_name, transform_config in self.config.items():
            input_col = transform_config["Input"]

            # Skip if input column doesn't exist and we're not in strict mode
            if input_col not in df.columns:
                msg = f"Input column {input_col} not found in DataFrame"
                if self.strict_mode:
                    raise ValueError(msg)
                logger.warning(f"{msg}. Skipping transformation {transform_name}")
                continue

            self.input_columns.add(input_col)
            output_cols = transform_config["Output"]

            logger.info(f"Applying transformation {transform_name}")

            try:
                transformed_data = self._apply_transformation(df[input_col], transform_config)

                # Handle multiple output columns
                if isinstance(output_cols, list):
                    if not isinstance(transformed_data[0], (list, tuple)):
                        raise ValueError(
                            f"Transformation {transform_name} did not return sequence for multiple outputs"
                        )
                    if len(output_cols) != len(transformed_data[0]):
                        raise ValueError(
                            f"Number of output columns {len(output_cols)} does not match "
                            f"transformation output size {len(transformed_data[0])}"
                        )
                    for idx, col in enumerate(output_cols):
                        result_df[col] = [row[idx] for row in transformed_data]
                else:
                    result_df[output_cols] = transformed_data

                applied_transforms.append(transform_name)

            except Exception as e:
                if self.strict_mode:
                    raise
                logger.warning(f"Error in transformation {transform_name}: {e}. Skipping.")
                continue

        # Drop input columns only if they were successfully transformed
        columns_to_drop = {
            col
            for col in self.input_columns
            if any(
                transform_name in applied_transforms
                for transform_name, config in self.config.items()
                if config["Input"] == col
            )
        }

        if columns_to_drop:
            logger.info(f"Dropping transformed input columns: {columns_to_drop}")
            result_df.drop(columns=list(columns_to_drop), inplace=True)

        return result_df


class Load(ABC):
    """Abstract base class for data loading."""

    @abstractmethod
    def load(self, df: pd.DataFrame):
        """Load data into target destination."""
        pass


class BigQueryLoad(Load):
    """Class to handle loading data into BigQuery."""

    def __init__(self, project_id: str, dataset_id: str, table_id: str, if_exists: str = "append"):
        """
        Initialize BigQuery loader.

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            if_exists: Action to take if table exists ('append' or 'replace')
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.if_exists = if_exists
        self.client = bigquery.Client(project=project_id)

    def load(self, df: pd.DataFrame):
        """
        Load DataFrame into BigQuery table.

        Args:
            df: Input DataFrame to load
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        logger.info(f"Loading data into BigQuery table {table_ref}")

        try:
            job_config = bigquery.LoadJobConfig()

            if self.if_exists == "replace":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            else:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

            # Automatically detect schema
            job_config.autodetect = True

            # Load data
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)

            # Wait for job to complete
            job.result()

            logger.info(f"Loaded {len(df)} rows into {table_ref}")

        except Exception as e:
            logger.error(f"Error loading data into BigQuery: {e}")
            raise
