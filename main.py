from machina import Transform, BigQueryLoad
import pandas as pd

# Example usage:
if __name__ == "__main__":
    # Create transformer with non-strict mode (skip missing columns)
    transformer = Transform("transformation.yaml", strict_mode=False)

    # Sample data
    df = pd.read_csv("~/Desktop/test.csv")

    # Transform will skip any configured transformations where input columns don't exist
    transformed_df = transformer.transform(df)

    # Load to BigQuery
    loader = BigQueryLoad(
        project_id="your-project", dataset_id="your_dataset", table_id="your_table"
    )
    loader.load(transformed_df)
    loader.load(transformed_df)
