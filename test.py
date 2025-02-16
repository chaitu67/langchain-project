import pandas as pd
from pyspark.sql import SparkSession
import os
from argparse import ArgumentParser
from langchain_core.messages import ChatMessage
from pydantic import BaseModel, Field
# Initialize the spark session
def init_spark_session():
    spark = SparkSession.builder.appName("csv_to_parquet").getOrCreate()
    return spark

# Function to convert csv file to pySpark dataframe
def csv_to_df(spark, csv_path):
    try:
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
        df.show()
        return df
    except Exception as e:
        print(f"Error occurred while reading csv: {e}")
        return None


    

# Function to convert pySpark dataframe to parquet file and write it to output-path
def df_to_parquet(spark, input_df, output_path):
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        # Create parquet file in Spark's memory cache
        #spark.cache(input_df)
        
        # Write the parquet file to disk
        input_df.write.parquet(output_path, mode='overwrite')
        
        print(f"Parquet file written successfully at {output_path}")
    except Exception as e:
        print(f"Error occurred while writing parquet: {e}")

def start_program(a):
    
    # Initialize spark session
    spark = init_spark_session()
    
    # Read input csv into df
    input_df = csv_to_df(spark,a.input_path )
    
    if input_df is not None:
        # If output_path is provided, write parquet file to it
        if a.output_path:
            df_to_parquet(spark, input_df, a.output_path)
        else:
            print("No output path provided. Parquet file will be written to the current directory.")
            
    spark.stop()
    return "executed job successfully"

def main():
    parser = ArgumentParser(description="csv_to_parquet program")
    parser.add_argument("--input_csv", type=str, help="Input csv file path")
    parser.add_argument("--output_path", type=str, default=None, help="Output parquet file path (optional)")
    
    args = parser.parse_args()
    
    # Initialize spark session
    spark = init_spark_session()
    
    # Read input csv into df
    input_df = csv_to_df(spark, args.input_csv)
    
    if input_df is not None:
        # If output_path is provided, write parquet file to it
        if args.output_path:
            df_to_parquet(spark, input_df, args.output_path)
        else:
            print("No output path provided. Parquet file will be written to the current directory.")
            
    spark.stop()

if __name__ == "__main__":
    main()
