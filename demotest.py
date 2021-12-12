from io import StringIO
import pandas as pd
from wafer.s3_bucket_operations.s3_operations import get_file_from_s3

file_data = get_file_from_s3(bucket="test-wafer")

data = StringIO(file_data)

df = pd.read_csv(data)

print(df)
