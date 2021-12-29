import boto3
import yaml


def read_params():
    try:
        s3 = boto3.resource("s3")

        bucket_name = "input-files-for-train-and-pred"

        bucket = s3.Bucket(bucket_name)

        for obj in bucket.objects.filter(Prefix="params.yaml"):
            content = obj.get()["Body"].read().decode()

            config = yaml.safe_load(content)

            return config

    except Exception as e:
        exception_msg = f"Exception occured in read_params.py,Method : read_params, Error : {str(e)}"

        raise Exception(exception_msg)
