import boto3


def get_file_from_s3(bucket):
    s3 = boto3.resource("s3")

    bucket = s3.Bucket("test-wafer")

    for obj in bucket.objects.all():
        key = obj.key

        schema = obj.get()["Body"].read().decode()

        return schema, key
