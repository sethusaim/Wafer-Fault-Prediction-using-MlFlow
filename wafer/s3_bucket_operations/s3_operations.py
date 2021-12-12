import boto3


def get_file_from_s3(bucket):
    s3 = boto3.resource("s3")

    s3_bucket = s3.Bucket(bucket)

    for obj in s3_bucket.objects.all():
        key = obj.key

        file_content = obj.get()["Body"].read().decode()

    return file_content
