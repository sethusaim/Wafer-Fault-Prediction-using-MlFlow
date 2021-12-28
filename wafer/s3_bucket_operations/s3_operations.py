import boto3
from utils.main_utils import convert_obj_to_json, convert_object_to_pickle


class S3_Operations:
    def __init__(self):
        self.s3 = boto3.resource("s3")

    def upload_to_s3(self, src_file, bucket, dest_file):
        try:
            self.s3.meta.client.upload_file(src_file, bucket, dest_file)

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : upload_to_s3, Error : {str(e)}"
            )

    def copy_data_to_other_bucket(self, src_bucket, src_file, dest_bucket, dest_file):
        try:
            copy_source = {"Bucket": src_bucket, "Key": src_file}

            bucket = self.s3.Bucket(dest_bucket)

            bucket.copy(copy_source, dest_file)

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : copy_data_to_other_bucket, Error : {str(e)}"
            )

    def move_data_to_other_bucket(self, src_bucket, src_file, dest_bucket, dest_file):
        try:
            self.copy_data_to_other_bucket(src_bucket, src_file, dest_bucket, dest_file)

            self.s3.Object(src_bucket, src_file).delete()

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : move_data_to_other_bucket, Error : {str(e)}"
            )

    def get_csv_objs_from_s3(self, bucket):
        lst_obj = []

        try:
            s3_bucket = self.s3.Bucket(bucket)

            for obj in s3_bucket.objects.all():
                lst_obj.append(obj)

            return lst_obj

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_csv_objs_from_s3, Error : {str(e)}"
            )

    def list_files_in_s3(self, bucket):
        try:
            list_of_files = []

            s3_bucket = self.s3.Bucket(bucket)

            for my_bucket_object in s3_bucket.objects.all():
                f = my_bucket_object.key

                list_of_files.append(f)

            return list_of_files

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : read_csv_from_s3, Error : {str(e)}"
            )

    def get_file_object_from_s3(self, bucket, filename):
        try:
            s3_bucket = self.s3.Bucket(bucket)

            for obj in s3_bucket.objects.filter(Prefix=filename):
                return obj

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_file_from_s3, Error : {str(e)}"
            )

    def load_model_from_s3(self, bucket, model_name):
        try:
            model_obj = self.get_file_object_from_s3(bucket=bucket, filename=model_name)

            model = convert_object_to_pickle(model_obj)

            return model

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_model_from_s3, Error : {str(e)}"
            )
    def get_schema_from_s3(self, bucket, filename):
        try: 
            res = self.get_file_object_from_s3(bucket=bucket, filename=filename)

            dic = convert_obj_to_json(res)

            return dic

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_model_from_s3, Error : {str(e)}"
            )
