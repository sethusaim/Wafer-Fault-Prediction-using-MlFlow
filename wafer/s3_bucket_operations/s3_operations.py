import os
import pickle

import boto3
from utils.logger import App_Logger
from utils.main_utils import convert_obj_to_json, convert_object_to_pickle, read_params


class S3_Operations:
    def __init__(self):
        self.s3 = boto3.resource("s3")

        self.log_writter = App_Logger()

        self.config = read_params()

        self.file_format = self.config["model_params"]["save_format"]

    def upload_to_s3(self, src_file, bucket, dest_file, db_name, collection_name):
        try:
            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploading {src_file} to s3 bucket {bucket}",
            )

            self.s3.meta.client.upload_file(src_file, bucket, dest_file)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploaded {src_file} to s3 bucket {bucket}",
            )

        except Exception as e:
            expection_msg = f"Exception occured in Class : S3_Operations, Method : upload_to_s3, Error : {str(e)}"

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=expection_msg,
            )

            raise Exception(expection_msg)

    def get_bucket_from_s3(self, bucket, db_name, collection_name):
        try:
            bucket = self.s3.Bucket(bucket)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {bucket} s3 bucket",
            )

            return bucket

        except Exception as e:
            self.log_writter.log(
                db_name=db_name, collection_name=collection_name, log_message=""
            )

            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_bucket_from_s3, Error : {str(e)}"
            )

    def copy_data_to_other_bucket(self, src_bucket, src_file, dest_bucket, dest_file):
        try:
            copy_source = {"Bucket": src_bucket, "Key": src_file}

            bucket = self.get_bucket_from_s3(bucket=dest_bucket)

            bucket.copy(copy_source, dest_file)

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : copy_data_to_other_bucket, Error : {str(e)}"
            )

    def delete_file_from_s3(self, bucket, file):
        try:
            self.s3.Object(bucket, file).delete()

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : delete_file_from_s3, Error : {str(e)}"
            )

    def move_data_to_other_bucket(self, src_bucket, src_file, dest_bucket, dest_file):
        try:
            self.copy_data_to_other_bucket(src_bucket, src_file, dest_bucket, dest_file)

            self.delete_file_from_s3(bucket=src_bucket, file=src_file)

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : move_data_to_other_bucket, Error : {str(e)}"
            )

    def get_file_objs_from_s3(self, bucket):
        try:
            s3_bucket = self.get_bucket_from_s3(bucket)

            lst_obj = [obj for obj in s3_bucket.objects.all()]

            return lst_obj

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_file_objs_from_s3, Error : {str(e)}"
            )

    def get_files_from_s3(self, bucket):
        try:
            lst = self.get_file_objs_from_s3(bucket=bucket)

            list_of_files = [obj.key for obj in lst]

            return list_of_files

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_files_from_s3, Error : {str(e)}"
            )

    def get_file_object_from_s3(self, bucket, filename):
        try:
            s3_bucket = self.get_bucket_from_s3(bucket)

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
                f"Exception occured in Class : S3_Operations, Method : load_model_from_s3, Error : {str(e)}"
            )

    def get_schema_from_s3(self, bucket, filename):
        try:
            res = self.get_file_object_from_s3(bucket=bucket, filename=filename)

            dic = convert_obj_to_json(res)

            return dic

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : S3_Operations, Method : get_schema_from_s3, Error : {str(e)}"
            )

    def save_model_to_s3(self, model, filename, db_name, collection_name):
        self.log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Entered the save_model method of the S3_operation class",
        )

        try:
            model_file = os.path.join(self.models_dir, filename + self.file_format)

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File " + filename + " saved. ",
            )

            s3_model_path = os.path.join("trained", model_file)

            self.s3_obj.upload_to_s3(
                src_file=model_file, bucket=self.model_bucket, dest_file=s3_model_path
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploading the model {model_file} to s3 bucket {self.model_bucket}",
            )

            os.remove(model_file)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Local copy of the model {model_file} is removed",
            )

            return "success"

        except Exception as e:
            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Exception occured in Class : Model_Finder, Method : save_model, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File "
                + filename
                + " could not be saved. Exited the save_model_to_s3 method of the Model_Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder, Method : save_model, Error : ",
                str(e),
            )
