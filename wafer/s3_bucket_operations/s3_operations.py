import os
import pickle

import boto3
from utils.logger import App_Logger
from utils.main_utils import (
    convert_obj_to_json,
    convert_object_to_pickle,
    raise_exception,
)
from utils.read_params import read_params


class S3_Operations:
    def __init__(self):
        self.s3 = boto3.resource("s3")

        self.log_writter = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.file_format = self.config["model_params"]["save_format"]

    def upload_to_s3(self, src_file, bucket, dest_file, db_name, collection_name):
        method_name = self.upload_to_s3.__name__

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

            os.remove(src_file)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Removed the local copy of {src_file}",
            )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_bucket_from_s3(self, bucket, db_name, collection_name):
        try:
            method_name = self.get_bucket_from_s3.__name__

            bucket = self.s3.Bucket(bucket)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {bucket} s3 bucket",
            )

            return bucket

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def copy_data_to_other_bucket(
        self, src_bucket, src_file, dest_bucket, dest_file, db_name, collection_name
    ):
        try:
            method_name = self.copy_data_to_other_bucket.__name__

            copy_source = {"Bucket": src_bucket, "Key": src_file}

            bucket = self.get_bucket_from_s3(
                bucket=dest_bucket, db_name=db_name, collection_name=collection_name
            )

            bucket.copy(copy_source, dest_file)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Copied data from bucket {src_bucket} to bucket {dest_bucket}",
            )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def delete_file_from_s3(self, bucket, file, db_name, collection_name):
        try:
            method_name = self.delete_file_from_s3.__name__

            self.s3.Object(bucket, file).delete()

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Deleted {file} from bucket {bucket}",
            )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def move_data_to_other_bucket(
        self, src_bucket, src_file, dest_bucket, dest_file, db_name, collection_name
    ):
        try:
            method_name = self.move_data_to_other_bucket.__name__

            self.copy_data_to_other_bucket(
                src_bucket,
                src_file,
                dest_bucket,
                dest_file,
                db_name=db_name,
                collection_name=collection_name,
            )

            self.delete_file_from_s3(
                bucket=src_bucket,
                file=src_file,
                db_name=db_name,
                collection_name=collection_name,
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Moved {src_file} from bucket {src_bucket} to {dest_bucket}",
            )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_file_objs_from_s3(self, bucket, db_name, collection_name):
        try:
            method_name = self.get_file_object_from_s3.__name__

            s3_bucket = self.get_bucket_from_s3(
                bucket, db_name=db_name, collection_name=collection_name
            )

            lst_obj = [obj for obj in s3_bucket.objects.all()]

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got a list of file objects from bucket {bucket}",
            )

            return lst_obj

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_files_from_s3(self, bucket, db_name, collection_name):
        try:
            method_name = self.get_files_from_s3.__name__

            lst = self.get_file_objs_from_s3(
                bucket=bucket, db_name=db_name, collection_name=collection_name
            )

            list_of_files = [obj.key for obj in lst]

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got list of files from bukcet {bucket}",
            )

            return list_of_files

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_file_object_from_s3(self, bucket, filename, db_name, collection_name):
        try:
            method_name = self.get_files_from_s3.__name__

            s3_bucket = self.get_bucket_from_s3(
                bucket=bucket, db_name=db_name, collection_name=collection_name
            )

            for obj in s3_bucket.objects.filter(Prefix=filename):
                self.log_writter.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message=f"Got {filename} from bucket {bucket}",
                )

                return obj

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def load_model_from_s3(self, bucket, model_name, db_name, collection_name):
        try:
            method_name = self.load_model_from_s3.__name__

            model_obj = self.get_file_object_from_s3(
                bucket=bucket,
                filename=model_name,
                db_name=db_name,
                collection_name=collection_name,
            )

            model = convert_object_to_pickle(
                obj=model_obj, db_name=db_name, collection_name=collection_name
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Loaded {model_name} from bucket {bucket}",
            )

            return model

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_schema_from_s3(self, bucket, filename, db_name, collection_name):
        try:
            method_name = self.get_schema_from_s3.__name__

            res = self.get_file_object_from_s3(
                bucket=bucket,
                filename=filename,
                db_name=db_name,
                collection_name=collection_name,
            )

            dic = convert_obj_to_json(
                obj=res, db_name=db_name, collection_name=collection_name
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {filename} schema from bucket {bucket}",
            )

            return dic

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )

    def save_model_to_s3(self, model, filename, db_name, collection_name, model_bucket):
        method_name = self.save_model_to_s3.__name__

        self.log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            model_file = os.path.join(filename + self.file_format)

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File " + filename + " saved. ",
            )

            s3_model_path = os.path.join("trained", model_file)

            self.upload_to_s3(
                src_file=model_file, bucket=model_bucket, dest_file=s3_model_path
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploading the model {model_file} to s3 bucket {model_bucket}",
            )

            return "success"

        except Exception as e:
            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File "
                + filename
                + f" could not be saved. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=db_name,
                collection_name=collection_name,
            )
