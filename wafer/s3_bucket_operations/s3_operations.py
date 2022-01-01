import os
import pickle

import boto3
import botocore
from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.main_utils import convert_obj_to_json, convert_object_to_pickle
from utils.read_params import read_params


class S3_Operations:
    def __init__(self):
        self.s3_client = boto3.client("s3")

        self.s3_resource = boto3.resource("s3")

        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.file_format = self.config["model_params"]["save_format"]

    def create_folder_in_s3(self, bucket_name, folder_name, db_name, collection_name):
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Folder already exists. Passing to next method",
            )

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.put_folder_in_s3(
                    bucket=bucket_name,
                    folder_name=folder_name,
                    db_name=db_name,
                    collection_name=collection_name,
                )

            else:
                self.log_writer.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message="Error occured in creating folder",
                )

                raise e

    def put_folder_in_s3(self, bucket, folder_name, db_name, collection_name):
        method_name = self.put_folder_in_s3.__name__

        try:
            self.s3_client.put_object(Bucket=bucket, Key=(folder_name + "/"))

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Created {folder_name} folder in {bucket} bucket",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def upload_to_s3(self, src_file, bucket, dest_file, db_name, collection_name):
        method_name = self.upload_to_s3.__name__

        try:
            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploading {src_file} to s3 bucket {bucket}",
            )

            self.s3_resource.meta.client.upload_file(src_file, bucket, dest_file)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploaded {src_file} to s3 bucket {bucket}",
            )

            os.remove(src_file)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Removed the local copy of {src_file}",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_bucket_from_s3(self, bucket, db_name, collection_name):
        try:
            method_name = self.get_bucket_from_s3.__name__

            bucket = self.s3_resource.Bucket(bucket)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {bucket} s3 bucket",
            )

            return bucket

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def copy_data_to_other_bucket(
        self, src_bucket, src_file, dest_bucket, dest_file, db_name, collection_name
    ):
        try:
            method_name = self.copy_data_to_other_bucket.__name__

            copy_source = {"Bucket": src_bucket, "Key": src_file}

            self.s3_resource.meta.client.copy(copy_source, dest_bucket, dest_file)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Copied data from bucket {src_bucket} to bucket {dest_bucket}",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def delete_file_from_s3(self, bucket, file, db_name, collection_name):
        try:
            method_name = self.delete_file_from_s3.__name__

            self.s3_resource.Object(bucket, file).delete()

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Deleted {file} from bucket {bucket}",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
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

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Moved {src_file} from bucket {src_bucket} to {dest_bucket}",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_files_from_s3(self, bucket, folder_name, db_name, collection_name):
        try:
            method_name = self.get_files_from_s3.__name__

            lst = self.get_file_objects_from_s3(
                bucket=bucket,
                db_name=db_name,
                collection_name=collection_name,
                filename=folder_name,
            )

            list_of_files = [obj.key for obj in lst]

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got list of files from bukcet {bucket}",
            )

            return list_of_files

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_file_objects_from_s3(self, bucket, filename, db_name, collection_name):
        try:
            method_name = self.get_files_from_s3.__name__

            s3_bucket = self.get_bucket_from_s3(
                bucket=bucket, db_name=db_name, collection_name=collection_name
            )

            lst_objs = [obj for obj in s3_bucket.objects.filter(Prefix=filename)]

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {filename} from bucket {bucket}",
            )

            if len(lst_objs) == 1:
                return lst_objs[0]

            else:
                return lst_objs

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def load_model_from_s3(self, bucket, model_name, db_name, collection_name):
        try:
            method_name = self.load_model_from_s3.__name__

            model_obj = self.get_file_objects_from_s3(
                bucket=bucket,
                filename=model_name,
                db_name=db_name,
                collection_name=collection_name,
            )

            model = convert_object_to_pickle(
                obj=model_obj, db_name=db_name, collection_name=collection_name
            )

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Loaded {model_name} from bucket {bucket}",
            )

            return model

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def get_schema_from_s3(self, bucket, filename, db_name, collection_name):
        try:
            method_name = self.get_schema_from_s3.__name__

            res = self.get_file_objects_from_s3(
                bucket=bucket,
                filename=filename,
                db_name=db_name,
                collection_name=collection_name,
            )

            dic = convert_obj_to_json(
                obj=res, db_name=db_name, collection_name=collection_name
            )

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Got {filename} schema from bucket {bucket}",
            )

            return dic

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def save_model_to_s3(self, model, filename, db_name, collection_name, model_bucket):
        method_name = self.save_model_to_s3.__name__

        self.log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            model_file = os.path.join(filename + self.file_format)

            self.trained_model_dir = self.config["models_dir"]["trained"]

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File " + filename + " saved. ",
            )

            s3_model_path = os.path.join(self.trained_model_dir, model_file)

            self.upload_to_s3(
                src_file=model_file,
                bucket=model_bucket,
                dest_file=s3_model_path,
                db_name=db_name,
                collection_name=collection_name,
            )

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Uploading the model {model_file} to s3 bucket {model_bucket}",
            )

            return "success"

        except Exception as e:
            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Model File "
                + filename
                + f" could not be saved. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )
