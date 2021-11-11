import os
import yaml

project_dir = os.getcwd()


def create_project_dir():
    try:
        if not os.path.exists(project_dir):
            os.makedirs(exist_ok=True)

        base_data = {
            "base": {
                "project_name": "null",
                "project_dir": project_dir,
                "random_state": "null",
                "target_col": "null",
                "test_size": "null",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(base_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_config_folder():
    try:
        config_path = "config"

        os.makedirs(config_path, exist_ok=True)

        schema_files = ["schema_training.json", "schema_prediction.json"]

        for files in schema_files:
            open(file=config_path + "/" + files, mode="a+").close()

        schema_data = {
            "schema_dir": {
                "train_schema_file": os.path.join(config_path, schema_files[0]),
                "pred_schema_file": os.path.join(config_path, schema_files[1]),
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(schema_data, f, line_break=2)

        f.close()

    except Exception as e:
        print(str(e))


def create_main_db_logs():
    try:
        main_db_logs = {
            "db_log": {
                "db_train_log": "training_logs",
                "db_pred_log": "prediction_logs",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(main_db_logs, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_train_db_logs():
    try:
        train_db_log_data = {
            "train_db_log": {
                "model_training": "modelTrainingLog",
                "col_validation": "columnValidationLog",
                "db_conn": "databaseConnectionLog",
                "data_transform": "dataTransform",
                "db_insert": "dbInsertLog",
                "db_create": "dbTableCreateLog",
                "export_csv": "ExportToCsvLog",
                "general": "GeneralLog",
                "load_prod_model": "loadProdModelLog",
                "missing_values_in_col": "missingValuesInColumn",
                "name_validation": "nameValidationLog",
                "train_main": "Training_Main_Log",
                "values_from_schema": "valuesfromSchemaValidationLog",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(train_db_log_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_pred_db_logs():
    try:
        pred_db_log_data = {
            "pred_db_log": {
                "col_validation": "columnValidationLog",
                "db_conn": "databaseConnectionLog",
                "data_transform": "dataTransform",
                "db_insert": "dbInsertLog",
                "db_create": "dbTableCreateLog",
                "export_csv": "ExportToCsvLog",
                "general": "GeneralLog",
                "pred_main": "prediction_main_log",
                "missing_values_in_col": "missingValuesInColumn",
                "name_validation": "nameValidationLog",
                "values_from_schema": "valuesfromSchemaValidationLog",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(pred_db_log_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_data_folder():
    try:
        data_path = os.path.join("data", "preprocessing_data")

        os.makedirs(data_path, exist_ok=True)

        elbow_plot_fig = {
            "elbow_plot_fig": os.path.join(data_path, "K-Means_Elbow.PNG")
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(elbow_plot_fig, f)

        f.close()

        temp = ["train", "pred"]

        for t in temp:
            good_data_path = os.path.join("data", t, "raw_valid", "good")

            bad_data_path = os.path.join("data", t, "raw_valid", "bad")

            archived_data_path = os.path.join("data", t, "archived")

            os.makedirs(good_data_path, exist_ok=True)

            os.makedirs(bad_data_path, exist_ok=True)

            os.makedirs(archived_data_path, exist_ok=True)

            data_yaml = {
                "data": {
                    "good": {
                        "train": os.path.join("data", temp[0], "raw_valid", "good"),
                        "pred": os.path.join("data", temp[1], "raw_valid", "good"),
                    },
                    "bad": {
                        "train": os.path.join("data", temp[0], "raw_valid", "bad"),
                        "pred": os.path.join("data", temp[1], "raw_valid", "bad"),
                    },
                    "archived": {
                        "train": os.path.join("data", temp[0], "archived"),
                        "pred": os.path.join("data", temp[1], "archived"),
                    },
                }
            }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(data_yaml, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_regex():
    try:
        regex_data = {"regex_pattern": "null"}

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(regex_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_raw_data_folder():
    try:
        data_given_path = "data_given"

        os.makedirs(data_given_path, exist_ok=True)

        batch_files = ["training_batch_files", "prediction_batch_files"]

        for bf in batch_files:
            data_source = os.path.join(data_given_path, bf)

            os.makedirs(data_source, exist_ok=True)

        data_source_yaml = {
            "data_source": {
                "train_data_source": os.path.join(data_given_path, batch_files[0]),
                "pred_data_source": os.path.join(data_given_path, batch_files[1]),
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(data_source_yaml, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_databases_folder():
    try:
        db_dir = "databases"

        files_from_db = ["Training_FileFromDB", "Prediction_FileFromDB"]

        dbs = ["Training_Database", "Prediction_Database"]

        for files in files_from_db:
            db_file_path = os.path.join(db_dir, files)

            os.makedirs(db_file_path, exist_ok=True)

        for db in dbs:
            db_path = os.path.join(db_dir, db)

            os.makedirs(db_path, exist_ok=True)

        databases_folder_data = {
            "db_file": {
                "train_db_file": os.path.join(
                    db_dir, files_from_db[0], "InputFile.csv"
                ),
                "pred_db_file": os.path.join(db_dir, files_from_db[1], "InputFile.csv"),
            },
            "db_file_path": {
                "train_db_path": os.path.join(db_dir, files_from_db[0]),
                "pred_db_path": os.path.join(db_dir, files_from_db[1]),
            },
            "database_dir": {
                "train_database": os.path.join(db_dir, dbs[0]),
                "pred_database": os.path.join(db_dir, dbs[1]),
            },
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(databases_folder_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_models_folder():
    try:
        models_dir = "models"

        models_list = ["trained", "prod", "stag"]

        for model in models_list:
            temp_folder = os.path.join(models_dir, model)

            os.makedirs(temp_folder, exist_ok=True)

        model_folder_data = {
            "model_dir": {
                "trained_models_dir": os.path.join(models_dir, models_list[0]),
                "prod_models_dir": os.path.join(models_dir, models_list[1]),
                "stag_model_folder": os.path.join(models_dir, models_list[2]),
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(model_folder_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_mlflow_config():
    try:
        mlflow_config_data = {
            "mlflow_config": {
                "artifacts_dir": "null",
                "experiment_name": "null",
                "remote_server_uri": "null",
                "run_name": "null",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(mlflow_config_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_output_folder():
    try:
        output_folder_path = os.path.join("output", "Prediction_Output_File")

        os.makedirs(output_folder_path, exist_ok=True)

        pred_file = os.path.join(output_folder_path, "Predictions.csv")

        open(pred_file, mode="a+").close()

        output_folder_data = {
            "pred_output_file": os.path.join(output_folder_path, "Predictions.csv")
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(output_folder_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_src_folder():
    try:
        data_ingestion_path = os.path.join(
            "src", "data_preprocessing", "data_ingestion"
        )

        os.makedirs(data_ingestion_path, exist_ok=True)

        di_files = ["data_loader_prediction.py", "data_loader_train.py"]

        for files in di_files:
            open(file=data_ingestion_path + "/" + files, mode="a+").close()

        data_preprocessing_path = os.path.join("src", "data_preprocessing")

        os.makedirs(data_preprocessing_path, exist_ok=True)

        dp_files = ["clustering.py", "preprocessing.py"]

        for files in dp_files:
            tmp_file = os.path.join(data_preprocessing_path, files)

            open(file=tmp_file, mode="a+").close()

        data_transform_path = os.path.join("src", "dataTransform")

        os.makedirs(data_transform_path, exist_ok=True)

        dt_files = ["data_transformation_pred.py", "data_transformation_train.py"]

        for files in dt_files:
            tmp_file = os.path.join(data_transform_path, files)

            open(file=tmp_file, mode="a+").close()

        dataTypeValid_path = os.path.join("src", "dataTypeValid")

        os.makedirs(dataTypeValid_path, exist_ok=True)

        dtv_files = ["data_type_valid_pred.py", "data_type_valid_train.py"]

        for files in dtv_files:
            tmp_file = os.path.join(dataTypeValid_path, files)

            open(file=tmp_file, mode="a+").close()

        file_operations_folder_path = os.path.join("src", "file_operations")

        os.makedirs(file_operations_folder_path, exist_ok=True)

        file_method_file = os.path.join(file_operations_folder_path, "file_methods.py")

        open(file_method_file, mode="a+").close()

        model_file_path = os.path.join("src", "model")

        os.makedirs(model_file_path, exist_ok=True)

        model_files = [
            "load_production_model.py",
            "predictionFromModel.py",
            "trainingModel.py",
        ]

        for model in model_files:
            tmp_file = os.path.join(model_file_path, model)

            open(file=tmp_file, mode="a+").close()

        model_finder_folder_path = os.path.join("src", "model_finder")

        os.makedirs(model_finder_folder_path, exist_ok=True)

        tuner_file = os.path.join(model_finder_folder_path, "tuner.py")

        open(tuner_file, mode="a+").close()

        raw_data_validation_path = os.path.join("src", "raw_data_validation")

        os.makedirs(raw_data_validation_path, exist_ok=True)

        rdv_files = ["pred_data_validation.py", "train_data_validation.py"]

        for files in rdv_files:
            tmp_file = os.path.join(raw_data_validation_path, files)

            open(file=tmp_file, mode="a+").close()

        validation_insertion_path = os.path.join("src", "validation_insertion")

        os.makedirs(validation_insertion_path, exist_ok=True)

        vi_files = [
            "prediction_validation_insertion.py",
            "train_validation_insertion.py",
        ]

        for files in vi_files:
            tmp_file = os.path.join(validation_insertion_path, files)

            open(file=tmp_file, mode="a+").close()

    except Exception as e:
        print(str(e))


def create_kmeans_cluster():
    try:
        kmeans_cluster_data = {
            "kmeans_cluster": {
                "init": "null",
                "max_clusters": "null",
                "knee_locator": {"curve": "null", "direction": "null"},
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(kmeans_cluster_data, f)

            f.close()

    except Exception as e:
        print(str(e))


def create_utils_folder():
    try:
        app_logging_path = os.path.join("utils", "application_logging")

        os.makedirs(app_logging_path, exist_ok=True)

        logger_file = os.path.join(app_logging_path, "logger.py")

        open(logger_file, mode="a+").close()

        utils_path = "utils"

        os.makedirs(utils_path, exist_ok=True)

        param_file = os.path.join(utils_path, "read_params.py")

        open(param_file, mode="a+").close()

    except Exception as e:
        print(str(e))


def create_knn_data():
    try:
        knn_data = {
            "knn_imputer": {
                "n_neighbors": "null",
                "weights": "null",
            }
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(knn_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_workflows_folder():
    try:
        workflows_path = os.path.join(".github", "workflows")

        os.makedirs(workflows_path, exist_ok=True)

        ci_file = os.path.join(workflows_path, "ci.yml")

        open(file=ci_file, mode="a+").close()

    except Exception as e:
        print(str(e))


def create_templates_folder():
    try:
        templates_path = "templates"

        os.makedirs(templates_path, exist_ok=True)

        index_html_file = os.path.join(templates_path, "index.html")

        open(file=index_html_file, mode="a+").close()

        templates_data = {
            "templates": {"dir": templates_path, "index_html_file": index_html_file}
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(templates_data, f)

        f.close()

    except Exception as e:
        print(str(e))


def create_other_files():
    try:
        other_files = [
            "main.py",
            "manifest.yml",
            "Procfile",
            "README.md",
            "DockerFile",
            "requirements.txt",
            "runtime.txt",
        ]

        other_folders = ["docs/EDA", "docs/LLD", "docs/HLD", "docs/Other"]

        for folder in other_folders:
            os.makedirs(folder, exist_ok=True)

        for files in other_files:
            tmp_file = os.path.join(project_dir, files)

            open(file=tmp_file, mode="a+").close()

        other_data = {
            "null_values_csv_file": os.path.join(
                "data", "preprocessing_data", "null_values.csv"
            ),
            "pred_output_file": os.path.join(
                "output", "Prediction_Output_File", "Predictions.csv"
            ),
            "export_csv_file_name": "InputFile.csv",
        }

        with open(file="params.yaml", mode="a+") as f:
            yaml.dump(other_data, f)

        f.close()

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    create_project_dir()

    create_config_folder()

    create_data_folder()

    create_regex()

    create_raw_data_folder()

    create_databases_folder()

    create_kmeans_cluster()

    create_models_folder()

    create_workflows_folder()

    create_mlflow_config()

    create_output_folder()

    create_train_db_logs

    create_pred_db_logs

    create_main_db_logs()

    create_src_folder()

    create_utils_folder()

    create_templates_folder()

    create_other_files()
