import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from utils.log_tables import Create_Log_Table
from utils.read_params import read_params
from wafer.model.load_production_model import Load_Prod_Model
from wafer.model.prediction_from_model import Prediction
from wafer.model.training_model import Train_Model
from wafer.validation_insertion.prediction_validation_insertion import Pred_Validation
from wafer.validation_insertion.train_validation_insertion import Train_Validation

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index_html_file"], {"request": request}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        raw_data_train_bucket_name = config["s3_bucket"]["wafer_raw_data"]

        table = Create_Log_Table()

        table.generate_log_tables(type="train")

        time.sleep(5)

        train_val = Train_Validation(bucket_name=raw_data_train_bucket_name)

        train_val.training_validation()

        training_model_obj = Train_Model()

        num_clusters = training_model_obj.training_model()

        load_prod_model_obj = Load_Prod_Model(num_clusters=num_clusters)

        load_prod_model_obj.load_production_model()

    except Exception as e:
        return Response("Error Occurred! %s" % e)

    return Response("Training successfull!!")


@app.get("/predict")
async def predictRouteClient():
    try:
        raw_data_pred_bucket_name = config["s3_bucket"]["wafer_raw_data"]

        table = Create_Log_Table()

        table.generate_log_tables(type="pred")

        pred_val = Pred_Validation(raw_data_pred_bucket_name)

        pred_val.prediction_validation()

        pred = Prediction()

        bucket, filename, json_predictions = pred.predict_from_model()

        return Response(
            f"prediction file created in {bucket} bucket with filename as {filename}, and few of the predictions are {str(json.loads(json_predictions))}"
        )

    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)
