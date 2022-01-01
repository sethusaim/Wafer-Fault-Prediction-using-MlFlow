import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from utils.read_params import read_params
from wafer.model.load_production_model import load_prod_model
from wafer.model.predictionFromModel import prediction
from wafer.model.trainingModel import train_model
from wafer.validation_insertion.prediction_validation_insertion import pred_validation
from wafer.validation_insertion.train_validation_insertion import train_validation

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
        raw_data_train_bucket_name = config["s3_bucket"]["wafer_raw_data_bucket"]

        train_valObj = train_validation(raw_data_train_bucket_name)

        train_valObj.train_validation()

        trainModelObj = train_model()

        num_clusters = trainModelObj.training_model()

        loadProdModelObj = load_prod_model(num_clusters=num_clusters)

        loadProdModelObj.load_production_model()

    except Exception as e:
        return "Error Occurred! %s" % e

    return "Training successfull!!"


@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        raw_data_pred_bucket_name = config["s3_bucket"]["wafer_raw_data_bucket"]

        pred_val = pred_validation(raw_data_pred_bucket_name)

        pred_val.prediction_validation()

        pred = prediction(raw_data_pred_bucket_name)

        path, json_predictions = pred.predictionFromModel()

        return Response(
            "Prediction File created at !!!"
            + str(path)
            + "and few of the predictions are "
            + str(json.loads(json_predictions))
        )

    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)
