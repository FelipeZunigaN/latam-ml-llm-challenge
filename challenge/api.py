import logging
import pandas as pd
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conint
from starlette.status import HTTP_400_BAD_REQUEST

from challenge.model import DelayModel

app = FastAPI()

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: conint(ge=1, le=12)

class FlightsRequest(BaseModel):
    flights: List[Flight]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: FlightsRequest) -> dict:
    logging.info(f"Input Data: {data}")
    
    model = DelayModel()

    df = pd.DataFrame([flight.model_dump() for flight in data.flights])

    features = model.preprocess(df)

    preds = model.predict(features)

    return {'predict': preds}