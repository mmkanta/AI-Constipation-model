from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter, UploadFile, File, Form
from model.questionnaire import prediction as questionnaire_model
from model.image import prediction as image_model
from model.integrated import prediction as integrated_model
import shutil
# import subprocess
import time
import traceback
# from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import base64

import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Finish clearing temporary file")

class QuestionnaireData(BaseModel):
    questionnaire: List[float]

# questionnaire inference
@router.post("/questionnaire", status_code=200)
async def questionnaire_inference(inferData: QuestionnaireData):
    try:
        if len(inferData.questionnaire) != 15:
            return JSONResponse(content={"success": False, "message": "Length of questionnaire must be 15"}, status_code=400)
        start_time = time.time()
        result = questionnaire_model.make_prediction(inferData.questionnaire)
        print(f"Inference time: {time.time() - start_time :.2f} seconds")
        return JSONResponse(content={"success": True, "message": "Questionnaire infer successfully", "data": {
            "DD_probability": result
        }}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

# image inference
@router.post("/image", status_code=200)
async def image_inference(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    start_time = time.time()
    result_path = os.path.join(TEMP_DIR, f"{start_time}_{file.filename}")
    background_tasks.add_task(remove_file, os.path.join(result_path))
    try:
        Path(os.path.join(result_path, "result")).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(result_path, file.filename), "wb") as f:
            f.write(file.file.read())

        _, _ = image_model.make_prediction(result_path, file.filename)

        print(f"Inference time: {time.time() - start_time :.2f} seconds")

        if os.path.exists(os.path.join(result_path, 'result')):
            shutil.make_archive(os.path.join(result_path, 'result'),
                                'zip',
                                root_dir=os.path.join(result_path, 'result'),
                                )
            return FileResponse(os.path.join(result_path, 'result' + '.zip'), status_code=200)
        return JSONResponse(content={"success": False, "message": "Result not found"}, status_code=500)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

# integrated inference
@router.post("/integrate", status_code=200)
async def integrated_inference(file: UploadFile = File(...), questionnaire: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    start_time = time.time()
    result_path = os.path.join(TEMP_DIR, f"{start_time}_{file.filename}")
    background_tasks.add_task(remove_file, os.path.join(result_path))
    try:
        try:
            question_list = json.loads(questionnaire)
            if not isinstance(question_list, list):
                raise Exception("Questionnaire must be array of integer")
            if len(question_list) != 15:
                raise Exception("Length of questionnaire must be 15")
            if not all(isinstance(x, (float, int)) for x in question_list):
                raise Exception("Questionnaire must be array of integer")
        except Exception as e:
            return JSONResponse(content={"success": False, "message": str(e)}, status_code=400)
            
        Path(os.path.join(result_path, "result")).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(result_path, file.filename), "wb") as f:
            f.write(file.file.read())

        _, _ = integrated_model.make_prediction(result_path, file.filename, question_list)

        print(f"Inference time: {time.time() - start_time :.2f} seconds")

        if os.path.exists(os.path.join(result_path, 'result')):
            shutil.make_archive(os.path.join(result_path, 'result'),
                                'zip',
                                root_dir=os.path.join(result_path, 'result'),
                                )
            return FileResponse(os.path.join(result_path, 'result' + '.zip'), status_code=200)
        return JSONResponse(content={"success": False, "message": "Result not found"}, status_code=500)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)