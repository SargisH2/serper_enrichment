import logging
import pandas as pd
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from main_app import run_app
from async_oai_sample.test_async import run_async_tasks


app = FastAPI()

class CarPartsEnrichmentRequest(BaseModel):
    table_json: str
    serper_limit: int = 2
    content_types: list = ['search', 'images', 'shopping']
    gpt_model: str = 'gpt-3.5-turbo'
    vision_model: str = 'gpt-4o'
    scrape_limit: int = 2

class TestAsyncOaiRequest(BaseModel):
    tasks: list

@app.post("/car_parts_enrichment")
async def car_parts_enrichment(req: CarPartsEnrichmentRequest):
    logging.info('Processing car parts enrichment request.')
    
    table = pd.read_json(req.table_json)
    logging.info(table.head(2).to_json())
    
    func_result = run_app(
        table=table,
        serper_limit=req.serper_limit,
        content_types=req.content_types,
        gpt_model=req.gpt_model,
        vision_model=req.vision_model,
        serper_scrape_limit_by_result_position=req.scrape_limit,
    )
    
    result = json.dumps(func_result)
    return result


@app.post("/test_async_oai")
async def test_async_oai(req: TestAsyncOaiRequest):
    logging.info('Processing test async OAI request.')
    
    results = await run_async_tasks(prompts=req.tasks)
    
    return json.dumps(results)
