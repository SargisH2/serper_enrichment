import azure.functions as func
import logging
import pandas as pd
import json
from main import run_app
from async_oai_sample.test_async import run_async_tasks

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="car_parts_enrichment", methods=['POST'])
def car_parts_enrichment(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    req_body = req.get_json()
    
    table_json = req_body.get("table_json")
    table = pd.read_json(table_json)
    logging.info(table.head(2))
    func_result = run_app(
        table = table,
        serper_limit = req_body.get("serper_limit") or 2,
        content_types = req_body.get("content_types") or ['search', 'images','shopping'],
        gpt_model = req_body.get("gpt_model") or 'gpt-3.5-turbo',
        vision_model = req_body.get("vision_model") or 'gpt-4o',
        serper_scrape_limit_by_result_position = req_body.get("scrape_limit") or 2,
    )
    result = json.dumps(func_result)
        
    return func.HttpResponse(result)

@app.route(route="test_async_oai", methods=['POST'])
async def test_async_oai(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    req_body = req.get_json()
    tasks = req_body.get("tasks")
    
    results = await run_async_tasks(prompts = tasks)
    
    return func.HttpResponse(json.dumps(results))