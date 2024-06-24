prooduct_json_schema = {
    "product_name": {
        "type": "string",
        "description": "Name of the product",
    },
    "product_number": {
        "type": "string",
        "description": "Product number of the product",
    },
    "product_brand_name": {
        "type": "string",
        "description": "Brand name of the product",
    },
    "sku": {
        "type": "string",
        "description": "SKU of the product",
    },
    "price": {
        "type": "string",
        "description": "Price of the product",
    },
    "currency": {
        "type": "string",
        "description": "Currency of the price",
    },
}

required_keys = ["product_brand_name", "product_number"]

images_schema = {
    "images_similarity_score": "a number representing how the items are similar in percents, from 0 to 100",
    "info": "a snippet, providing key-points, why are similar and why not "
}
img_similarity_prompt = (
            f"You are a expert in car parts classification."
            f"our task is to compare images and send an answer-how much are the parts similar."
            f"Don't compare images - like comparing size or quality"
            f"images_similarity_score is a number representing how much these parts are similar in percents(from 0 to 100)"
            f"info is a text snippet in 4 key points-why are the parts similar and why not"
            f"Please respond with your analysis directly in JSON format "
            f"(without using Markdown code blocks or any other formatting). "
            f"The JSON schema should include: {images_schema}.")


import os
import json
import base64
import requests
import pandas as pd
from tqdm import tqdm
from typing import Literal
from datetime import datetime
from openai import OpenAI

now = datetime.now() # get current timestamp
current_timestamp = now.strftime('%y_%m_%d_%H_%M')
           
    
###################### Work with files
# init folders and files
data_to_save_folder = f'files/results/saved_{current_timestamp}/'
if not os.path.exists(data_to_save_folder):
    os.makedirs(data_to_save_folder)
file_to_save_without_enrichment = data_to_save_folder + 'serper_collected.json'
# file_to_save_filtered = data_to_save_folder + 'serper_collected_filtered.json'
file_to_save_enriched = data_to_save_folder + 'enrichment_results.json'
file_to_save_with_page_data = data_to_save_folder + 'all_collected_results.json'


data_to_read_folder = f'files/read/'
table_file = data_to_read_folder + 'df_1106.xlsx' # table     ##### Ready To Run
header_row = 2 # which row represents the column names? Index from 0
all_data_file_from_motorad = data_to_read_folder + 'full_data_parts.json' # all scraped motorad data


# read files
def read_default_table():
    initial_table = pd.read_excel(table_file, header=header_row)
    
with open(all_data_file_from_motorad, 'r') as file:
    motorad_all_items = json.load(file)
    
    
###################### Serper and oai init
## serper options
serper_base_url = "https://google.serper.dev/"
content_types = ['search', 'images','shopping']

# template for serper search. Used in function get_query. 
query_template = """{brand_name} {major} {company_num} site:{website}"""
query_template_full = """("{brand_name} {major}" OR "{brand_name} {minor}" OR "{brand_name} {long_desc}") "{company_num}" "price" ("part" OR "Part #" OR "Part number") ("MPN" OR "Manufacturer Part Number" OR "SKU") (site:autozone.com OR site:{website} OR site:"")"""
shopping_query_template = """{product_description} '{brand}' '{part_number}' sku '{sku}'"""
results_count_by_content_type = {
    'search': 10,
    'shopping': 10,
    'images': 100
}
    
serper_key = os.environ['SERPER_API_KEY']
headers_search_preferences = {
    "location": "United States", # default United States
    "gl": "us", # default us
    "hl": "en", # default en
    # "num": results_count_by_content_type # dynamic update
}
serper_headers = {
    'X-API-KEY': serper_key,
    'Content-Type': 'application/json'
}

## oai options
os.environ['OPENAI_API_KEY'] = os.environ['PROJECT_OAI_KEY']#################################################
client = OpenAI()
    

###################### Define functions
# Function for scraping. Used before enrichment
def scrape_url(url:str, method: Literal['jina', 'serper'] = 'jina')-> str:
    if method == 'jina':
        response = requests.get(f'https://r.jina.ai/{url}')
        return response.text
    elif method == 'serper':
        payload = json.dumps({
            "url": url
        })
        serper_crape_url = "https://scrape.serper.dev"
        try:
            response = requests.request("POST", serper_crape_url, headers=serper_headers, data=payload)
            return json.dumps(response.json())
        except Exception as e:
            print(f"ERROR: {response},\n {e}")
            return None


# function to read unstructured data and return structured
def get_product_unstructured_data_to_json(schema, data, model)->dict: # AK
    system_message = (
        f"You are a data analyst API."
        f"Your task is to extract info about the main product from a raw text. Identify and extract only data releated to the main part."
        f"Please respond with your analysis directly in JSON format "
        f"(without using Markdown code blocks or any other formatting). "
        f"If you can't find required keys in the text, just use empty text or empty list as a default value in the schema. "
        f"The JSON schema should include: {schema}.")
    response = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": str(data)}
            ],
    )
    result = response.choices[0].message.content.strip()
    try:
        result = json.loads(result) 
        return result if isinstance(result, dict) else dict()
    except:
        return dict()


# function to scrape url and validate
def scrape_item(
        url,
        source: dict, 
        schema: dict = prooduct_json_schema, 
        required_keys:list = required_keys,
        gpt_model: str = 'gpt-3.5-turbo'
    ) -> tuple:
    results = {
        'jina_scrape': {},
        'serper_scrape': {},
        'serper_requests': 0,
        'gpt_requests': 0,
        'gpt_errors': 0,
    }
    
    jina_page_text = scrape_url(url, method = 'jina')
    try:
        results['gpt_requests'] += 1
        jina_page_json = get_product_unstructured_data_to_json(
            schema= schema,
            data = jina_page_text,
            model= gpt_model
        )
        results['jina_scrape'] = jina_page_json
    except:
        jina_page_json = {}
        results['gpt_errors'] += 1
    
    for required_key in required_keys:
        if not isinstance(jina_page_json, dict) or not jina_page_json.get(required_key):
            break
    else: # works if not break
        return results
    
    results['serper_requests'] += 1
    serper_page_text = scrape_url(url, method='serper')
    try:
        results['gpt_requests'] += 1
        serper_page_json = get_product_unstructured_data_to_json(
            schema= schema,
            data = serper_page_text,
            model= gpt_model
        )
        results['serper_scrape'] = serper_page_json
    except:
        results['gpt_errors'] += 1
    
    return results


# function to save a dict in json format
def save_result(
        data: dict, 
        path: str
    ) -> None:
    """
    Save collected results to a JSON file.

    Args:
    data (dict): dict with metadata and a list of collected results.
    path (str): Path to save the JSON file.
    """
    with open(path, 'w', encoding = 'utf-8') as file:
        json.dump(data, file, indent=4)
    print(f'Saved {path}')
    

# find item in motorad products by the part number
def find_in_motorad(number) -> dict:
    """
    Find an item in the Motorad products by the part number.

    Args:
        number (str or int): The part number to search for.

    Returns:
        dict: The item from Motorad products that matches the part number. If no item is found, returns None.
    
    Raises:
        ValueError: If the part number is not a string.
    """
        
    for item in motorad_all_items:
        if item['part_number'] == number:
            return item
    
    return None

        
def to_name(text: str) -> str:
    """
    Convert a given string to a readable name format.

    This function takes a string, converts it to lowercase, and replaces spaces with underscores.

    Args:
        text (str): The input string to be converted.

    Returns:
        str: The converted string in lowercase with spaces replaced by underscores.
    
    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("The input must be a string.")
        
    return str(text).lower().replace(' ', '_')



# merge table data and json file data. Returns a list
def merge_table_and_json(table: pd.DataFrame) -> list:
    """
    Merge data from a pandas DataFrame and a JSON file containing Motorad product data.

    This function takes a pandas DataFrame and merges it with data from Motorad products
    by matching part numbers. The merged data is returned as a list of dictionaries.

    Args:
        table (pd.DataFrame): The input DataFrame containing table data.

    Returns:
        list: A list of dictionaries where each dictionary represents merged data from the table and Motorad products.
    
    Raises:
        ValueError: If the input is not a pandas DataFrame.
    """
    if not isinstance(table, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")
    
    data_to_use = []

    for row in table.iloc:
        motorad_num = row[0]
        
        motorad_data = find_in_motorad(motorad_num)
        if motorad_data:
            data_temp = motorad_data.copy()  # don't modify the original data
            for col_name in row.index:  # row.index is a list-like object with column names
                key = to_name(col_name)
                value = row[col_name]
                data_temp.update({  # add table data to the Motorad dict, every column as a key
                    key: str(value) if not pd.isna(value) else None
                })
            data_to_use.append(data_temp)
            
    return data_to_use

def get_query_(item: dict, template: Literal['shopping', 'default'] = 'default') -> str:
    """
    Create a search query from a template using the given item dictionary.
    This function takes a dictionary with various keys and formats a query string 
    based on the specified template.
    Args:
        item (dict): A dictionary containing the keys needed to fill the query template.
        template: What template to use. One of ('shopping', 'default').

    Returns:
        str: A formatted query string.
    """
    if template == 'shoppingXX':
        content:dict = item.get('serper_scrape') or item.get('jina_scrape') or dict()
        query = query_template.format(
            brand_name = item.get('competitor_brand_name') or item.get('competitor_name') or content.get('brand_name') or "",
            major = item.get('major'), #or '',
            # minor = item.get('minor') or '',
            long_desc = item.get('long_desc') or "",
            company_num = content.get('product_number') or item.get('competitor_cross-list_part_number'),# or "",
            website = item.get('competitor_web_domain') #or ""
        )
    else:
        query = query_template.format(
            brand_name = item.get('competitor_brand_name') or item.get('competitor_name'),
            major = item.get('major'),
            minor = item.get('minor'),
            long_desc = item.get('long_desc'),
            company_num = item.get('competitor_cross-list_part_number'), #AK Fixed 15.06.2024
            website = item.get('competitor_web_domain') or ""
        )
    return query



def get_query2(item: dict, template: Literal['shopping', 'default'] = 'default') -> str:
    """
    Create a search query from a template using the given item dictionary.
    This function takes a dictionary with various keys and formats a query string 
    based on the specified template.
    
    Args:
        item (dict): A dictionary containing the keys needed to fill the query template.
        template: What template to use. One of ('shopping', 'default').

    Returns:
        str: A formatted query string.
    """
    # Use common part to fetch nested content
    content = item.get('serper_scrape') or item.get('jina_scrape') or {}

    # Fill out the query template with both item and content info
    query = query_template.format(
        brand_name=item.get('competitor_brand_name') or item['original'].get('competitor_name') or content.get('brand_name') or "",
        major=item.get('major', ''),
        company_num=content.get('product_number') or item.get('competitor_cross-list_part_number', ''),
        website=item.get('competitor_web_domain', '')
    )

    # Debugging: print out the constructed query
    print(f'Constructed Query: {query}')
    
    return query
# Create a query from the template for serper search
def get_query(item: dict, template: Literal['shopping', 'default'] = 'default') -> str:
    """
    Create a search query from a template using the given item dictionary.
    This function takes a dictionary with various keys and formats a query string 
    based on the specified template.
    Args:
        item (dict): A dictionary containing the keys needed to fill the query template.
        template: What template to use. One of ('shopping', 'default').

    Returns:
        str: A formatted query string.
    """
    if template == 'shoppingXXX':
        content:dict = item.get('serper_scrape') # or item.get('jina_scrape') or dict()
        # original:dict = item.get('original') 
        
        # query = shopping_query_template.format(
        #     product_description = content.get('description'),
        #     brand = content.get('brand_name') or item.get('competitor_brand_name') or item.get('competitor_name'),
        #     part_number = content.get('product_number') or '',
        #     sku = content.get('sku') or ''
        # ) ##### commented 20.06
        query = query_template.format(
            brand_name = content.get('brand_name') or item.get('competitor_brand_name') or item.get('competitor_name'),# or "",
            major = item.get('major'), #or '',
            minor = item.get('minor') or '',
            long_desc = item.get('long_desc') or "",
            company_num = content.get('product_number') or item.get('competitor_cross-list_part_number'),# or "",
            website = item.get('competitor_web_domain') #or ""
        )
    else:
        query = query_template.format(
            brand_name = item.get('competitor_brand_name') or item.get('competitor_name'),
            major = item.get('major'),
            minor = item.get('minor'),
            long_desc = item.get('long_desc'),
            company_num = item.get('competitor_cross-list_part_number'), #AK Fixed 15.06.2024
            website = item.get('competitor_web_domain') or ""
        )
    return query


# function to search with serper and return a complete item with the original
def get_item_search_result(
        query: str, 
        original_item: dict,
        content_types: list
    ) -> dict:
    """
    Get search results for a given query string.

    Args:
    query (str): Query string for search.
    original_item (dict): Original item dictionary.

    Returns:
    tuple: dict: Dictionary containing search results, int:requests count.
    """
    counter = 0
    if not isinstance(query, str):
        return None, counter
    
    payload_with_query =  headers_search_preferences.copy() # avoid editing the original payload template
    payload_with_query['q'] = query
    
    result = { # will be updated with query results
        'query': query,
        'original': original_item
    }
    
    for content_type in content_types:
        serper_url = serper_base_url + content_type
        payload = payload_with_query.copy()
        payload['num'] = results_count_by_content_type.get(content_type) or 10
        payload = json.dumps(payload)
        try:
            counter += 1
            response = requests.request("POST", serper_url, headers=serper_headers, data=payload)
            response_obj: dict = response.json()
        except Exception as e:
            print(f"An error occurred while making the request to {serper_url}: {e}")
            continue
        
        response_values = list(response_obj.values())
        result[content_type] = response_values[1] # 0 is {SearchParameters}
        
    return result, counter


# function to search with serper. For all items. The result will be saved in a json file
def get_all_items_search_results(
            items_to_search: list,
            content_types: list = content_types,
            serper_limit: int = 2
        ) -> dict:
    """
    Search with Serper for all items and save the results in a JSON file.

    This function takes a list of items, creates a search query for each item and
    and collects the Serper results. The results are then saved to a JSON file.

    Args:
        items_to_search (list): A list of items to be enriched with search results.
        content_types (list): A list representing what type of content need to get from serper
        serper_limit (int, optional): The maximum number of items to search. Defaults to 0, which means no limit.

    Returns:
        dict: A dictionary containing the metadata and collected search results.
    """
    collected = []    
    total = serper_limit or len(items_to_search)
    
    metadata = {
        'types': content_types,
        'time': current_timestamp,
        'count_by_type': results_count_by_content_type,
        'input_items': total,
        'payload': headers_search_preferences,
        'serper_requests': 0,
        'gpt_requests': 0,
        'gpt_errors': 0
    }
    for item in tqdm(items_to_search[:total]):
        query = get_query(item)
        results, count = get_item_search_result(query, item, content_types)
        metadata['serper_requests'] += count
        if results:
            collected.append(results)
    
    final_results = {
        'metadata': metadata,
        'collected': collected
    }
    save_result(final_results, file_to_save_without_enrichment)
    
    return final_results


# add shopping search results 
def add_shopping_results(item: dict) -> int: # retuens requests count
    shopping_query = get_query(
        item,
        template = 'shopping'
    )
    
    shopping_content, count = get_item_search_result(
        query = shopping_query,
        original_item = None,
        content_types = ['shopping']
    )
    item['shopping'] = shopping_content['shopping']
    return count

    
# function to return collected all results
def all_results_with_page_content(search_results: dict, content_types:list, gpt_model:str = 'gpt-3.5-turbo', serper_scrape_limit_by_result_position:int = 2) -> dict:
    collected_items =  search_results['collected']
    total = len(collected_items)
    metadata = search_results['metadata']
    items_by_url = dict()
    for i, item in enumerate(collected_items):
        print(f'item: {i+1}/{total}')
        for content_type in content_types:
            print('content type:', content_type)
            sources = item[content_type] or []
            sources = sources[:serper_scrape_limit_by_result_position] # AK fix 19.06
            for source in tqdm(sources):
                metadata['gpt_requests'] += 1
                try:
                    gpt_resp =  get_product_unstructured_data_to_json(prooduct_json_schema, str(source), gpt_model)
                except:
                    gpt_resp = dict()
                    metadata['gpt_errors'] += 1
                    
                source['gpt_response'] = gpt_resp
                link = source['link'].split('?')[0].strip('/') # get url main part
                
                source['matches'] = gpt_resp.get('product_number') == item['original'].get('competitor_cross-list_part_number')
                
                serper_req_count = 0
                for key in required_keys:
                    if not gpt_resp.get(key):
                        scraping_result = scrape_item(link, source = source, gpt_model=gpt_model)
                        source.update({
                            'jina_scrape': scraping_result['jina_scrape'],
                            'serper_scrape': scraping_result['serper_scrape'],
                        })
                        
                        source['matches'] = (scraping_result['jina_scrape'].get('product_number') == item['original'].get('competitor_cross-list_part_number')) or\
                                            (scraping_result['serper_scrape'].get('product_number') == item['original'].get('competitor_cross-list_part_number'))
                        
                        metadata['gpt_requests'] += scraping_result['gpt_requests']
                        metadata['gpt_errors'] += scraping_result['gpt_errors']
                        serper_req_count += scraping_result['serper_requests']
                        
                        
                if items_by_url.get(link): # if duplicated, update with new content_type
                    if items_by_url[link].get(content_type):
                        items_by_url[link][content_type].append(source)
                    else:
                        items_by_url[link][content_type] = [source]
                else: # else create new key-link and get page data
                    items_by_url[link] = {
                        'query': item['query'],
                        'original': item['original'],
                        content_type: [source]
                    }
                    # serper_req_count_shopping = add_shopping_results(items_by_url[link]) # results added inside, returns count only
                    # serper_req_count += serper_req_count_shopping
                    metadata['serper_requests'] += serper_req_count
                    
    print()
    final_schema = {
        'metadata': metadata,
        'products': items_by_url
    }
    
    save_result(final_schema, file_to_save_with_page_data)
    return final_schema


def create_image_data(url: str) -> dict:
    """
    Helper function to create image data dictionary.
    
    Parameters:
    url (str): The URL or file path of the image.
    
    Returns:
    dict: Dictionary containing the image type and URL or base64 encoded string.
    """
    if url.startswith('http'):
        return {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
    else:
        with open(url, "rb") as image_file:
            base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
            
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
        }

def images_prompt(prompt: str, vision_model:str, *urls: str) -> str:
    """
    Generate a response from a chat model using a prompt and multiple images.
    
    Parameters:
    prompt (str): The text prompt to be sent to the model.
    *urls (str): Variable length argument list of URLs or file paths for the images.
    
    Returns:
    str: The content of the response generated by the model.
    
    The function checks if the image URLs start with 'http'. If they do, it uses them as is.
    Otherwise, it reads the images from the provided file paths, encodes them to base64, 
    and formats them appropriately. It then sends a chat completion request to the model
    with the prompt and the images.
    """
    images = [create_image_data(url) for url in urls]

    response = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *images
                ],
            }
        ]
    )
    
    return response.choices[0].message.content


# function tf rcomparing images
def image_similarity(result_with_page_contents, content_types_for_simil, vision_model):
    metadata = result_with_page_contents['metadata']
    for item in result_with_page_contents['products'].values(): # keys are links
        original_image = item['original'].get('angle_view')
        if not original_image: continue
        original_image = original_image.replace('images/', 'https://images.motorad.com/MotoRad-stills/')
        for source_type in content_types_for_simil:
            sources = item.get(source_type) or []
            for source in sources:
                source_image = source.get('imageUrl')
                if not (source_image and source.get('matches')): continue
                
                metadata['gpt_requests'] += 1
                try:
                    simil_result = images_prompt(
                        img_similarity_prompt, 
                        vision_model,
                        original_image, source_image
                    )
                    img_similarity_result = json.loads(
                        simil_result
                    )

                    source.update(img_similarity_result)
                except:
                    metadata['gpt_errors'] += 1
    save_result(result_with_page_contents, file_to_save_enriched)
    return result_with_page_contents
                
    
###################### The Main function
def run_app(
        table:pd.DataFrame = None,
        serper_limit:int = 2,
        content_types:list = content_types,
        gpt_model:str = 'gpt-3.5-turbo',
        vision_model:str = 'gpt-4o',
        serper_scrape_limit_by_result_position:int = 2,
    ) -> dict: 
    print("Merging table and json...")
    if not isinstance(table, pd.DataFrame):
        table = read_default_table()
    merged_data = merge_table_and_json(table)
    
    print('Serper requests...')
    search_results = get_all_items_search_results(merged_data, content_types, serper_limit=serper_limit)
    
    print('Receiving page contents...')
    result_with_page_contents = all_results_with_page_content(search_results, content_types, gpt_model, serper_scrape_limit_by_result_position)
    
    print('Starting images similarity...')
    enriched_results = image_similarity(result_with_page_contents, content_types, vision_model)
    
    print("Finished!")
    return enriched_results 
    