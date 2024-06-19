###### Armen ToDo List 15.06.2024:######
#
# 1. Replace Prompt with Function calling (to extratt info)------
# 2. "serper_scraping" to execure befor the Enrichments step (run func by unic url - duplicated url between search image and shop) -
# 2. "page_content": Write fun to extart json part detials from any web page to Json format using fucntion calling
# 3. Use Json "page_content" to mach the search with the results using pytohn (not LLM) + Change similarity logic (Extrat data like temp and other to json format).
# 3.2 Add "page_content" from Shop and images as well - Need to run befor the filtering and enrichment. (include in"serper_collected.json")
# 4. Fix "enrichment_limit" Run total 2 Not For each Source
# ### not for now 4. Image Similarity using The Search and products url with motorad Part
# 5. Add Function to get product info from product_id
# 6. Add Combined Search Results with search,shopping and images. Linked via source url (Search for Product descrition, attributes, etc., Products for Price, )
# 7.1. Image Search Return 100 search results (not 10)
# 8. Done. Fix the "None" (("Duralast Thermostat" OR "Duralast Thermostat w/ Housing" OR "Duralast Automatic Transmission Oil Cooler Thermostat-167 Degrees") "None" "price" ("part" OR "Part #" OR "Part number") ("MPN" OR "Manufacturer Part Number" OR "SKU") (site:autozone.com OR site:AutoZone.com OR site:""))
# 9. include all main specs from Motorad catalog + Serper & serper_scraping results (use func call to expart all deatisl to standarrtize json format)
###########################################
### functions to add
# 1)compare json objects 
# 2)function to get all product images from a url (main product images only)
# 3)function to collect data from google shopping with google product id(collect data. product desc, prices comparition, images)
# 4)clean urls from image/search/shopping(after ?)


###
# functions TODO 18.06: function to get all parameters from raw text dynamically
# function vision comparision to compare first 5 products with 1-2 image comparisions


# search steps
""""
run search 10 results, images 20 results
extract link from source, make them unique
# jina_scrape
# serper_scrape
# serper_search
# serper_images_results
# serper_shopping_results
## save them all. if jina completed requirements, don't use serper

Add Function to get product info from product_id
""{brand_name} {major} {company_num} site:{website} "" # add sku, description(from scraping result), part_number(from search results) with OR
search with this query in shopping
"""






prooduct_json_schema = {
    "name": {
        "type": "string",
        "description": "Name of the product",
    },
    "product_number": {
        "type": "string",
        "description": "Product number of the product",
    },
    "sku": {
        "type": "string",
        "description": "SKU of the product",
    },
    "description": {
        "type": "string",
        "description": "Description of the product",
    },
    "material": {
        "type": "string",
        "description": "Material of the product",
    },
    "temperature": {
        "type": "string",
        "description": "Temperature rating of the thermostat",
    },
    "connector": {
        "type": "string",
        "description": "Connector type of the product",
    },
    "type": {
        "type": "string",
        "description": "Type of the product",
    },
    "sub_type": {
        "type": "string",
        "description": "Sub type of the product",
    },
    "pins": {
        "type": "string",
        "description": "Number of pins in the connector",
    },
    "include_housing": {
        "type": "string",
        "description": "Whether the product includes housing",
    },
    "include_gaskets": {
        "type": "bool",
        "description": "Whether the product includes gaskets",
    },
    "features": {
        "type": "string",
        "description": "Features of the product",
    },
    "compatibility": {
        "type": "string",
        "description": "Compatibility of the product",
    },
    "price": {
        "type": "string",
        "description": "Price of the product",
    },
    "currency": {
        "type": "string",
        "description": "Currency of the price",
    },
    "oem": {
        "type": "string",
        "description": "Original Equipment Manufacturer (OEM) of the product",
    },
    "car_model": {
        "type": ["string", "array"],
        "description": "Car model(s) compatible with the product",
    },
    "cross_reference": {
        "type": "array",
        "items": {
            "type": "string"
        },
        "description": "Cross-reference numbers for the product",
    },
    "additional_data": {
        "type": "string",
        "description": "Additional data related to the product",
    },
    "detail_desc": {
        "type": "string",
        "description": "Detailed product description without all cross-references, in 20 words",
    }
}
required_keys = ["name", "product_number", "sku", "description"]

import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from typing import Literal
from datetime import datetime
from openai import OpenAI
client = OpenAI()
now = datetime.now() # get current timestamp
current_timestamp = now.strftime('%y_%m_%d_%H_%M')
           
    
###################### Work with files
# init folders and files
data_to_save_folder = f'files/results/saved_{current_timestamp}/'
if not os.path.exists(data_to_save_folder):
    os.makedirs(data_to_save_folder)
file_to_save_without_enrichment = data_to_save_folder + 'serper_collected.json'
file_to_save_filtered = data_to_save_folder + 'serper_collected_filtered.json'
file_to_save_enriched = data_to_save_folder + 'enrichment_results.json'
file_to_save_with_page_data = data_to_save_folder + 'all_collected_results.json'

data_to_read_folder = f'files/read/'
table_file = data_to_read_folder + 'df_1106.xlsx' # table     ##### Ready To Run
header_row = 2 # which row represents the column names? Index from 0
all_data_file_from_motorad = data_to_read_folder + 'full_data_parts.json' # all scraped motorad data


# read files
initial_table = pd.read_excel(table_file, header=header_row)
with open(all_data_file_from_motorad, 'r') as file:
    motorad_all_items = json.load(file)
    
    
###################### Serper and oai init
## serper options
scrape_with_serper = True ############################################### Important
serper_base_url = "https://google.serper.dev/"
content_types = ['search', 'shopping', 'images']
serper_limit = 2 # Filter items to search wiith serper. Set to 0 to search all products

# template for serper search. Used in function get_query. 
query_template = """{brand_name} {major} {company_num} site:{website}"""
query_template_full = """("{brand_name} {major}" OR "{brand_name} {minor}" OR "{brand_name} {long_desc}") "{company_num}" "price" ("part" OR "Part #" OR "Part number") ("MPN" OR "Manufacturer Part Number" OR "SKU") (site:autozone.com OR site:{website} OR site:"")"""
results_count_by_content_type = {
    'search': 10,
    'shopping': 10,
    'images': 50
} # AK: Need to bee different 'search' = 10  'shopping' = 10, 'images'= 50]  
    # Fixed 18.06
    
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
gpt_model = 'gpt-4o'
oai_key = os.environ['OPENAI_API_KEY']
    

###################### Define functions
# Function for scraping. Used before enrichment
def scrape_url(url:str, method: Literal['jina', 'serper'] = 'jina')-> str: # 18.06: appended jina, optional.
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
def get_product_unstructured_data_to_json(schema, data, model = gpt_model)->dict: # AK
    system_message = (
        f"You are a data analyst API."
        f"Your task is to extract info from a raw text"
        f"Please respond with your analysis directly in JSON format "
        f"(without using Markdown code blocks or any other formatting). "
        f"If you can't find required keys in the text, just use empty text or empty list as a default value in the schema. "
        f"The JSON schema should include: {schema}.")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": str(data)}
            ],
    )
    result =  response.choices[0].message.content.strip()
    try:
        return json.loads(result)
    except:
        return result


# function to scrape url and validate
def scrape_item(
        url,
        schema: dict = prooduct_json_schema, 
        required_keys:list = required_keys
    ) -> dict:
    results = {
        'jina_scrape': {},
        'serper_scrape': {}
    }
    jina_page_text = scrape_url(url, method = 'jina')
    jina_page_json = get_product_unstructured_data_to_json(
        schema= schema,
        data = jina_page_text
    )
    results['jina_scrape'] = jina_page_json
    
    for required_key in required_keys:
        if not jina_page_json.get(required_key):
            break
    else: # works if not breaked
        return results
    
    serper_page_text = scrape_url(url, method='serper')
    serper_page_json = get_product_unstructured_data_to_json(
        schema= schema,
        data = serper_page_text
    )
    results['serper_scrape'] = serper_page_json
    
    return results


# TODO 18.06: Enrichment step: to append this function as llm function tool
# function to compare 2 json objects. dynamically compare results and available data. use python without llm
# def compare_objects(item1:dict, item2:dict)->float: # ? # add logic
#     matched_keys = 0
#     matched_values = 0
#     for key, value in item1.items():
#         if key in item2:
#             matched_keys += 1
#             if value == item2[key]:
#                 matched_values += 1
#     return matched_values/matched_keys if matched_keys else 0
    

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
                    key: value if not pd.isna(value) else None
                })
            data_to_use.append(data_temp)
            
    return data_to_use


# Create a query from the template for serper search
def get_query(item: dict) -> str:
    """
    Create a search query from a template using the given item dictionary.
    This function takes a dictionary with various keys and formats a query string 
    based on the specified template.
    Args:
        item (dict): A dictionary containing the keys needed to fill the query template.

    Returns:
        str: A formatted query string.
    """
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
    dict: Dictionary containing search results.
    """
    if not isinstance(query, str):
        return None
    
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
            response = requests.request("POST", serper_url, headers=serper_headers, data=payload)
            response_obj: dict = response.json()
        except Exception as e:
            print(f"An error occurred while making the request to {serper_url}: {e}")
            continue
        
        response_values = list(response_obj.values())
        result[content_type] = response_values[1] # 0 is {SearchParameters}
        
    return result


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
    for item in tqdm(items_to_search[:total]):
        query = get_query(item)
        results = get_item_search_result(query, item, content_types)
        if results:
            collected.append(results)
        
    metadata = {
        'types': content_types,
        'time': current_timestamp,
        'count_by_type': results_count_by_content_type,
        'input_items': total,
        'payload': headers_search_preferences
    }
    
    final_results = {
        'metadata': metadata,
        'collected': collected
    }
    save_result(final_results, file_to_save_without_enrichment)
    
    return final_results


# function to filter items. In this stage filters only duplicate urls. Collects all type sources in one kwy
def filter_sources(search_results: dict) -> dict:
    """
    Filter items by removing duplicate URLs and collect all type sources into one key.

    This function filters out duplicate URLs from the search results and collects 
    all source types into a single list of sources for each item.

    Args:
        search_results (dict): A dictionary containing the search results to be filtered.

    Returns:
        dict: The filtered search results with duplicates removed and sources combined.
    
    """
    collected:list[dict] = search_results['collected']
    total_sources = 0
    filtered_items = []
    for item in collected:
        item_data = item['original']
        filtered_sources = []
        checked_links = set()
        for source_type in content_types:
            sources = item.get(source_type) or []
            for source in sources:
                source_link = source.get('link')
                if (not source_link) or (source_link in checked_links):
                    sources.remove(source)
                else:
                    checked_links.add(source_link)
            filtered_sources.extend(sources)

        if filtered_sources:
            filtered_items.append({
                'query': item['query'],
                'original': item_data,
                'sources': filtered_sources
            })
            total_sources += len(filtered_sources)
            
    search_results['collected'] = filtered_items
    print(f'{len(filtered_items)} items and {total_sources} sources after filtering')
    save_result(search_results, file_to_save_filtered)
    
    return search_results
    
    
# function to return collected all results
def all_results_with_page_content(search_results: dict, content_types:list) -> dict:
    collected_items =  search_results['collected']
    count = 1
    items_by_url = dict()
    for item in collected_items:
        for content_type in content_types:
            sources = item[content_type] or []
            for source in sources:
                print('\rpage N', count, end='')
                count += 1
                link = source['link'].split('?')[0].strip('/') # get url main part
                if items_by_url.get(link): # if duplicated, update with new content_type
                    items_by_url[link][content_type] = source
                else: # else create new key-link and get page data
                    items_by_url[link] = {
                        'query': item['query'],
                        'original': item['original'],
                        content_type: source
                    }

                    page_data_dict = scrape_item(link)
                    items_by_url[link].update(page_data_dict)
    
    print()
    final_schema = {
        'metadata': search_results['metadata'],
        'products': items_by_url
    }
    
    save_result(final_schema, file_to_save_with_page_data)
    return final_schema

    
###################### The Main function
def main(
        serper_limit:int = serper_limit,
        content_types:list = content_types
    ) -> dict:
    print("Merging table and json...")
    merged_data = merge_table_and_json(initial_table)
    
    print('Serper requests...')
    search_results = get_all_items_search_results(merged_data, content_types, serper_limit=serper_limit)
    
    print('Filtering sources...')
    filtered_sources_with_metadata = filter_sources(search_results.copy()) # 
    
    print('Receiving page contents...')
    result_with_page_contents = all_results_with_page_content(search_results, content_types)
    
    print("Finished!")
    return result_with_page_contents
    
####
result = main()
print(result['metadata'])
