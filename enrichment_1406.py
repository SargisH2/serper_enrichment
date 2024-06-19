import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

now = datetime.now() # get current timestamp
current_timestamp = now.strftime('%y_%m_%d_%H_%M')

###################### Work with files
# init folders and files
data_to_save_folder = f'files/saved_{current_timestamp}/'
if not os.path.exists(data_to_save_folder):
    os.makedirs(data_to_save_folder)
file_to_save_without_enrichment = data_to_save_folder + 'serper_collected.json'
file_to_save_filtered = data_to_save_folder + 'serper_collected_filtered.json'
file_to_save_enriched = data_to_save_folder + 'enrichment_results.json'

data_to_read_folder = f'files/read/'
table_file = data_to_read_folder + 'df_1106.xlsx' # table
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
serper_limit = 2 # items to search wiith serper. Set to 0 to search all products
# template for serper search. Used in function get_query. 
query_template = """("{brand_name} {major}" OR "{brand_name} {minor}" OR "{brand_name} {long_desc}") "{company_num}" "price" ("part" OR "Part #" OR "Part number") ("MPN" OR "Manufacturer Part Number" OR "SKU") (site:autozone.com OR site:{website} OR site:"")"""
results_count_by_content_type = 10
serper_key = os.environ['SERPER_API_KEY']
headers_search_preferences = {
    "location": "United States", # default United States
    "gl": "us", # default us
    "hl": "en", # default en
    "num": results_count_by_content_type, # default 10
}
serper_headers = {
    'X-API-KEY': serper_key,
    'Content-Type': 'application/json'
}

## enrichment options
gpt_model = 'gpt-4o'
oai_key = os.environ['OPENAI_API_KEY']
enrichment_limit = 5 # how many sources will be used for enrichment. Max 2 gpt requests and 1 serper request to gpt by every source
similarity_prompt_template = """
item1:
{}

item2:
{}"""

json_schema = {
    "similarity": "how the items are similar? one of same/almost_same/similar/not_similar",
    "similarity_score": "a number representing how the items are similar in percents, from 0 to 100",
    "source_quality": "Source quality good/medium/bad"
}
images_schema = {
    "images_similarity_score": "a number representing how the items are similar in percents, from 0 to 100",
    "info": "a snippet, providing key-points, why are similar and why not "
}

system_message = (
            f"You are a data analyst API."
            f"Your task is to define similarity and source quality"
            f"Similarity is one of [same, almost_same, similar, not_similar]"
            f"Source quality is one of [good, med, bad]. for example, ebay is a bad source, because there are used products, but Autozone is a good, because it is a car parts market with new and quality products"
            f"Help user to analyse 2 items. First is a product, second is a search result."
            f"Please respond with your analysis directly in JSON format "
            f"(without using Markdown code blocks or any other formatting). "
            f"The JSON schema should include: {json_schema}.")

img_similarity_prompt = (
            f"You are a expert in car parts classification."
            f"our task is to compare images and send an answer-how much are the parts similar."
            f"Don't compare images - like comparing size or quality"
            f"images_similarity_score is a number representing how much these parts are similar in percents(from 0 to 100)"
            f"info is a text snippet in 4 key points-why are the parts similar and why not"
            f"Please respond with your analysis directly in JSON format "
            f"(without using Markdown code blocks or any other formatting). "
            f"The JSON schema should include: {images_schema}.")


###################### Define functions
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
        company_num = item.get('company_num'),
        website = item.get('competitor_web_domain') or ""
    )
    return query


# function to search with serper and return a complete item with the original
def get_item_search_result(
        query: str, 
        original_item: dict
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
    
    payload =  headers_search_preferences.copy() # avoid editing the original payload template
    payload['q'] = query
    payload = json.dumps(payload)
    
    result = { # will be updated with query results
        'query': query,
        'original': original_item
    }
    
    for content_type in content_types:
        serper_url = serper_base_url + content_type
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
            items_to_enrichment: list,
            serper_limit: int = 0
        ) -> dict:
    """
    Search with Serper for all items and save the results in a JSON file.

    This function takes a list of items, creates a search query for each item and
    and collects the Serper results. The results are then saved to a JSON file.

    Args:
        items_to_enrichment (list): A list of items to be enriched with search results.
        serper_limit (int, optional): The maximum number of items to search. Defaults to 0, which means no limit.

    Returns:
        dict: A dictionary containing the metadata and collected search results.
    """
    collected = []
    total = serper_limit or  len(items_to_enrichment)
    for item in tqdm(items_to_enrichment[:total]):
        query = get_query(item)
        results = get_item_search_result(query, item)
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


############################################################### enrichment
# Class for enrichment interactions with openai 
class OpenaiEnrichment:
    def __init__(self, client:OpenAI, model: str):
        """
        Initialize the OpenaiEnrichment with a specified model and client.
        
        Parameters:
        model (str): The model to be used for generating completions.
        client: The client used to interact with the API.
        """
        self.model = model
        self.client = client
        self.total_requests = 0 # for metadata

    def __create_image_data(self, url: str) -> dict:
        """
        Create image data dictionary from a URL.
        
        Parameters:
        url (str): The URL of the image.
        
        Returns:
        dict: Dictionary containing the image type and URL or base64 encoded string.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
            
    def images_prompt(self, prompt: str, *urls: str) -> str:
        """
        Generate a response from a chat model using a prompt and multiple images.
        
        Parameters:
        prompt (str): The text prompt to be sent to the model.
        *urls (str): Variable length argument list of URLs or file paths for the images.
        
        Returns:
        str: The content of the response generated by the model.
        """
        self.total_requests += 1
        images = [self.__create_image_data(url) for url in urls]

        response = self.client.chat.completions.create(
            model=self.model,
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
    
    def ask_llm(self, prompt: str, system_message:str = system_message) -> str:
        """
        Send a prompt to a Language Learning Model (OpenAI) and return the response.

        This function increments the total request count, sends the prompt to the LLM
        with an optional system message, and returns the LLM's response.

        Args:
            prompt (str): The prompt to send to the LLM.
            system_message (str, optional): The system message to send to the LLM. Defaults to a predefined system message.

        Returns:
            str: The response from the LLM, stripped of leading and trailing whitespace.
        """
        self.total_requests += 1
        client = self.client
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
        )
        return response.choices[0].message.content.strip()


# Function for serper scraping. Used with enrichment
def serper_scraping(url:str):
    payload = json.dumps({
        "url": url
    })
    serper_crape_url = "https://scrape.serper.dev"
    try:
        response = requests.request("POST", serper_crape_url, headers=serper_headers, data=payload)
        return response.json()
    except Exception as e:
        return f"ERROR: {response},\n {e}"
    

def start_enrichment(
        data:dict, 
        enrichment_client: OpenaiEnrichment,
        enrichment_limit:int,
        get_pages_data:bool
    ) -> dict:
    """
    Enrich data with additional keys using OpenAI enrichment and Serper(optional).

    This function processes each item in the filtered data, uses OpenAI's LLM to enrich the data with
    new information from sources, and performs image similarity analysis. The enriched data is saved
    to a file and returned.

    Args:
        data (dict): The input data containing items to be enriched.
        enrichment_client (OpenaiEnrichment): An instance of the OpenaiEnrichment class to interact with OpenAI's LLM.
        enrichment_limit (int): The maximum number of items to enrich.
        get_pages_data (bool): Scrape pages with serper
        
    Returns:
        dict: The enriched data.
    """
    filtered_items = data['collected']
    limit = enrichment_limit
    gpt_errors = 0
    count = 0
    items_enriched = []
    for item in filtered_items:
        original_item_data = item['original']
        origin_photo = original_item_data.get('angle_view').replace('images/', 'https://images.motorad.com/MotoRad-stills/')
        new_sources = []
        for source in item['sources']:
            count += 1
            print(f"\r{count}/{limit}", end = '') # show the progress
            if count > limit:
                break
            
            try: # validation
                ########### json objects similarity
                json_object_prompt = similarity_prompt_template.format(original_item_data, source)
                json_object_result = json.loads(
                    enrichment_client.ask_llm(json_object_prompt)
                ) 
                source.update(json_object_result)

                ############ images similarity
                source_photo = source.get('imageUrl')
                if source_photo:
                    images_result = json.loads(
                        enrichment_client.images_prompt(img_similarity_prompt, origin_photo, source_photo)
                    ) 
                    source.update(images_result)

                ########### serper scrape
                if get_pages_data:
                    url = source.get('link')
                    page_content = serper_scraping(url)
                    source.update({
                        "page_content": page_content
                    })
                    
                new_sources.append(source)
            except Exception as e:
                if not gpt_errors % 5:
                    print(f"\nGpt error N{gpt_errors+1}.\nSource: {source},\n\n Error: {e}")
                gpt_errors += 1
            
        item['sources'] = new_sources
        items_enriched.append(item)
        if count > limit:
            data['metadata'].update({ # update metadata
                'gpt_requests': enrichment_client.total_requests,
                'gpt_errors': gpt_errors,
                'scrape_with_serper': scrape_with_serper
            })
            data['collected'] = items_enriched
            
            print()
            print(f'Gpt errors: {gpt_errors}')
            break
        
    save_result(data, file_to_save_enriched)
    return data
    
    
###################### The Main function
def main(
        enrichment_limit:int = enrichment_limit,
        gpt_model:str = gpt_model,
        oai_key:str = oai_key,
        serper_limit:int = serper_limit
    ) -> dict:
    print("Merging table and json...")
    merged_data = merge_table_and_json(initial_table)
    print('Serper requests...')
    search_results = get_all_items_search_results(merged_data, serper_limit=serper_limit)
    print('filtering sources...')
    filtered_sources_with_metadata = filter_sources(search_results)
    
    print("OpenAI initialization...")
    client = OpenAI(api_key=oai_key)
    enrichment_client = OpenaiEnrichment(
        client=client,
        model=gpt_model
    )
    print('Starting enrichment...')
    final_results = start_enrichment(
        data=filtered_sources_with_metadata,
        enrichment_client=enrichment_client,
        enrichment_limit=enrichment_limit,
        get_pages_data = scrape_with_serper
    )
    print("Finished!")
    return final_results
    
####
result = main()
print(result['metadata'])
