import os
import asyncio
from openai import AsyncOpenAI
os.environ['OPENAI_API_KEY'] = os.environ['PROJECT_OAI_KEY']


async def send_async_message(client:AsyncOpenAI, user_message, system_message = "", model='gpt-3.5-turbo'):
    return await client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
    )
    
async def run_chat_completions(client, prompts):
    results = []
    calls = [send_async_message(client, prompt['user'], prompt['system']) for prompt in prompts]
    for completed_task in asyncio.as_completed(calls):
        response = await completed_task
        results.append(response.choices[0].message.content.strip())
    return results

async def run_async_tasks(prompts: list): 
    client = AsyncOpenAI()
    results = await run_chat_completions(client, prompts)
    return results