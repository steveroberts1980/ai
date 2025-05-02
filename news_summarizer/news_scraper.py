import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from bs4 import BeautifulSoup
import requests

def get_text(url):
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    }

    # Make the request and create BeautifulSoup object
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the div with class starting with "article-body"
    article_div = soup.find('div', class_=lambda x: x and x.startswith(('article-body', 'article__content')))

    if article_div:
        # Extract the text with preserved formatting
        article_text = article_div.get_text(separator=' ', strip=True)
        return article_text
    else:
        return "Article body not found"

def initialize_ai():
    # Initialization

    load_dotenv(override=True)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    MODEL = "gpt-4o-mini"
    openai = OpenAI()

    return openai, MODEL

summary_length = "50"

def get_system_message():
    global summary_length
    system_message = f"You are a news assistant who expertly summarizes news articles in {summary_length} words or less. "
    system_message += "When given the text of a news article, you will summarize the main points and any important information. "
    system_message += "You will ensure the summary is not biased and only includes information that is objective and that allows the "
    system_message += "reader to form their own opinion."
    return system_message


get_news_function = {
    "name": "get_text",
    "description": "Get the news text from a website for a given URL. Call this whenever the user provides a URL or asks for you to summarize the article.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the news article",
            },
        },
        "required": ["url"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": get_news_function}]

llm, MODEL = initialize_ai()

def chat(message, history):
    messages = [{"role": "system", "content": get_system_message() }] + history + [{"role": "user", "content": message}]
    response = llm.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = llm.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    url = arguments.get('url')
    article_text = get_text(url)
    response = {
        "role": "tool",
        "content": json.dumps({"article_text": article_text}),
        "tool_call_id": tool_call.id
    }
    return response

# Get the value of the gr.Dropdown and set to the summary_length


def updateSummaryLength(value):
    global summary_length
    summary_length = int(value)
    print(f'updated summary length to {summary_length}')
    gr.update(visible=True)

def getTitle(value):
    if not value:
        value = "50"
    return f"## News Summarizer {value}"

with gr.Blocks() as demo:
    gr.ChatInterface(fn=chat, type="messages")
    dropdown = gr.Dropdown(["50", "75", "100", "125"], label="Summary Length", value=summary_length, interactive=True)
    dropdown.change(updateSummaryLength, dropdown)
    #md = gr.Markdown(getTitle, inputs=[dropdown])


#gr.ChatInterface(fn=chat, type="messages").launch()

if __name__ == "__main__":
    demo.launch()

# Test the function
# url = 'https://www.reuters.com/business/stockpiling-ahead-tariffs-likely-hurt-us-economy-first-quarter-2025-04-30/'
# url = 'https://www.reuters.com/world/china/china-creates-list-us-made-goods-exempt-125-tariffs-sources-say-2025-04-30/'
# text = get_text(url)
# print(text)
