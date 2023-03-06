import os
import json 
import openai
import backoff
import requests
import pandas as pd
from tqdm import tqdm
from openai.error import RateLimitError
from openai.error import APIConnectionError
from openai.error import APIError
from openai.error import Timeout

# hot restart
@backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError, Timeout))
def get_oai_completion(prompt, temperature, max_tokens):
    oai_key = os.environ.get('OPENAI_API_KEY')
    openai.api_key = oai_key
    # model_name = "text-davinci-003"
    model_name = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response['choices'][0]['message']['content']

# 0 "Welcome to the Alexa Prize Social Bot! What do you want to talk about?", [fallback start]
# 1 "music", [user utterance turn n-3]
# 2 "I am enjoying listening to rock music by Pink Floyd these days. what music do you listen to?", [bot utterance turn n-3]
# 3 "i like rock music", [user utterance turn n-2]
# 4 "You&apos;ve a great taste! If you don&apos;t mind me asking, what is your favorite rock music song?", [bot utterance turn n-2]
# 5 "i like songs from the Backstreet Bohys", [user utterance turn n-1]
# 6 "I remember listening this song a while back! I connect with prime music when ever I am in a music mood! How do you get your music?", [bot utterance turn n-1]
# 7 "i listen on Spotify" [user utterance turn n]

# construct a conversation history for the entity tracker
def get_entity(cur_uter_idx, subset):
    if cur_uter_idx < 7:
        conversation_history = subset.iloc[0:cur_uter_idx+1]['utterance'].tolist()
    else:
        conversation_history = subset.iloc[cur_uter_idx-7:cur_uter_idx+1]['utterance'].tolist()
        
    str_conv = '\n'.join(conversation_history)
    prompt =f"""You are given a dialog between user1 and user2. You need to identify the current entity being discussed.

    [Dialog]
    {str_conv}

    [Topic]
    The entity should be a short noun phrase with less than 10 words.
    entity="""

    # send a request to the entity tracker
    entity = get_oai_completion(prompt, 0.3, 15)
    return entity

def get_cosmo_uter(cur_uter_idx, subset, entity):
    # you can simply put all the dialog hitherto to cosmo for response generation
    # because the cosmo already handles all the sliding window strategy
    # However, you have to make the first utterance a bot utterance
    url = 'http://cosmo-alb-942116337.us-east-1.elb.amazonaws.com/cosmo'
    
    conversation_history = subset.iloc[:cur_uter_idx+1]['utterance'].tolist()
    dialog_act = {'topic': ''}
    content = {"dialog_act": dialog_act, "entity": entity, "conversation_history": conversation_history}

    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(content))
    cosmo_utterance = response.json()['response']
    return cosmo_utterance

def get_query(cur_uter_idx, subset, cosmo_utterance, entity):
    # use GPT-3 to generate a response, with the conversation history, entity, and cosmo response
    # specifically, we use last bot utterance, this user utterance, this cosmo utterance, and the entity to generate a query

    last_bot_utterance = subset.iloc[cur_uter_idx-1]['utterance']
    this_user_utterance = subset.iloc[cur_uter_idx]['utterance']
    conversation_history = [last_bot_utterance, this_user_utterance, cosmo_utterance]

    prompt = f"""You are given a short dialog between a user and a bot for a discussion on {entity}. Convert the bot response to a question to use for internet search to get relevant knowledge for continuing the response.

    [Dialog]
    {conversation_history}

    [Query]
    query="""

    query = get_oai_completion(prompt, 0.7, 50).strip()
    return query
