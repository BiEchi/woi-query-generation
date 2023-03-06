import json 
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

for split in ['test', 'valid', 'train']:
    print(f'Processing {split} dataset...')
    path = f'data/raw/{split}.jsonl'
    with open(path, 'r') as json_file:
        json_list = [json.loads(item) for item in list(json_file)]

    print('building the cleaned dataset...')
    dataset = []
    for i in range(len(json_list)):
        key = list(json_list[i].keys())[0]
        content = json_list[i][key]
        content['id'] = key
        dataset.append(content)
        
    data = pd.DataFrame(dataset)
    data_new = data
    data_new.drop(columns=['apprentice_persona', 'start_timestamp'], inplace=True)
    data_new['length'] = data_new['dialog_history'].apply(lambda x: len(x))

    print('de-aggregating the dataset...')
    df_deaggr = pd.DataFrame(columns=['action', 'utterance', 'query', 'id'])
    df_deaggr['query'] = False
    for i in tqdm(range(len(data_new))):
        utterance_existance = False
        # if the first utterance is a user utterance, add a bot utterance on top
        if data_new['dialog_history'].iloc[i][0]['action'] == 'Apprentice => Wizard':
            df_deaggr = df_deaggr.append({'action': 'Wizard => Apprentice', 'utterance': 'Hello, welcome to Alexa social bot. What do you want to chat?', 'id': data_new['id'].iloc[i], 'query': False}, ignore_index=True)
        for j in range(len(data_new.iloc[i]['dialog_history'])):
            utterance = data_new['dialog_history'].iloc[i][j]
            # if data_new['id'].iloc[i] == '146585':
                # print(utterance['action'], utterance['text'])
            if utterance['action'] == 'Wizard => Apprentice' or utterance['action'] == 'Apprentice => Wizard':
                utterance_existance = True
            # each dialog should start with a bot utterance
            if utterance_existance == True:
                if utterance['action'] in ['Wizard => Apprentice', 'Apprentice => Wizard']:
                    df_deaggr = df_deaggr.append({'action': utterance['action'], 'utterance': utterance['text'], 'id': data_new['id'].iloc[i], 'query': False}, ignore_index=True)
                # we think the query is only valid when the last utterance is a user utterance
                if utterance['action'] == 'Wizard => SearchAgent':
                    if df_deaggr['action'].iloc[-1] == 'Apprentice => Wizard':
                        df_deaggr['query'].iloc[-1] = True

    print(f'Number of utterances: {len(df_deaggr)}')
    df_deaggr['action'] = df_deaggr['action'].apply(lambda x: 'bot' if x == 'Wizard => Apprentice' else 'user')
    # save the dataframe to jsonl file line by line
    df_deaggr.to_json(f'data/processed/{split}.jsonl', orient='records', lines=True)
