import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from deep_translator import GoogleTranslator
import fasttext
from huggingface_hub import hf_hub_download
import spacy
from tqdm.auto import tqdm
from collections import Counter
import json
import ast
import requests

tqdm.pandas()  # Enable progress_apply for pandas

# Load the pre-trained English model
nlp = spacy.load('en_core_web_md')


translator = GoogleTranslator(source='ar', target='en')

from dotenv import load_dotenv

load_dotenv()

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

# write a function that build pairs with the same file name for different folders
def get_trans_dict(directory1='datasets/transcripts', directory2='datasets/transcripts_en'):
    trans_dict = {}
    files_set2 = set([file for file in os.listdir(directory2) if file.endswith('.tsv')])

    for file1 in os.listdir(directory1):
        if file1.endswith('.tsv'):
            if file1 in files_set2:
                trans_dict[os.path.join(directory1, file1)] = os.path.join(directory2, file1)
    return trans_dict

trans_dict = get_trans_dict()

def plot_lines_distribution(transcripts):

    # Calculate bin edges using logspace
    bin_edges = np.logspace(np.log10(transcripts['lines'].min()),
                          np.log10(transcripts['lines'].max()),
                          10)

    # Print bin ranges
    print("\nBin ranges:")
    for i in range(len(bin_edges)-1):
        print(f"Bin {i+1}: {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}")

    # draw a histogram with exponential scale
    plt.figure(figsize=(12, 6))  # Made figure wider for better label visibility
    plt.hist(transcripts['lines'], bins=bin_edges, edgecolor='black')
    plt.xscale('log')

    # Create labels for x-ticks showing bin ranges
    labels = [f'{bin_edges[i]:.1f}\n-\n{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]
    plt.xticks(np.sqrt(bin_edges[:-1] * bin_edges[1:]), labels, rotation=45)

    plt.title('Distribution of Lines per Transcript (Log Scale)')
    plt.xlabel('Number of Lines (bin ranges)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

def trans(text, trans_text):
    if trans_text == '' or trans_text is None:
        return translator.translate(text[:1000])
    else:
        return trans_text

def get_lang_id(text):
    subtext = ' '.join(text.split('\n')[:50])
    return model.predict(subtext)[0][0]

def read_transcript_files(directory='datasets/transcripts'):
    # Dictionary to store DataFrames
    transcript_dfs = []
    print(f"Reading transcripts")
    # List all files in directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.tsv'):
            try:            
                # Create full file path
                file_path = os.path.join(directory, filename)
                
                # Read TSV file into DataFrame
                df = pd.read_csv(file_path, sep='\t')
                
                if file_path in trans_dict:
                    df_en = pd.read_csv(trans_dict[file_path], sep='\t')
                    df = pd.merge(df, df_en, on='id', how='left', suffixes=('', '_en'))
                    df = df [['id', 'text', 'text_en']]
                
                else:
                    df['text_en'] = ''
                
                df['lines'] = df['text'].apply(lambda x: len(x.split('\n')) + 1) # count '\n' in text
                
                # drop rows with no lines
                df = df[(df['lines'] >= 50) & (df['lines'] <= 850)]
                
                df = df.sample(10)
                
                df['lang_id'] = df['text'].apply(lambda x: get_lang_id(x))
                
                df = df[(df['lang_id'].str.contains('_Arab')) & (df['lang_id'].str.contains('__label__a'))]
                
                df['text_en'] = df.apply(lambda x: trans(x['text'], x['text_en']), axis=1)
                                 
                transcript_dfs.append(df)
                
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
    # concatenate all DataFrames
    return pd.concat(transcript_dfs, ignore_index=True)

def extract_entities(text):
    doc = nlp(text)
    people = set([])
    organizations = set([])
    locations = set([])
    others = set([])
    locations_labels = {"GPE", "LOC"}

    # Iterate over the entities in the doc
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.add(ent.text)
        elif ent.label_ == "ORG":
            organizations.add(ent.text)
        elif ent.label_ in locations_labels:
            locations.add(ent.text)
        else:
            others.add(ent.text)
            
    return {"people": list(people), "organizations": list(organizations), "locations": list(locations), "others": list(others)}
  
def predict_toxicity(text):
    '''
    curl -X 'POST' \
    'http://10.127.105.182:7444/predict_toxicity' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": "You'\''re completely wrong and stupid!"
    }'
    '''
    resp = requests.post('http://10.127.105.182:7444/predict_toxicity', json={'text': text})
    score = -1
    if resp.status_code == 200:
     resp_obj = resp.json()
     if 'overall_toxicity' in resp_obj:
       score = resp_obj['overall_toxicity']
    return score

def enriched_df(df):
    # Add tqdm progress bar to the entity extraction
    df['entities'] = df['text_en'].progress_apply(lambda x: extract_entities(x))
    df['locations'] = df['entities'].apply(lambda x: x['locations'])
    df['people'] = df['entities'].apply(lambda x: x['people'])
    df['organizations'] = df['entities'].apply(lambda x: x['organizations'])
    df['others'] = df['entities'].apply(lambda x: x['others'])
    df.drop(columns=['entities'], inplace=True)
    return df
  
def get_most_mentioned_themes_series(df):
    counter = Counter()
    for index, row in df.iterrows():
        try:  
            themes = ast.literal_eval(row['theme'])
            counter.update(themes)
        except:
          
            continue
    return dict(counter)
  
def get_most_mentioned_locations_series(df):
    # generate a Counter object from the locations series
    counter = Counter()
    for index, row in df.iterrows():
        # Convert string representation of list to actual list using ast.literal_eval
        try:
            # Handle string representation of Python lists
            locations = ast.literal_eval(row["locations"])
            counter.update(locations)
        except:
            # Skip invalid entries
            continue
    # return dictionary with the most common locations and frequencies
    return dict(counter.most_common(50))
  
def get_most_mentioned_people_series(df):
    # generate a Counter object from the people series
    counter = Counter()
    for index, row in df.iterrows():
        # Convert string representation of list to actual list using ast.literal_eval
        try:
            # Handle string representation of Python lists
            locations = ast.literal_eval(row["people"])
            counter.update(locations)
        except:
            # Skip invalid entries
            continue
    # return dictionary with the most common locations and frequencies
    return dict(counter.most_common(50))
  
def draw_most_mentioned_people():
    df = pd.read_csv('subsampling_qna_enriched.tsv', sep='\t')
    counter = get_most_mentioned_locations_series(df)
    counter = get_most_mentioned_people_series(df)
    #counter = get_most_mentioned_organizations_series(df)
    # render a bar chart for the most mentioned locations with matplotlib
    # render labels on the x-axis
    plt.figure(figsize=(12, 10))
    plt.bar(counter.keys(), counter.values())
    plt.xticks(rotation=90)
    # make more room for the x labels (height)
    plt.tight_layout()
    # add value on the top of the bars
    for i, v in enumerate(counter.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.show()
 
  
def get_most_mentioned_organizations_series(df):
    # generate a Counter object from the people series
    counter = Counter()
    for index, row in df.iterrows():
        # Convert string representation of list to actual list using ast.literal_eval
        try:
            # Handle string representation of Python lists
            locations = ast.literal_eval(row["organizations"])
            counter.update(locations)
        except:
            # Skip invalid entries
            continue
    # return dictionary with the most common locations and frequencies
    return dict(counter.most_common(50))
  
def get_most_mentioned_others_series(df):
    counter = Counter()
    for index, row in df.iterrows():
        counter.update(row['theme'])
    return dict(counter.most_common(50))
  
def predict_theme(text, threshold=0.18):
  '''
  curl -X 'POST' \
  'http://10.127.105.182:7555/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Through the meetings, it became clear that the Palestinians tend not to care about the US election results, considering that there is no difference in the policies of the Democratic and Republican parties regarding the Palestinian issue",
  "candidate_labels": [
    "Business",
    "War",
    "Religion",
    "Entertainment",
    "Sport",
    "Culture",
    "Travels",
    "Science",
    "Education",
    "Politics"
  ]
}'
Transform to code
  '''
  resp = requests.post('http://10.127.105.182:7555/classify', json={'text': text})
  results = {}
  if resp.status_code == 200:
    resp_obj = resp.json()
    #if len(resp_obj['scores']) > 0:
    #  print(f"Top confidence: {resp_obj['scores'][0]}")
    for label, confidence in zip(resp_obj['labels'], resp_obj['scores']):
      if confidence > threshold:
        results[label] = confidence
  return results

def get_toxicity_plot(toxicity_df):
    # Calculate bin edges using linear space instead of logspace
    bin_edges = np.linspace(toxicity_df['toxicity'].min(),
                           toxicity_df['toxicity'].max(),
                           10)

    # Print bin ranges
    print("\nBin ranges:")
    for i in range(len(bin_edges)-1):
        print(f"Bin {i+1}: {bin_edges[i]:.5f} - {bin_edges[i+1]:.5f}")

    # draw a histogram with linear scale
    plt.figure(figsize=(12, 6))
    plt.hist(toxicity_df['toxicity'], bins=bin_edges, edgecolor='black')
    plt.yscale('log')  # Add this line to make y-axis logarithmic
    # Removed xscale('log') to keep linear scale

    # Create labels for x-ticks showing bin ranges
    labels = [f'{bin_edges[i]:.5f}\n-\n{bin_edges[i+1]:.5f}' for i in range(len(bin_edges)-1)]
    # Changed tick positions to use bin centers
    tick_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.xticks(tick_positions, labels, rotation=45)

    plt.title('Distribution of Toxicity per Transcript')  # Removed "(Log Scale)"
    plt.xlabel('Toxicity level (bin ranges)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# Usage example:
#transcripts = read_transcript_files()


if __name__ == "__main__":
    #df = transcripts = read_transcript_files()
    #df.to_csv('subsampling_qna.tsv', sep='\t', index=False)
    
    
    #df = pd.read_csv('subsampling_qna.tsv', sep='\t')
    #df = enriched_df(df)
    #df.to_csv('subsampling_qna_enriched.tsv', sep='\t', index=False)
    
    #df = pd.read_csv('subsampling_qna_enriched.tsv', sep='\t')
    #df['theme'] = df['text_en'].progress_apply(lambda x: list(predict_theme(x).keys()))
    #df.to_csv('subsampling_qna_enriched_with_theme.tsv', sep='\t', index=False)
    
    #df = pd.read_csv('subsampling_qna_enriched_with_theme.tsv', sep='\t')
    #df['toxicity'] = df['text_en'].progress_apply(lambda x: predict_toxicity(x))
    #df.to_csv('subsampling_qna_enriched_with_theme_and_toxicity.tsv', sep='\t', index=False)
    df = pd.read_csv('subsampling_qna_enriched_with_theme_and_toxicity.tsv', sep='\t')
    print(df['toxicity'].describe())
    get_toxicity_plot(df)
    #counter = get_most_mentioned_themes_series(df)
    
    #counter = get_most_mentioned_locations_series(df)
    #counter = get_most_mentioned_people_series(df)
    #counter = get_most_mentioned_organizations_series(df)
    # render a bar chart for the most mentioned locations with matplotlib
    # render labels on the x-axis
    #plt.figure(figsize=(12, 10))
    #plt.bar(counter.keys(), counter.values())
    #plt.xticks(rotation=90)
    # make more room for the x labels (height)
    #plt.tight_layout()
    # add value on the top of the bars
    #for i, v in enumerate(counter.values()):
    #    plt.text(i, v, str(v), ha='center', va='bottom')
    #plt.show()
    

