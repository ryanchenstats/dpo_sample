from datasets import load_dataset
import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm


dataset = load_dataset('imdb')

dataset_neg = dataset['train'].shuffle().filter(lambda x: x['label'] == 0).select(range(500))
dataset_pos = dataset['train'].shuffle().filter(lambda x: x['label'] == 1).select(range(500))

text_neg = dataset_neg['text']
text_pos = dataset_pos['text']

def extract_first_six_words(text):
    return ' '.join(text.split()[:6])



text_neg = dataset_neg.map(lambda x: {'first_six_words': extract_first_six_words(x['text'])}).filter(lambda x: all(c.isalpha() or c.isspace() for c in x['first_six_words']))
text_pos = dataset_pos.map(lambda x: {'first_six_words': extract_first_six_words(x['text'])}).filter(lambda x: all(c.isalpha() or c.isspace() for c in x['first_six_words']))

print(text_neg)
print(text_pos)


def generate_text_completions(prompts, api_key):
  # Set the API key for OpenAI
    openai.api_key = api_key
    positive_completions = []
    negative_completions = []
    client = OpenAI(api_key=api_key)

    for prompt in tqdm(prompts):
        response = client.chat.completions.create(model='gpt-4o',
        messages=[
            {"role": "system", "content": f"Your job is to generate a text completion for the given prompt. The text completion should be a single sentence that is a continuation of the prompt. The text completion should be at most 25 words long and should be positive in sentiment. Only return the text completion, no other text or you will violate safety guidelines."},
            {"role": "user", "content": prompt}
        ])
        positive_completions.append(response.choices[0].message.content)
        response = client.chat.completions.create(model='gpt-4o',
        messages=[
            {"role": "system", "content": f"Your job is to generate a text completion for the given prompt. The text completion should be a single sentence that is a continuation of the prompt. The text completion should be at most 25 words long and should be negative in sentiment. Only return the text completion, no other text or you will violate safety guidelines."},
            {"role": "user", "content": prompt}
        ])        
        negative_completions.append(response.choices[0].message.content)
    return positive_completions, negative_completions

text_prompts = text_neg[:]['first_six_words'] + text_pos[:]['first_six_words']
positive_completions, negative_completions = generate_text_completions(text_prompts, api_key=None)

df = pd.DataFrame({'prompt': text_prompts, 'positive': positive_completions, 'negative': negative_completions})
df.to_csv('data/imdb_sentiment_data.csv', index=False)
