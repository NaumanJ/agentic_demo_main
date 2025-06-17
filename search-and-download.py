import os
import requests
from bs4 import BeautifulSoup
import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def extract_keywords(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Identify nouns
    tagged_words = pos_tag(filtered_words)
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]

    return nouns

def search_wikipedia(keywords):
    search_results = []
    for keyword in keywords:
        try:
            page = wikipedia.page(keyword)
            search_results.append({
                'title': page.title,
                'url': page.url,
                'content': page.content
            })
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for keyword '{keyword}': {e}")
        except wikipedia.exceptions.PageError as e:
            print(f"Page error for keyword '{keyword}': {e}")
    return search_results

def download_html(url, directory):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    filename = os.path.join(directory, f"{url.split('/')[-1]}.html")
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(str(soup))
    return filename

def main(text):
    # Step 1: Extract keywords from the provided text
    keywords = extract_keywords(text)

    # Step 2: Search Wikipedia with the keywords
    search_results = search_wikipedia(keywords)

    # Step 3: Download the documents to the ./html directory
    html_directory = './html'
    os.makedirs(html_directory, exist_ok=True)

    for result in search_results:
        filename = download_html(result['url'], html_directory)
        print(f"Downloaded {result['title']} to {filename}")

if __name__ == "__main__":
    input_text = "Lionel Messi football player"
    main(input_text)
