from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import pandas as pd

import nltk,spacy,re,requests,time
import numpy as np
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

import torch
import torch.nn as nn

# 68 classes are trained for:




url_dict = {1.1:'https://www.primevideo.com/movie',
            5.1: 'https://www.google.com/',
            1.2: 'https://primocapital.ae/property/apartment?page=4',
            2: 'https://landing.gosecure.ai/W-eBook-Fight-the-Phishing-Epidemic-Landing.html',
            5: 'https://hesh.am/about',
            7: 'https://oraimo.ae/speakers/wiwu-p50/',
            10: 'https://www.roids.biz/topic/buying-anabolic-steroids/page/2',
            12: 'https://divihomes.ai/personable-filtration/all/all',
            26: 'https://jobs.mitula.ae/staff-professional-saudi-arabia-jobs-jobs-jobs',
            46: 'https://nccca.com.au/residential-care/residential-faq/',
            # 54: 'https://fullscatmovies.club/diosa-susi-smilling-hearts-shi/',
            68: 'https://www.abortionclinics.com/tag/arizona-abortion/',
            79: 'https://www.scentube.ae/products/Men-Perfumes/Abercrombie-And-Fitch-First-Instinct-Eau-De-Toilette-100ml-For-Men',
            81: 'https://shop836.netlify.app/white-tall-ceramic-tree-trunk-vase-d24170018b2c9caaf5eae00e8a2143d3.html'
        }

examples = [[val] for key,val in url_dict.items()]

categories = {
    1: "Real Estate",
    2: "Computer and Internet Security",
    3: "Financial Services",
    4: "Business and Economy",
    5: "Computer and Internet Info",
    6: "Auctions",
    7: "Shopping",
    8: "Cult and Occult",
    9: "Travel",
    10: "Abused Drugs",
    11: "Adult",
    12: "Home and Garden",
    13: "Military",
    14: "Social Networking",
    15: "Dead",
    16: "Stock Advice and Tools",
    17: "Training and Tools",
    18: "Dating",
    19: "Sex Education",
    20: "Religion",
    21: "Entertainment and Arts",
    22: "Personal Sites and Blogs",
    23: "Legal",
    24: "Local Information",
    25: "Streaming Media",
    26: "Job Search",
    27: "Gambling",
    28: "Translation",
    29: "Reference and Research",
    30: "Shareware and Freeware",
    31: "Peer to Peer",
    32: "Marijuana",
    33: "Hacking",
    34: "Games",
    35: "Philosophy and Political Advocacy",
    36: "Weapons",
    37: "Pay to Surf",
    38: "Hunting and Fishing",
    39: "Society",
    40: "Educational Institutions",
    41: "Online Greeting Cards",
    42: "Sports",
    43: "Swimsuits and Intimate Apparel",
    44: "Questionable",
    45: "Kids",
    46: "Abortion Pro Choice (deprecated)",
    47: "Online Personal Storage",
    48: "Abortion Pro Life (deprecated)",
    49: "Keyloggers and Monitoring",
    50: "Search Engines",
    51: "Internet Portals",
    52: "Web Advertisements",
    53: "IM Client (deprecated)",
    54: "Internet Telephony Client (deprecated)",
    55: "Web-based Email",
    56: "Malware",
    57: "Phishing and other Frauds",
    58: "Proxy Avoidance and Anonymizers",
    59: "Spyware and Adware",
    60: "Music",
    61: "Government",
    62: "Nudity",
    63: "News",
    64: "Computer News (deprecated)",
    65: "Content Delivery Networks",
    66: "Internet Communications",
    67: "Bot Nets",
    68: "Abortion",
    69: "Health and Medicine",
    70: "Confirmed SPAM sources",
    71: "SPAM URLs",
    72: "Unconfirmed SPAM sources",
    73: "Open HTTP Proxies",
    74: "Dynamically Generated Content",
    75: "Parked",
    76: "Alcohol and Tobacco",
    77: "Private IP Addresses",
    78: "Image and Video Search",
    79: "Fashion and Beauty",
    80: "Recreation and Hobbies",
    81: "Motor Vehicles",
    82: "Web Hosting",
    85: "Self Harm",
    86: "DNS Over HTTPS",
    87: "Low-THC Cannabis Products",
    88: "Generative AI"
}

# Label Encoded vs Actual GT:
map_label_enc_act_label = {0: 0,
                                1: 1,
                                2: 2,
                                3: 3,
                                4: 4,
                                5: 5,
                                6: 6,
                                7: 7,
                                8: 8,
                                9: 9,
                                10: 10,
                                11: 11,
                                12: 12,
                                13: 14,
                                14: 15,
                                15: 16,
                                16: 17,
                                17: 18,
                                18: 19,
                                19: 20,
                                20: 21,
                                21: 22,
                                22: 24,
                                23: 25,
                                24: 26,
                                25: 27,
                                26: 28,
                                27: 30,
                                28: 31,
                                29: 32,
                                30: 33,
                                31: 34,
                                32: 35,
                                33: 36,
                                34: 38,
                                35: 40,
                                36: 41,
                                37: 42,
                                38: 44,
                                39: 45,
                                40: 46,
                                41: 48,
                                42: 49,
                                43: 50,
                                44: 51,
                                45: 52,
                                46: 54,
                                47: 55,
                                48: 56,
                                49: 57,
                                50: 58,
                                51: 59,
                                52: 62,
                                53: 63,
                                54: 64,
                                55: 66,
                                56: 68,
                                57: 69,
                                58: 71,
                                59: 75,
                                60: 76,
                                61: 79,
                                62: 80,
                                63: 81,
                                64: 82,
                                65: 86,
                                66: 87,
                                67: 88}


# Load the SpaCy model
nlp = spacy.load("en_core_web_lg")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

entity_columns = ['PERSON_present', 'NORP_present', 'FAC_present', 'ORG_present',
       'GPE_present', 'LOC_present', 'PRODUCT_present', 'EVENT_present',
       'WORK_OF_ART_present', 'LAW_present', 'LANGUAGE_present',
       'DATE_present', 'TIME_present', 'PERCENT_present', 'MONEY_present',
       'QUANTITY_present', 'ORDINAL_present', 'CARDINAL_present']

print(f"Entity Columns length is : {len(entity_columns)}")


def clean_text(text):
    # Clean the text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)  # Keeps only alphanumeric characters, spaces, periods, and commas
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Removes extra spaces
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text) # Remove punctuation
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    normalized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    normalized_text = ' '.join(normalized_tokens)
    print(f"Cleaned text")
    return normalized_text


def extract_named_entities(normalized_text):
    # Extract named entities
    doc = nlp(normalized_text)
    # entities = {ent_type: set() for ent_type in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]}
    entities = {'PERSON': set(), 'NORP': set(), 'FAC': set(), 'ORG': set(), 'GPE': set(), 'LOC': set(), 'PRODUCT': set(), 'EVENT': set(), 'WORK_OF_ART': set(), 'LAW': set(), 'LANGUAGE': set(), 'DATE': set(), 'TIME': set(), 'PERCENT': set(), 'MONEY': set(), 'QUANTITY': set(), 'ORDINAL': set(), 'CARDINAL': set()}
    for ent in doc.ents:
        entities[ent.label_].add(ent.text)
    
    # Convert sets to comma-separated strings
    # for key in entities:
        # entities[key] = ", ".join(sorted(entities[key]))  # Sorting for consistency
    return entities


## Get Title and text from the given URL:
def read_text_and_title(url, timeout=6):
    num_words = 2000
    try:
        # Add timeout parameter to the requests.get call
        with requests.get(url, verify=False, timeout=timeout) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove all script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text(separator=' ')
            text = ' '.join(text.split())
            words = text.split()[:num_words]

            if soup.title is None:
                title = None
            else:
                # Extract title
                title = soup.title.text

            print(f"### Length of words: {len(words)} and title: {title}")

    except requests.exceptions.Timeout:
        print("Request timed out")
        title = None
        text = None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        title = None
        text = None

    return title, text

# Basic model-1:
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Model-2 : Transformer model:
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)  # Project to hidden size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.input_proj(x)  # Shape: [batch_size, hidden_size]
        
        # To use TransformerEncoder, we need [seq_len, batch_size, hidden_size]
        # Add sequence dimension (assuming seq_len=1 for each vector)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, hidden_size]
        
        x = self.transformer(x)  # x shape: [1, batch_size, hidden_size]
        
        # Use the output of the single sequence token
        x = x.squeeze(0)  # Shape: [batch_size, hidden_size]
        x = self.fc(x)    # Shape: [batch_size, num_classes]
        
        return x



def preprocess_text(title, text):
    print(f"### Processing Title and Text")
    # Normalize text and title:
    title_norm = clean_text(title)
    text_norm = clean_text(text)
    print(f"### Cleaned Title and Text")
    # Extract Named entities from Normalized text :
    text_ne = extract_named_entities(text_norm)
    print(f"### Text named entities : {text_ne}")

    # Generate embeddings for title and text
    title_embedding = sentence_model.encode(title_norm, convert_to_tensor=True, device='cpu')
    text_embedding = sentence_model.encode(text_norm, convert_to_tensor=True, device='cpu')
    
    # Concatenate embeddings
    combined_embedding = torch.cat((title_embedding, text_embedding), dim=0)
    
    # Convert to a numpy array
    combined_embedding_np = combined_embedding.numpy()
    
    # Add dummy values for named entities if necessary (make sure to match the feature size)
    # Assuming named entity columns are zeros for simplicity
    entity_features = np.zeros((1, len(entity_columns)))
    for key,val in text_ne.items():
        if val and key+"_present" in entity_columns:
            index = entity_columns.index(key + "_present")
            entity_features[0, index] = 1

    print(entity_features)
    # print(combined_embedding_np)
    # Combine embeddings with entity features
    features = np.concatenate((combined_embedding_np, entity_features[0]), axis=0)
    print(f"Features shape is : {features.shape}")
    return torch.tensor(features, dtype=torch.float32)

def predict(title, text):
    inputs = preprocess_text(title, text)
    # print(f"inputs shape is : {inputs.shape}")
    with torch.no_grad():
        outputs = model(inputs)
        # print(f"outputs shape is : {len(outputs)}, {outputs}")
        _, predicted_class = torch.max(outputs, 0)  # Use dim=0 since `outputs` is now a 1D tensor
        # Get top 5 predictions and their indices
        topk_values, topk_indices = torch.topk(outputs, 5)

        # Print the top 5 prediction scores and their corresponding class indices
        print("Top 5 Prediction Scores:", topk_values)
        print("Top 5 Class Indices:", topk_indices)
        
        classified_into= []
        for i in topk_indices:
           i = int(i)
           classified_into.append(categories[map_label_enc_act_label[i]])
        
        # return pd.DataFrame(classified_into, columns=["Predictions"])
        #    print(f"{i} : {categories[map_label_enc_act_label[i]]}" )
        
        print(f"map_label_enc_act_label Key is : {predicted_class.item()}")
        return categories[map_label_enc_act_label[predicted_class.item()]]


# # Initialize the model
# input_dim = 768+18                          #X_train.shape[1]  # Number of features (embedding dimensions + entity features)
# output_dim = 12  #len(df['cat1'].unique())  # Number of classes
# print(f"Input dim : {input_dim}")
# model = Classifier(input_dim = input_dim, output_dim = output_dim)
# model.load_state_dict(torch.load('/Users/v.kanukollu/MyStuff/Personal/Projects/Technovate-Ideas/2024/02-URL-Classification/code/12_model.pth'))
# model.eval()  # Set the model to evaluation mode


# Instantiate the model
trained_model_file = "/Users/v.kanukollu/MyStuff/Personal/Projects/Technovate-Ideas/2024/02-URL-Classification/68_transfrmr_model_acc75_dropout_0.3.pth"
input_size = 786
num_classes = 68
hidden_size = 768  # Must be divisible by nhead
nhead = 6
num_layers = 6
model = TransformerModel(input_size, num_classes, hidden_size, nhead, num_layers)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model = Classifier(input_dim = input_dim, output_dim = output_dim)
model.load_state_dict(torch.load(trained_model_file))
model.eval()  # Set the model to evaluation mode

def classify_url(url):
    print(f"### Entered URL is : {url}")
    title, text = read_text_and_title(url=url)

    if title is None and text is None:
        return "Request timed out"
    elif title is None:
        return f"This domain is temporarily unavailable : {url}"
    else:
        print(f"### Got Text and Title from the URL : {url} : {title}")
        preprocess_text(title,text)
        # print(f"### Processed text and title")

        start_time = time.time()
        predicted_class = predict(title, text)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Predicted time : {execution_time} seconds")
        
        print(f"### Predicted class: {predicted_class}")
        return predicted_class



@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        url = data.get('url')
    
        # Call classify_url function with the input URL
        classification = classify_url(url)
        print(f"### Classification is : {classification}")
        # Return the result as JSON
        return jsonify({'classification': classification})
    except Exception as e:
        return jsonify({"error": str(e)})

    # Here you would typically use your deep learning model to classify the URL
    # For this example, we'll use a simple classification based on the URL content
    # if 'primevideo' in url:
    #     classification = 'Entertainment'
    # elif 'primocapital' in url:
    #     classification = 'Real Estate'
    # elif 'jobs.mitula' in url:
    #     classification = 'Job Listing'
    # else:
    #     classification = 'Unknown'
    

if __name__ == '__main__':
    app.run(debug=True)
