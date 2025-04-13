import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(paper_text):
    inputs = tokenizer(paper_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)

    return features

def classify_paper(paper_path):
    with open(paper_path, 'r', encoding='utf-8') as file:
        paper_text = file.read()
    
    # Extract features
    features = extract_features(paper_text)
    torch.save(features, 'paper_features.pt')

if __name__ == '__main__':
    paper_path = 'uploaded_paper.txt'  # Replace with the path to your uploaded file
    classify_paper(paper_path)
