from transformers import BertForSequenceClassification, BertTokenizer
import dash
import torch
import os
from dash import dcc, html, Input, Output, State
from dotenv import load_dotenv
load_dotenv() 

if os.environ.get("environment") == "development":
    model = BertForSequenceClassification.from_pretrained("bert-saved", torch_dtype=torch.float32)
    tokenizer = BertTokenizer.from_pretrained("bert-saved")
else:
    model = BertForSequenceClassification.from_pretrained("username/imdb-bert-sentiment", torch_dtype=torch.float32)
    tokenizer = BertTokenizer.from_pretrained("username/imdb-bert-sentiment")   


model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    return prediction, probs.numpy().tolist()[0]

# Dash app
app = dash.Dash(__name__)
server = app.server  

app.layout = html.Div([
    html.H1("IMDB Sentiment Classifier (BERT)"),
    dcc.Textarea(
        id='input-text',
        placeholder='Enter a movie review...',
        style={'width': '100%', 'height': 200},
    ),
    html.Button('Classify', id='submit-button', n_clicks=0),
    html.Div(id="output-container") 
])

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, value):
    if not value:
        return "Please enter a review first."
    label, probs = predict_sentiment(value)
    sentiment = "Positive" if label == 1 else "Negative"
    return f"Prediction: {sentiment} (Probabilities: {probs})"

if os.environ.get("environment") == "development": #Environment Var
    # On Local 
    if __name__ == '__main__':
        app.run(debug=True)
else:
    if __name__ == '__main__':
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port, debug=False)