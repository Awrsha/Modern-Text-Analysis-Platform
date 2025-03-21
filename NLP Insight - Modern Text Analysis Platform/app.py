from flask import Flask, render_template, request, jsonify, send_from_directory
import spacy
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
import re
import string
import os
import logging
import textstat

app = Flask(__name__, static_folder="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_trf")
sentiment_analyzer = SentimentIntensityAnalyzer()
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

topic_model = BERTopic(language="english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def get_word_frequencies(text):
    words = [token.lemma_.lower() for token in nlp(text) if token.is_alpha and token.text.lower() not in stop_words]
    return Counter(words)

def get_ngrams(text, n=2):
    tokens = [token.lemma_.lower() for token in nlp(text) if token.is_alpha and token.text.lower() not in stop_words]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    wordcloud_base64 = base64.b64encode(img.getvalue()).decode()
    return wordcloud_base64

def calculate_readability(text):
    grade_level = textstat.flesch_kincaid_grade(text)
    
    if grade_level < 6:
        label = "Elementary"
    elif grade_level < 9:
        label = "Middle School"
    elif grade_level < 13:
        label = "High School"
    elif grade_level < 17:
        label = "College"
    else:
        label = "Graduate Level"
    
    score = min(1.0, max(0.0, grade_level / 20.0))
    return label, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    
    cleaned_text = clean_text(text)
    
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if sentiment_score["compound"] > 0 else "Negative" if sentiment_score["compound"] < 0 else "Neutral"
    sentiment_confidence = abs(sentiment_score["compound"])

    emotion_result = emotion_analyzer(text)
    emotion = emotion_result[0]["label"]
    emotion_confidence = emotion_result[0]["score"]

    doc = nlp(text)
    entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

    topics, probs = topic_model.fit_transform([text])
    topic_info = topic_model.get_topic_info()
    topic_data = []
    
    if not topic_info.empty:
        for index, row in topic_info.head(8).iterrows():
            if row['Topic'] != -1:
                topic_words = topic_model.get_topic(row['Topic'])
                label = " ".join([word[0] for word in topic_words[:3]])
                topic_data.append({"label": label, "score": float(row['Count'] / topic_info['Count'].sum())})

    word_freq = get_word_frequencies(text)
    bigrams = get_ngrams(text, 2)
    trigrams = get_ngrams(text, 3)
    
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    except:
        summary = "Text too short or unsuitable for summarization."
    
    complexity = min(1.0, len(set(word for word in text.lower().split())) / 500.0)
    readability_label, readability_score = calculate_readability(text)
    
    word_freq_data = word_freq.most_common(10)
    
    return jsonify({
        "sentiment": sentiment,
        "sentimentScore": sentiment_confidence,
        "emotion": emotion,
        "emotionScore": emotion_confidence,
        "summary": summary,
        "entities": entities,
        "topics": topic_data,
        "wordFreq": word_freq_data,
        "bigrams": bigrams[:5],
        "trigrams": trigrams[:5],
        "complexity": complexity,
        "readability": readability_label,
        "readabilityScore": readability_score
    })

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")

    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)

    sentiment_score1 = sentiment_analyzer.polarity_scores(text1)
    sentiment_score2 = sentiment_analyzer.polarity_scores(text2)

    sentiment1 = "Positive" if sentiment_score1["compound"] > 0 else "Negative" if sentiment_score1["compound"] < 0 else "Neutral"
    sentiment2 = "Positive" if sentiment_score2["compound"] > 0 else "Negative" if sentiment_score2["compound"] < 0 else "Neutral"

    sentiment_confidence1 = abs(sentiment_score1["compound"])
    sentiment_confidence2 = abs(sentiment_score2["compound"])

    emotion_result1 = emotion_analyzer(text1)
    emotion_result2 = emotion_analyzer(text2)

    emotion1 = emotion_result1[0]["label"]
    emotion2 = emotion_result2[0]["label"]

    emotion_confidence1 = emotion_result1[0]["score"]
    emotion_confidence2 = emotion_result2[0]["score"]

    doc1 = nlp(text1)
    doc2 = nlp(text2)

    entities1 = [{"entity": ent.text, "label": ent.label_} for ent in doc1.ents]
    entities2 = [{"entity": ent.text, "label": ent.label_} for ent in doc2.ents]

    word_freq1 = get_word_frequencies(text1)
    word_freq2 = get_word_frequencies(text2)

    complexity1 = min(1.0, len(set(word for word in text1.lower().split())) / 500.0)
    complexity2 = min(1.0, len(set(word for word in text2.lower().split())) / 500.0)
    
    def calc_positivity(score):
        return (score["compound"] + 1) / 2
    
    def calc_emotionality(text):
        emotional_words = ["love", "hate", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        words = text.lower().split()
        return min(1.0, sum(1 for word in words if word in emotional_words) / max(1, len(words) * 0.1))
    
    def calc_formality(text):
        formal_indicators = ["therefore", "thus", "consequently", "furthermore", "moreover", "hence", "accordingly"]
        informal_indicators = ["like", "you know", "kind of", "sort of", "pretty", "really", "actually"]
        
        words = text.lower().split()
        formal_count = sum(1 for word in words if word in formal_indicators)
        informal_count = sum(1 for word in words if word in informal_indicators)
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5
        return min(1.0, formal_count / total)
    
    def calc_specificity(doc):
        specific_pos = ["NOUN", "PROPN", "NUM"]
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0.5
        return min(1.0, sum(1 for token in doc if token.pos_ in specific_pos) / total_tokens)
    
    def calc_subjectivity(text):
        subjective_words = ["I", "me", "my", "mine", "we", "us", "our", "ours", "think", "believe", "feel", "opinion", "perspective"]
        words = text.lower().split()
        return min(1.0, sum(1 for word in words if word in subjective_words) / max(1, len(words) * 0.1))

    metrics1 = [
        complexity1,
        calc_positivity(sentiment_score1),
        calc_emotionality(text1),
        calc_formality(text1),
        calc_specificity(doc1),
        calc_subjectivity(text1)
    ]
    
    metrics2 = [
        complexity2,
        calc_positivity(sentiment_score2),
        calc_emotionality(text2),
        calc_formality(text2),
        calc_specificity(doc2),
        calc_subjectivity(text2)
    ]

    return jsonify({
        "text1": {
            "sentiment": sentiment1,
            "sentimentScore": sentiment_confidence1,
            "emotion": emotion1,
            "emotionScore": emotion_confidence1,
            "entities": entities1,
            "wordFreq": word_freq1.most_common(10),
            "metrics": metrics1
        },
        "text2": {
            "sentiment": sentiment2,
            "sentimentScore": sentiment_confidence2,
            "emotion": emotion2,
            "emotionScore": emotion_confidence2,
            "entities": entities2,
            "wordFreq": word_freq2.most_common(10),
            "metrics": metrics2
        }
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    env = os.environ.get('FLASK_ENV', 'development')
    debug_mode = env != 'production'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)