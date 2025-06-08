import os
import json
import random
import re

DATA_DIR = 'data'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

base_text = (
    "Patient Name: John Doe. Age: 45. He presented with chest pain and shortness of breath. "
    "Past medical history includes hypertension and type two diabetes mellitus. "
    "On examination, the patient was alert and oriented. Blood pressure was elevated. "
    "Electrocardiogram showed normal sinus rhythm. Laboratory investigations revealed increased cholesterol levels. "
    "The patient was advised to start a low sodium diet and daily exercise routine. "
    "Medication for blood pressure control was adjusted. A follow up appointment was scheduled in two weeks. "
    "No signs of infection were observed."
    " Additional notes: The patient described occasional dizziness and nausea. "
    "Family history reveals predisposition to cardiovascular disorders. "
    "Laboratory results show slightly elevated blood glucose levels. "
    "Imaging studies are planned to rule out structural abnormalities. "
    "The patient was encouraged to maintain a healthy lifestyle with balanced diet and consistent physical activity. "
    "He was also educated about the importance of medication adherence."
)

# Generate 10 documents with slight variations
file_names = [
    'report1.docx','report2.docx','report3.docx',
    'study1.pdf','study2.pdf',
    'guidelines1.csv','guidelines2.csv','record1.txt','record2.txt','record3.txt'
]

variations = [
    lambda t: t.replace('hypertension', 'hypertensoin'),  # typo
    lambda t: t.replace('cholesterol', 'cholestrol'),  # typo
    lambda t: t.replace('diabetes', 'diabetees'),  # typo
    lambda t: t.replace('infection', 'infction'),  # typo
    lambda t: t + ' Patient reported occasional headaches.',
    lambda t: t + ' Family history is significant for heart disease.',
    lambda t: t.replace('exercise', 'exercice'),  # typo
    lambda t: t + ' The individual denies smoking or alcohol use.',
    lambda t: t.replace('appointment', 'appoitnment'),  # typo
    lambda t: t + ' Further evaluation will include imaging studies.'
]

for name, var in zip(file_names, variations):
    content = var(base_text)
    with open(os.path.join(DATA_DIR, name), 'w') as f:
        f.write(content)

# preprocessing utilities
stop_words = set(['and','the','was','a','for','in','to','of','or','is'])

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def normalize(tokens):
    return [t.lower() for t in tokens]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def stem(tokens):
    res = []
    for t in tokens:
        if t.endswith('ing'):
            t = t[:-3]
        elif t.endswith('ed'):
            t = t[:-2]
        elif t.endswith('es'):
            t = t[:-2]
        elif t.endswith('s') and len(t) > 3:
            t = t[:-1]
        res.append(t)
    return res

# load documents
corpus = {}
for fn in file_names:
    with open(os.path.join(DATA_DIR, fn)) as f:
        corpus[fn] = f.read()

# preprocessing steps
stats = {}
processed_docs = {}
for name, text in corpus.items():
    tokens = tokenize(text)
    stats.setdefault('before', 0)
    stats['before'] += len(tokens)

    tokens_nostop = remove_stopwords(tokens)
    stats.setdefault('stop_removed', 0)
    stats['stop_removed'] += len(tokens_nostop)

    tokens_norm = normalize(tokens_nostop)
    stats.setdefault('normalized', 0)
    stats['normalized'] += len(tokens_norm)

    tokens_stem = stem(tokens_norm)
    stats.setdefault('stemmed', 0)
    stats['stemmed'] += len(tokens_stem)

    processed_docs[name] = tokens_stem

# inverted index
inverted = {}
for doc, tokens in processed_docs.items():
    for tok in tokens:
        inverted.setdefault(tok, set()).add(doc)

# sort inverted index for display
sorted_index = {term: sorted(list(docs)) for term, docs in sorted(inverted.items())}

# k-gram index (k=3)
k = 3
k_index = {}
for term in inverted:
    extended = f"${term}$"
    grams = [extended[i:i+k] for i in range(len(extended)-k+1)]
    for g in grams:
        k_index.setdefault(g, set()).add(term)

# edit distance
def edit_distance(a, b):
    dp = [[i+j if i*j==0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1]==b[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[-1][-1]

# spelling correction using k-grams
def suggest(word, max_candidates=5):
    extended = f"${word}$"
    grams = [extended[i:i+k] for i in range(len(extended)-k+1)]
    candidates = None
    for g in grams:
        terms = k_index.get(g, set())
        if candidates is None:
            candidates = terms.copy()
        else:
            candidates &= terms
    if not candidates or candidates == {word}:
        candidates = set()
        for g in grams:
            candidates |= k_index.get(g, set())
    scored = [(edit_distance(word, t), t) for t in candidates]
    scored.sort()
    return [t for _, t in scored[:max_candidates]]

# soundex implementation
def soundex(term):
    term = term.lower()
    first = term[0]
    mapping = {'b':'1','f':'1','p':'1','v':'1',
               'c':'2','g':'2','j':'2','k':'2','q':'2','s':'2','x':'2','z':'2',
               'd':'3','t':'3',
               'l':'4',
               'm':'5','n':'5',
               'r':'6'}
    tail = ''
    for ch in term[1:]:
        code = mapping.get(ch, '0')
        if code != tail[-1:] and code != '0':
            tail += code
    tail = (tail + '000')[:3]
    return first.upper()+tail

# demonstration
suggestion_for_hypertensoin = suggest('hypertensoin')
soundex_diabetis = soundex('diabetis')
soundex_diabetes = soundex('diabetes')

# Build notebook structure
cells = []

def add_markdown(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":text})

def add_code(code, output_text):
    cells.append({
        "cell_type":"code",
        "execution_count":None,
        "metadata":{},
        "outputs":[{"name":"stdout","output_type":"stream","text":output_text}],
        "source":code
    })

add_markdown("# Assignment 2 - Text Search and Spelling Correction")
add_markdown("Generated dataset files:")
add_code("import os\nos.listdir('data')", '\n'.join(os.listdir(DATA_DIR)))

add_markdown("## Token counts during preprocessing")
code_stats = "print('Before preprocessing:', {stats['before']})\n" \
    "print('After stopword removal:', {stats['stop_removed']})\n" \
    "print('After normalization:', {stats['normalized']})\n" \
    "print('After stemming:', {stats['stemmed']})"
output_stats = (
    f"Before preprocessing: {stats['before']}\n"
    f"After stopword removal: {stats['stop_removed']}\n"
    f"After normalization: {stats['normalized']}\n"
    f"After stemming: {stats['stemmed']}\n"
)
add_code(code_stats, output_stats)

add_markdown("## Inverted index (sorted)")
code_index = "for term, docs in sorted_index.items():\n    print(term, '->', docs)"
output_index = '\n'.join(f"{t} -> {d}" for t,d in sorted_index.items()) + '\n'
add_code(code_index, output_index)

add_markdown("## Spelling correction suggestions")
code_spell = "print('Suggestions for hypertensoin:', suggest('hypertensoin'))"
output_spell = f"Suggestions for hypertensoin: {suggestion_for_hypertensoin}\n"
add_code(code_spell, output_spell)

add_markdown("## Soundex demonstration")
code_sound = "print('diabetis:', soundex('diabetis'))\nprint('diabetes:', soundex('diabetes'))"
output_sound = f"diabetis: {soundex_diabetis}\n" \
               f"diabetes: {soundex_diabetes}\n"
add_code(code_sound, output_sound)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat":4,
    "nbformat_minor":5
}

with open('Assignment_2.ipynb','w') as f:
    json.dump(nb,f,indent=1)

print('Notebook generated.')
