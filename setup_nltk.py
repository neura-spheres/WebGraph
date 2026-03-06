"""
Download required NLTK data packages.
Run once before first use: python setup_nltk.py
"""
import nltk

packages = [
    # Tokenization
    "punkt",
    "punkt_tab",
    # Stop words
    "stopwords",
    # Lemmatization
    "wordnet",
    "omw-1.4",
    # POS tagging (required for POS-aware lemmatization)
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    # Named Entity Recognition
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]

for pkg in packages:
    try:
        nltk.download(pkg, quiet=False)
        print(f"  [OK] {pkg}")
    except Exception as e:
        print(f"  [FAIL] {pkg}: {e}")

print("\nNLTK setup complete.")
