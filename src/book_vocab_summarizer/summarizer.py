import re
import os
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pypdf import PdfReader  # comment this out if only using .txt

COMMON_WORDS_FILE = "20k.txt"  # path to frequency list file

EN_STOPWORDS = set(stopwords.words("english"))

def load_common_words(top_n: int | None = None) -> set[str]:
    common = []

    # directory of this file (summarizer.py)
    here = Path(__file__).parent
    # word_lists/google-10000-english.txt
    file_path = here / "word_lists" / COMMON_WORDS_FILE

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if not w:
                continue
            common.append(w)
            if top_n is not None and len(common) >= top_n:
                break
    return set(common)

COMMON_WORDS = load_common_words(top_n=5000)  # tune this

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def get_rare_words(
    text: str,
    min_len: int = 4,
    max_freq: int = 3,
    stopword_lang: str = "english",
    top_n: int = 100,
    include_common: bool = True
):
    print(f"Search for rare words, greater than {min_len} with less than {max_freq} occurrences.")
    print(f"Filtering common words.") if not include_common else ""

    # basic cleanup
    text = text.lower()
    # tokenize using nltk
    tokens = word_tokenize(text)

    # filter tokens:
    # - alphabetic
    # - at least min_len characters
    # - not in stopwords
    stops = EN_STOPWORDS  # could swap for other languages
    
    # Toggle inclusion of common words
    filtered_words = []
    if not include_common:
        filtered_words = COMMON_WORDS

    words = [
        w for w in tokens
        if w.isalpha()
        and len(w) >= min_len
        and w not in EN_STOPWORDS
        and w not in filtered_words
    ]

    counts = Counter(words)

    # keep only words that appear at most max_freq times
    rare_items = [(w, c) for w, c in counts.items() if c <= max_freq]

    # sort: first by frequency (ascending), then alphabetically
    rare_items.sort(key=lambda x: (x[1], x[0]))

    # limit to top_n
    return rare_items[:top_n]

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a list of least common words in a document."
    )
    parser.add_argument("input_file", type=str, help="Path to .txt or .pdf file")
    parser.add_argument(
        "--min-len", type=int, default=4, help="Minimum word length to keep"
    )
    parser.add_argument(
        "--max-freq", type=int, default=3,
        help="Maximum frequency for a word to count as 'rare'"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Maximum number of rare words to output"
    )
    parser.add_argument(
        "--exclude-common",
        action="store_true",
        help="Exclude most common English words from the results",
    )
    args = parser.parse_args()

    path = Path(args.input_file)
    text = extract_text(path)
    rare_words = get_rare_words(
        text,
        min_len=args.min_len,
        max_freq=args.max_freq,
        top_n=args.top_n,
        include_common=not args.exclude_common,
    )

    print(f"Found {len(rare_words)} rare words:")
    for word, freq in rare_words:
        print(f"{word}\t{freq}")

if __name__ == "__main__":
    main()
