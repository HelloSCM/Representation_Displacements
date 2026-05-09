import nltk
from nltk.corpus import wordnet as wn


try:
    wn.ensure_loaded()
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def get_vision_parent(wnid: str) -> str:
    try:
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset("n", offset)
        hypernyms = synset.hypernyms()
        return hypernyms[0].name() if hypernyms else "root"
    except Exception:
        return "unknown_parent"


def get_language_parent(word: str) -> str:
    try:
        formatted = word.lower().replace(" ", "_")
        synsets = wn.synsets(formatted, pos=wn.NOUN)
        if not synsets:
            return "unknown_parent"
        hypernyms = synsets[0].hypernyms()
        return hypernyms[0].name() if hypernyms else "root"
    except Exception:
        return "unknown_parent"
