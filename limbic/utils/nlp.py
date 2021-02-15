from typing import List

import spacy

nlp = spacy.load('en_core_web_sm')


def get_negated_words(sentence: str) -> List[str]:
    """
    Given a sentence, identify all negated words following different
    NLP rules. Based on Spacy's dependency trees and some hand.
    """
    doc = nlp(sentence)
    negation_tokens = [token for token in doc if token.dep_ == 'neg']
    attr_tokens = [token for token in doc if token.dep_ == 'attr']
    negation_head_tokens = [token.head for token in negation_tokens]
    negated_words = []
    for token in negation_head_tokens:
        if token.dep_ == 'advmod':
            negated_words.append(token.text)
            negated_words.append(token.head.text)
        if token.dep_ == 'ROOT' and token.head.pos_ == 'VERB':
            negated_words.append(token.text)
            for token_ in set(attr_tokens).intersection(set(token.children)):
                negated_words.append(token_)
    return [str(x) for x in negated_words]
