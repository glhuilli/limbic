
# Limbic Basic Example

This notebook contains a few basic examples on how to use the `limbic` package.

Currently, the only emotion model supported in `limbic` is lexicon-based. More advanced models (e.g. using machine learning trained models) will be included in future releases of this package.


## Importing a lexicon-based emotion classifier

The only thing you need to create a new lexicon-based emotion classifier is of course the lexicon. However, in case you are dealing with a specific context, it's possible to use a terms mapping dictionary which will automatically replace terms on the input you want to process.

The lexicon has to be loaded by the user and it could be either a custom lexicon or [lexicons from the NRC](http://saifmohammad.com/WebPages/AccessResource.htm). To load a lexicon you can either use a generic `load_lexicon` or `load_nrc_lexicon` tailored for some NRC lexicons.

To use the generic `load_lexicon` method you can do the following:

```python
from limbic.emotion.utils import load_lexicon

my_lexicon_file_path = '../data/my_lexicon.csv'
lexicon = load_lexicon(my_lexicon_file_path)
```

where the hypothetical file `../data/lexicon.csv` is a `csv` file with the header `term,emotion,score`.

To use the `load_nrc_lexicon` method you need to download one of the supported NRC files, and do

```python
from limbic.emotion.nrc_utils import load_nrc_lexicon

nrc_lexcon_file_path = '../data/lexicons/NRC-AffectIntensity-Lexicon.txt'
lexicon = load_nrc_lexicon(nrc_lexicon_file_path, 'affect_intensity')
```

The supported files are the `affect_intensity` lexicon, the `emotion` lexicon (aka `EmoLex`), and the `vad` lexicon.

Finally, it's important to note that the terms mapping dictionary has to be of type `Dict[str, str]`, where a given term or collection of terms will be mapped to another term of collection of terms.


### Important note

In case you are using NRC lexicons, you need to know that there are some constraints about using them for profit. Please refer to the NRC website for more information on how to notify and work with their data. Otherwise, you are free to use limbic however you want under the MIT license.



```python
from limbic.emotion.models import LexiconLimbicModel
from limbic.emotion.nrc_utils import load_nrc_lexicon

lexicon = load_nrc_lexicon('../data/lexicons/NRC-AffectIntensity-Lexicon.txt', 'affect_intensity')
lb = LexiconLimbicModel(lexicon)

```

### Emotions from Terms


Once the `limbic` model is loaded, you can either get the emotions that a either a single term or a full sentence has. For example, you can get the emotions associated to the word `love` or `hate`. Alternatively, you can get te emotions associated to `not love` and `not hate`, which is passing a `is_negated` parameter to the method.

For each term, a list of `Emotion` named tuples will be returned. Each `Emotion` will have the fields `category` which indicates one of the motions that the term has been assigned, `value` that quantifies how strong the emotion category has been assigned to the term , and `term`. This term in case the method is called with `is_negated=True` will have a dash as a prefix, e.g. `term=love, is_negated=True` will generate an `Emotion` with `term=-love`.



```python
print('-'* 100)
print('Emotions for love, hate, not love, and not hate.')
print('-'* 100)
for term in ['love', 'hate']:
    print(f'{term} -> {lb.get_term_emotions(term)}')
for term in ['LOVE', 'Hate']:
    print(f'{term} (negated) -> {lb.get_term_emotions(term, is_negated=True)}')
```

    ----------------------------------------------------------------------------------------------------
    Emotions for love, hate, not love, and not hate.
    ----------------------------------------------------------------------------------------------------
    love -> [Emotion(category='joy', value=0.828, term='love')]
    hate -> [Emotion(category='anger', value=0.828, term='hate'), Emotion(category='fear', value=0.484, term='hate'), Emotion(category='sadness', value=0.656, term='hate')]
    LOVE (negated) -> [Emotion(category='sadness', value=0.828, term='-love')]
    Hate (negated) -> [Emotion(category='fear', value=0.828, term='-hate'), Emotion(category='anger', value=0.484, term='-hate'), Emotion(category='joy', value=0.656, term='-hate')]


### Negated terms

The categories supported for the `is_negated` parameter are the ones included in the [Plutchik's wheel of emotions](https://en.wikipedia.org/wiki/Contrasting_and_categorization_of_emotions), shown below (source: Wikipedia)

<img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/Plutchik-wheel.svg" alt="Drawing" style="width: 450px;"/>

Here, each emotion is placed in a wheel where the any emotion is facing its "opposite" in the opposite side of the wheel. For example, `joy` is placed to the opposite side of `sadness`, `rage` on the opposite side of `terror`, and so on. Whenever a term is negated, the opposite emotion will be used, as well as the `value` of the initial emotion. For example, `love` has an emotion of `joy` with score `0.828` (following the NRC `affect_intensity` lexicon). Then `love` negated will have an emotion of `sadness` with score `0.828`.

### Emotions for sentences

Like getting the emotions of a term, `limbic` has a method for getting the emotions for full or partial sentence. This is supported by the fact that each sentence has multiple terms, which some of them could have one or multiple emotions. Note that in some cases a sentence could have some negated terms that need to be considered. Some examples on how to process sentences and the expected output are presented below.


```python
sentence = 'I love and enjoy this string.'

lb.get_sentence_emotions(sentence)
```




    [Emotion(category='joy', value=0.828, term='love'),
     Emotion(category='joy', value=0.812, term='enjoy')]




```python
sentence = "I don't love but I enjoy this string."

lb.get_sentence_emotions(sentence)
```




    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy')]




```python
sentence = "I don't love but I enjoy this sentence."

lb.get_sentence_emotions(sentence)
```




    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy'),
     Emotion(category='anger', value=0.203, term='sentence'),
     Emotion(category='fear', value=0.266, term='sentence'),
     Emotion(category='sadness', value=0.234, term='sentence')]



### Emotions using the terms mapping

Note that in the last example `I don't love but I enjoy this sentence`, the word `sentence` could be placed under two different contexts: `sentence` as in a set for words or `sentence` as in punishment.

If you are under the context that `sentence` is just a collection of words, you can use the `terms_mapping` when defining the `limbic` object.


```python
terms_mapping = {'sentence': 'string'}
lb = LexiconLimbicModel(lexicon, terms_mapping=terms_mapping)

sentence = "I don't love but I enjoy this sentence."

lb.get_sentence_emotions(sentence)
```




    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy')]




```python
sentence = 'When the snows fall and the white winds blow, the lone wolf die but the pack survives'
lb.get_sentence_emotions(sentence)
```




    [Emotion(category='sadness', value=0.418, term='fall'),
     Emotion(category='joy', value=0.109, term='white'),
     Emotion(category='sadness', value=0.446, term='lone'),
     Emotion(category='fear', value=0.766, term='die'),
     Emotion(category='sadness', value=0.773, term='die')]


