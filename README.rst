limbic
======

Lexicon-based ``Emotion`` analysis from text tool for series/movie
subtitles, books, etc.

Introduction
------------

The objective of this package is simple: If you need to compute some
sort of ``emotion analysis`` on a word or a set of words this should be
able to help. For now, it only support plain text and subtitles, but the
idea is to extend it to other formats (``pdf``, ``email``, etc.). In the
meantime, this includes a basic example on how to use it on plain text
and another example on how to use it in a collection of subtitles for
series (all episodes for all seasons of a show).

The main strategy to compute the emotions from text supported right now
is via lexicon-based word matching, which is quite straightforward and
arguably it might not need a package. However, this has a set of tools
that are easy to reuse and extend for different use cases. For example
contains tools for the analysis of subtitles in a show, but can be
easily extended to analyze books, papers, websites, customer reviews, or
even further applications like comparing a movie script with its book,
comparing properties of movies in a sequel, etc.

More advanced strategies will be added as I can assess the performance
and properly setup an experimental framework that anyone can replicate.
These will be considered as future work. However, if you have some ideas
or want to contribute, please do! just let me know how can I help :)

It's important to note that if you are using the NRC or other
proprietary lexicons you should follow their `terms of
use <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>`__.
Otherwise, if you are using this package with your own lexicons you can
use it however you want following the MIT license.

Install
-------

In the meantime I finish adding this as a `pypi` package, you can install
it by building the source code from the repository by first
installing all the dependencies from the ``requirements.txt`` file and
the dependencies for ``Spacy``, the NLP framework used through this
package.

::

    git clone https://github.com/glhuilli/limbic.git
    cd limbic
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en
    python setup.py install

Usage
-----

Below there's a step by step explanation on how to use this package, but
you can go directly to the examples included in the ``scripts`` folder.

As mentioned before, the only emotion model supported in ``limbic`` at
the moment is lexicon-based. So the first step is to get a lexicon for
emotion analysis.

Importing a lexicon-based emotion classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only thing you need to create a new lexicon-based emotion classifier
is of course the lexicon. However, in case you are dealing with a
specific context, it's possible to use a terms mapping dictionary which
will automatically replace terms on the input you want to process.

The lexicon has to be loaded by the user and it could be either a custom
lexicon or `lexicons from the
NRC <http://saifmohammad.com/WebPages/AccessResource.htm>`__. To load a
lexicon you can either use a generic ``load_lexicon`` or
``load_nrc_lexicon`` tailored for some NRC lexicons.

To use the generic ``load_lexicon`` method you can do the following:

.. code:: python

    from limbic.emotion.utils import load_lexicon

    my_lexicon_file_path = '../data/my_lexicon.csv'
    lexicon = load_lexicon(my_lexicon_file_path)

where the hypothetical file ``../data/lexicon.csv`` is a ``csv`` file
with the header ``term,emotion,score``.

To use the ``load_nrc_lexicon`` method you need to download one of the
supported NRC files, and do

.. code:: python

    from limbic.emotion.nrc_utils import load_nrc_lexicon

    nrc_lexcon_file_path = '../data/lexicons/NRC-AffectIntensity-Lexicon.txt'
    lexicon = load_nrc_lexicon(nrc_lexicon_file_path, 'affect_intensity')

Currently, the supported files are the ``affect_intensity`` lexicon, the
``emotion`` lexicon (aka ``EmoLex``), and the ``vad`` lexicon.

Finally, it's important to note that the terms mapping dictionary has to
be of type ``Dict[str, str]``, where a given term or collection of terms
will be mapped to another term of collection of terms.

Building ``limbic`` model
^^^^^^^^^^^^^^^^^^^^^^^^^

For this, you need the lexicon to be loaded and that's it. Below an
example using the ``affect_intensity`` lexicon from NRC.

.. code:: python

    from limbic.emotion.models import LexiconLimbicModel
    from limbic.emotion.nrc_utils import load_nrc_lexicon

    lexicon = load_nrc_lexicon('data/lexicons/NRC-AffectIntensity-Lexicon.txt', 'affect_intensity')
    lb = LexiconLimbicModel(lexicon)

Emotions from Terms
^^^^^^^^^^^^^^^^^^^

Once the ``limbic`` model is loaded, you can either get the emotions for
either a single term or a full sentence. For example, you can get the
emotions associated to the word ``love`` or ``hate``. Alternatively, you
can get te emotions associated to ``not love`` and ``not hate``, which
is would work by passing a ``is_negated=True`` parameter to the
``get_term_emotions`` method.

For each term, a list of ``Emotion`` named tuples will be returned. Each
``Emotion`` will have the following fields: \* ``category``: indicates
one of the motions that the term has been assigned \* ``value``:
quantifies how strong the emotion category has been assigned to the term
\* ``term``: the term for which the emotion was computed. This term in
case the method is called with ``is_negated=True`` will have a dash as a
prefix, e.g. ``term=love, is_negated=True`` will generate an ``Emotion``
with ``term=-love``.

For example,

::

    >>> for term in ['love', 'hate']:
    ...     print(f'{term} -> {lb.get_term_emotions(term)}')
    ...
    love -> [Emotion(category='joy', value=0.828, term='love')]
    hate -> [Emotion(category='anger', value=0.828, term='hate'), Emotion(category='fear', value=0.484, term='hate'), Emotion(category='sadness', value=0.656, term='hate')]

if with negated terms:

::

    >>> for term in ['LOVE', 'Hate']:
    ...     print(f'{term} (negated) -> {lb.get_term_emotions(term, is_negated=True)}')
    ...
    LOVE (negated) -> [Emotion(category='sadness', value=0.828, term='-love')]
    Hate (negated) -> [Emotion(category='fear', value=0.828, term='-hate'), Emotion(category='anger', value=0.484, term='-hate'), Emotion(category='joy', value=0.656, term='-hate')]

Negated terms
^^^^^^^^^^^^^

The categories supported for the ``is_negated`` parameter are the ones
included in the `Plutchik's wheel of
emotions <https://en.wikipedia.org/wiki/Contrasting_and_categorization_of_emotions>`__,
shown below (source: Wikipedia)

Here, each emotion is placed in a wheel where the any emotion is facing
its "opposite" in the opposite side of the wheel. For example, ``joy``
is placed to the opposite side of ``sadness``, ``rage`` on the opposite
side of ``terror``, and so on. Whenever a term is negated, the opposite
emotion will be used, as well as the ``value`` of the initial emotion.
For example, ``love`` has an emotion of ``joy`` with score ``0.828``
(following the NRC ``affect_intensity`` lexicon). Then ``love`` negated
will have an emotion of ``sadness`` with score ``0.828``.

Emotions for sentences
^^^^^^^^^^^^^^^^^^^^^^

Like getting the emotions of a term, ``limbic`` has a method for getting
the emotions for full or partial sentence. This is supported by the fact
that each sentence has multiple terms, which some of them could have one
or multiple emotions. Note that in some cases a sentence could have some
negated terms that need to be considered.

Some examples on how to process sentences and the expected output are
presented below.

::

    >>> from pprint import pprint
    >>> sentence = 'I love and enjoy this string.'
    >>> pprint(lb.get_sentence_emotions(sentence))
    [Emotion(category='joy', value=0.828, term='love'),
     Emotion(category='joy', value=0.812, term='enjoy')]
    >>> sentence = "I don't love but I enjoy this string."
    >>> pprint(lb.get_sentence_emotions(sentence))
    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy')]
    >>> sentence = "I don't love but I enjoy this sentence."
    >>> pprint(lb.get_sentence_emotions(sentence))
    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy'),
     Emotion(category='anger', value=0.203, term='sentence'),
     Emotion(category='fear', value=0.266, term='sentence'),
     Emotion(category='sadness', value=0.234, term='sentence')]

Emotions using the terms mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that in the last example
``I don't love but I enjoy this sentence``, the word ``sentence`` could
be placed under two different contexts: ``sentence`` as in a set for
words or ``sentence`` as in punishment.

If you are under the context that ``sentence`` is just a collection of
words, you can use the ``terms_mapping`` when defining the ``limbic``
object.

::

    >>> terms_mapping = {'sentence': 'string'}
    >>> lb = LexiconLimbicModel(lexicon, terms_mapping=terms_mapping)
    >>> sentence = "I don't love but I enjoy this sentence."
    >>> pprint(lb.get_sentence_emotions(sentence))
    [Emotion(category='sadness', value=0.828, term='-love'),
     Emotion(category='joy', value=0.812, term='enjoy')]


Changelog
=========

v0.0.1 (2019-05-13)
-------------------

* Initial release with basic lexicon-based emotion classifier with support for plain text and subtitles.
