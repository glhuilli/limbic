# From Plutchik's wheel of emotions and negative/positive mapping
# Opposites depicted in the following wikipedia link:
# https://en.wikipedia.org/wiki/Contrasting_and_categorization_of_emotions#/media/File:Plutchik-wheel.svg
_ECSTASY = 'ecstasy'
_ADMIRATION = 'admiration'
_TERROR = 'terror'
_AMAZEMENT = 'amazement'
_GRIEF = 'grief'
_LOATHING = 'loathing'
_RAGE = 'rage'
_VIGILANCE = 'vigilance'

_JOY = 'joy'
_TRUST = 'trust'
_FEAR = 'fear'
_SURPRISE = 'surprise'
_SADNESS = 'sadness'
_DISGUST = 'disgust'
_ANGER = 'anger'
_ANTICIPATION = 'anticipation'

_SERENITY = 'serenity'
_ACCEPTANCE = 'acceptance'
_APPREHENSION = 'apprehension'
_DISTRACTION = 'distraction'
_PENSIVENESS = 'pensiveness'
_BOREDOM = 'boredom'
_ANNOYANCE = 'annoyance'
_INTEREST = 'interest'

_POSITIVE = 'positive'
_NEGATIVE = 'negative'

PLUTCHIK_EMOTIONS_OPPOSITE_MAPPING = {
    _ECSTASY: _GRIEF,
    _GRIEF: _ECSTASY,
    _ADMIRATION: _LOATHING,
    _LOATHING: _ADMIRATION,
    _TERROR: _RAGE,
    _RAGE: _TERROR,
    _AMAZEMENT: _VIGILANCE,
    _VIGILANCE: _AMAZEMENT,
    _JOY: _SADNESS,
    _SADNESS: _JOY,
    _TRUST: _DISGUST,
    _DISGUST: _TRUST,
    _FEAR: _ANGER,
    _ANGER: _FEAR,
    _SURPRISE: _ANTICIPATION,
    _ANTICIPATION: _SURPRISE,
    _SERENITY: _PENSIVENESS,
    _PENSIVENESS: _SERENITY,
    _ACCEPTANCE: _BOREDOM,
    _BOREDOM: _ACCEPTANCE,
    _APPREHENSION: _ANNOYANCE,
    _ANNOYANCE: _APPREHENSION,
    _DISTRACTION: _INTEREST,
    _INTEREST: _DISTRACTION,
    _POSITIVE: _NEGATIVE,
    _NEGATIVE: _POSITIVE
}
