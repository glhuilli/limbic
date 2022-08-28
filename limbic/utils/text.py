import re
from typing import Dict, Iterable, List, Optional

_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any",
    "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do",
    "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma",
    "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",
    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she",
    "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than",
    "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was",
    "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd",
    "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd",
    "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll",
    "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst",
    "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj",
    "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already",
    "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow",
    "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately",
    "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully",
    "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning",
    "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief",
    "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly",
    "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date",
    "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty",
    "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even",
    "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f",
    "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former",
    "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting",
    "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens",
    "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi",
    "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate",
    "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead",
    "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know",
    "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least",
    "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look",
    "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean",
    "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover",
    "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near",
    "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless",
    "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally",
    "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok",
    "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside",
    "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per",
    "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp",
    "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud",
    "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily",
    "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
    "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run",
    "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed",
    "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed",
    "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar",
    "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan",
    "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically",
    "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially",
    "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell",
    "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter",
    "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres",
    "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh",
    "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward",
    "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un",
    "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used",
    "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve",
    "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome",
    "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas",
    "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod",
    "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish",
    "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd",
    "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate",
    "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly",
    "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently",
    "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings",
    "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates",
    "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second",
    "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough",
    "thoroughly", "three", "well", "wonder"
})

_ES_STOPWORDS = frozenset({
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", "a", "actualmente", "acuerdo",
    "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi", "ahora", "ahí", "al",
    "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí", "alrededor", "ambos",
    "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas", "aproximadamente",
    "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla", "aquéllas",
    "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun", "aunque",
    "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas",
    "bueno", "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos",
    "cinco", "claro", "comentó", "como", "con", "conmigo", "conocer", "conseguimos", "conseguir",
    "considera", "consideró", "consigo", "consigue", "consiguen", "consigues", "contigo", "contra",
    "cosas", "creo", "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto",
    "cuantos", "cuatro", "cuenta", "cuál", "cuáles", "cuándo", "cuánta", "cuántas", "cuánto",
    "cuántos", "cómo", "d", "da", "dado", "dan", "dar", "de", "debajo", "debe", "deben", "debido",
    "decir", "dejó", "del", "delante", "demasiado", "demás", "dentro", "deprisa", "desde",
    "despacio", "despues", "después", "detras", "detrás", "dia", "dias", "dice", "dicen", "dicho",
    "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde", "dos", "durante", "día",
    "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos", "embargo", "empleais",
    "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra", "enfrente", "enseguida",
    "entonces", "entre", "era", "erais", "eramos", "eran", "eras", "eres", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban", "estabas", "estad", "estada",
    "estadas", "estado", "estados", "estais", "estamos", "estan", "estando", "estar", "estaremos",
    "estará", "estarán", "estarás", "estaré", "estaréis", "estaría", "estaríais", "estaríamos",
    "estarían", "estarías", "estas", "este", "estemos", "esto", "estos", "estoy", "estuve",
    "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese",
    "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis",
    "estuviéramos", "estuviésemos", "estuvo", "está", "estábamos", "estáis", "están", "estás",
    "esté", "estéis", "estén", "estés", "ex", "excepto", "existe", "existen", "explicó", "expresó",
    "f", "fin", "final", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron", "fuese",
    "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis", "fuéramos", "fuésemos",
    "g", "general", "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habida", "habidas",
    "habido", "habidos", "habiendo", "habla", "hablan", "habremos", "habrá", "habrán", "habrás",
    "habré", "habréis", "habría", "habríais", "habríamos", "habrían", "habrías", "habéis", "había",
    "habíais", "habíamos", "habían", "habías", "hace", "haceis", "hacemos", "hacen", "hacer",
    "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "has", "hasta", "hay", "haya",
    "hayamos", "hayan", "hayas", "hayáis", "he", "hecho", "hemos", "hicieron", "hizo", "horas",
    "hoy", "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese",
    "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis", "hubiéramos",
    "hubiésemos", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó", "intenta",
    "intentais", "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "j", "junto",
    "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar", "lo",
    "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante",
    "medio", "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios",
    "mis", "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho",
    "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni",
    "ninguna", "ningunas", "ninguno", "ningunos", "ningún", "no", "nos", "nosotras", "nosotros",
    "nuestra", "nuestras", "nuestro", "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca",
    "o", "ocho", "os", "otra", "otras", "otro", "otros", "p", "pais", "para", "parece", "parte",
    "partir", "pasada", "pasado", "paìs", "peor", "pero", "pesar", "poca", "pocas", "poco", "pocos",
    "podeis", "podemos", "poder", "podria", "podriais", "podriamos", "podrian", "podrias", "podrá",
    "podrán", "podría", "podrían", "poner", "por", "por qué", "porque", "posible", "primer",
    "primera", "primero", "primeros", "principalmente", "pronto", "propia", "propias", "propio",
    "propios", "proximo", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
    "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
    "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó",
    "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "sal",
    "salvo", "se", "sea", "seamos", "sean", "seas", "segun", "segunda", "segundo", "según", "seis",
    "ser", "sera", "seremos", "será", "serán", "serás", "seré", "seréis", "sería", "seríais",
    "seríamos", "serían", "serías", "seáis", "señaló", "si", "sido", "siempre", "siendo", "siete",
    "sigue", "siguiente", "sin", "sino", "sobre", "sois", "sola", "solamente", "solas", "solo",
    "solos", "somos", "son", "soy", "soyos", "su", "supuesto", "sus", "suya", "suyas", "suyo",
    "suyos", "sé", "sí", "sólo", "t", "tal", "tambien", "también", "tampoco", "tan", "tanto",
    "tarde", "te", "temprano", "tendremos", "tendrá", "tendrán", "tendrás", "tendré", "tendréis",
    "tendría", "tendríais", "tendríamos", "tendrían", "tendrías", "tened", "teneis", "tenemos",
    "tener", "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis", "tenida", "tenidas",
    "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais", "teníamos", "tenían", "tenías",
    "tercera", "ti", "tiempo", "tiene", "tienen", "tienes", "toda", "todas", "todavia", "todavía",
    "todo", "todos", "total", "trabaja", "trabajais", "trabajamos", "trabajan", "trabajar",
    "trabajas", "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuve", "tuviera",
    "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses",
    "tuvimos", "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo", "tuya", "tuyas", "tuyo",
    "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos", "usa", "usais", "usamos",
    "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor", "vamos", "van",
    "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
    "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y",
    "ya", "yo", "z", "él", "éramos", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos",
    "última", "últimas", "último", "últimos"
})


def _clean_text(text):
    """
    General cleanup of anything that might not match a word.
    """
    if not text:  # if no text then skip processing
        return ''
    if 'font color' in text or '@' in text:  # skip / clean anything related to subtitles
        return ''
    text = text.lower()
    text = text.replace('<i>', '')  # skip / clean anything related to html
    text = text.replace('<\\i>', '')  # skip / clean anything related to html

    text = re.sub(r"[^\w\-\'\s\.]", '',
                  text)  # remove all non name related characters (\w, '-' and "'")
    text = re.sub(r'\d|\_|\*', '', text)  # remove digits, '_' and '*' which are matched by \w
    text = re.sub(r'(^|\s+)-(\s+|$)', '', text)  # remove "-" when not used in between a word
    text = re.sub(r'\.+', ' ', text)  # remove "." when it's at the end of the sentence
    text = re.sub(r'\s+', ' ', text).strip()  # clean all extra spaces
    if text.count("'") > 1:
        text = re.sub(r'\'', '', text)
    return text


def _tokenizer(sentence: str) -> Iterable[str]:
    """
    Tokenize a sentence.

    TODO: Explore more advanced tokenization strategies (e.g. Spacy)
    """
    yield from sentence.split(' ')


def _remove_stopwords(sentence_tokens: Iterable[str], language: str) -> Iterable[str]:
    """
    Exclude any stop word in the input set of tokens
    """
    if language == 'en-us':
        stopwords = _STOPWORDS
    elif language == 'es':
        stopwords = _ES_STOPWORDS
    else:
        stopwords = _STOPWORDS
    yield from [x for x in sentence_tokens if x.lower() not in stopwords]


def process_content(sentence: str,
                    terms_mapping: Optional[Dict[str, str]] = None,
                    language: str = 'en-us') -> List[str]:
    """
    Process any input sentence using a dictionary of terms to be mapped, and returning
    a clean set of tokens excluding all stop words.

    TODO: Tokenizer should also consider multi-language capabilities
    """
    if terms_mapping:
        for term, mapping in terms_mapping.items():
            sentence = re.sub(rf'\b{term}\b', mapping, sentence, flags=re.I)
    return list(_remove_stopwords(_tokenizer(_clean_text(sentence)), language))
