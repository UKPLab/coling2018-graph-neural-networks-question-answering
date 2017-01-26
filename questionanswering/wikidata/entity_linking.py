import nltk
import re

from wikidata import wdaccess

entity_linking_p = {
    "max.entity.options": 3
}

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
roman_nums_pattern = re.compile("^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
stop_words_en = set(nltk.corpus.stopwords.words('english'))


def possible_variants(entity_tokens, entity_type):
    """
    Construct all possible variants of the given entity,

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of entity variants
    >>> possible_variants(['the', 'current', 'senators'], 'NN')
    [('The', 'Current', 'Senators'), ('the', 'current', 'senator')]
    >>> possible_variants(["awards"], "NN")
    [('Awards',), ('award',)]
    >>> possible_variants(["senators"], "NN")
    [('Senators',), ('senator',)]
    >>> possible_variants(["star", "wars"], "NN")
    [('Star', 'Wars'), ('star', 'war')]
    >>> possible_variants(["rings"], "NN")
    [('Rings',), ('ring',)]
    >>> possible_variants(["Jfk"], "NNP")
    [('JFK',)]
    >>> possible_variants(['the', 'president', 'after', 'jfk'], 'NN')
    [('The', 'President', 'After', 'Jfk')]
    >>> possible_variants(['Jj', 'Thomson'], 'PERSON')
    [('J. J.', 'Thomson')]
    >>> possible_variants(['J', 'J', 'Thomson'], 'URL')
    [['J.', 'J.', 'Thomson']]
    >>> possible_variants(['W', 'Bush'], 'PERSON')
    [('W.', 'Bush')]
    >>> possible_variants(["Us"], "LOCATION")
    [('US',)]
    >>> possible_variants(['Atlanta', 'United', 'States'], "LOCATION")
    []
    >>> possible_variants(['Names', 'Of', 'Walt', 'Disney'], 'ORGANIZATION')
    []
    >>> possible_variants(['Mcdonalds'], 'URL')
    [('McDonalds',)]
    >>> possible_variants(['Super', 'Bowl', 'Xliv'], 'NNP')
    [('Super', 'Bowl', 'XLIV')]
    >>> possible_variants(['2009'], 'CD')
    []
    >>> possible_variants(['102', 'dalmatians'], 'NN')
    [('102', 'Dalmatians'), ('102', 'dalmatian')]
    >>> possible_variants(['Martin', 'Luther', 'King', 'Jr'], 'PERSON')
    [('Martin', 'Luther', 'King', 'Jr.'), ('Martin', 'Luther', 'King,', 'Jr.')]
    """
    new_entities = []
    entity_lemmas = []
    if entity_type is "NN":
        entity_lemmas = [lemmatizer.lemmatize(n) for n in entity_tokens]

    if entity_type is "PERSON":
        if len(entity_tokens) > 1:
            if len(entity_tokens[0]) < 3:
                new_entities.append((" ".join([c.upper() + "." for c in entity_tokens[0]]),) + tuple(entity_tokens[1:]))
            if any(t.startswith("Mc") for t in entity_tokens):
                new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
            if entity_tokens[-1].lower() == "jr":
                new_entities.append(tuple(entity_tokens[:-1] + [entity_tokens[-1] + "."]))
                new_entities.append(tuple(entity_tokens[:-2] + [entity_tokens[-2] + ","] + [entity_tokens[-1] + "."]))
    elif entity_type == "URL":
        new_entity = [t + "." if len(t) == 1 else t for t in entity_tokens]
        if new_entity != entity_tokens:
            new_entities.append(new_entity)
        if any(t.startswith("Mc") for t in entity_tokens):
            new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
    else:
        if entity_type in ['LOCATION', 'ORGANIZATION', 'NNP', 'NN'] and len(entity_tokens) == 1:
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en])
        if entity_type in ['NN']:
            new_entities.append(tuple([ne.title() for ne in entity_tokens]))
            if entity_lemmas != entity_tokens:
                new_entities.append(tuple(entity_lemmas))
    if any(roman_nums_pattern.match(ne.upper()) for ne in entity_tokens):
        new_entities.append(tuple([ne.upper() if roman_nums_pattern.match(ne.upper()) else ne for ne in entity_tokens]))
    return new_entities


def possible_subentities(entity_tokens, entity_type):
    """
    Construct all possible sub-entities of the given entity. Short title tokens are also capitalized.

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of sub-entities.
    >>> possible_subentities(["Nfl", "Redskins"], "ORGANIZATION")
    [('NFL',), ('Nfl',), ('Redskins',)]
    >>> possible_subentities(["senators"], "NN")
    []
    >>> possible_subentities(['the', 'current', 'senators'], 'NN')
    [('the', 'current'), ('current', 'senators'), ('current',), ('senators',), ('senator',)]
    >>> possible_subentities(["awards"], "NN")
    []
    >>> possible_subentities(["star", "wars"], "NN")
    [('star',), ('wars',), ('war',)]
    >>> possible_subentities(["Grand", "Bahama", "Island"], "LOCATION")
    [('Grand', 'Bahama'), ('Bahama', 'Island'), ('Grand',), ('Bahama',), ('Island',)]
    >>> possible_subentities(["Dmitri", "Mendeleev"], "PERSON")
    [('Mendeleev',), ('Dmitri',)]
    >>> possible_subentities(["Dmitrii", "Ivanovich",  "Mendeleev"], "PERSON")
    [('Dmitrii', 'Mendeleev'), ('Mendeleev',), ('Dmitrii',)]
    >>> possible_subentities(["Victoria"], "PERSON")
    []
    >>> possible_subentities(["Jfk"], "NNP")
    []
    >>> possible_subentities(['the', 'president', 'after', 'jfk'], 'NN')
    [('the', 'president', 'after'), ('president', 'after', 'jfk'), ('the', 'president'), ('president', 'after'), ('after', 'jfk'), ('JFK',), ('president',), ('jfk',)]
    >>> possible_subentities(['Jj', 'Thomson'], 'PERSON')
    [('Thomson',), ('Jj',)]
    >>> possible_subentities(['J', 'J', 'Thomson'], 'URL')
    []
    >>> possible_subentities(['Natalie', 'Portman'], 'URL')
    []
    >>> possible_subentities(['W', 'Bush'], 'PERSON')
    [('Bush',), ('W',)]
    >>> possible_subentities(["Us"], "LOCATION")
    []
    >>> possible_subentities(['Atlanta', 'Texas'], "LOCATION")
    [('Atlanta',), ('Texas',)]
    >>> possible_subentities(['Atlanta', 'United', 'States'], "LOCATION")
    [('Atlanta', 'United'), ('United', 'States'), ('Atlanta',), ('United',), ('States',)]
    >>> possible_subentities(['Names', 'Of', 'Walt', 'Disney'], 'ORGANIZATION')
    [('Names', 'Of', 'Walt'), ('Of', 'Walt', 'Disney'), ('Names', 'Of'), ('Of', 'Walt'), ('Walt', 'Disney'), ('Names',), ('Walt',), ('Disney',)]
    >>> possible_subentities(['Timothy', 'Mcveigh'], 'PERSON')
    [('Mcveigh',), ('Timothy',)]
    >>> possible_subentities(['Mcdonalds'], 'URL')
    []
    >>> possible_subentities(['Super', 'Bowl', 'Xliv'], 'NNP')
    [('Super', 'Bowl'), ('Bowl', 'Xliv'), ('Super',), ('Bowl',), ('Xliv',)]
    >>> possible_subentities(['2009'], 'CD')
    []
    >>> possible_subentities(['102', 'dalmatians'], 'NN')
    [('dalmatians',), ('dalmatian',)]
    >>> possible_subentities(['Martin', 'Luther', 'King', 'Jr'], 'PERSON')
    [('Martin', 'Luther', 'King'), ('Martin',)]
    """
    if len(entity_tokens) == 1:
        return []
    new_entities = []
    entity_lemmas = []
    if entity_type is "NN":
        entity_lemmas = [lemmatizer.lemmatize(n) for n in entity_tokens]

    if entity_type is "PERSON":
        if entity_tokens[-1].lower() == "jr":
            new_entities.append(tuple(entity_tokens[:-1]))
            if len(entity_tokens) > 1:
                new_entities.append((entity_tokens[0],))
        else:
            if len(entity_tokens) > 2:
                new_entities.append((entity_tokens[0], entity_tokens[-1]))
            if len(entity_tokens) > 1:
                new_entities.append((entity_tokens[-1],))
                new_entities.append((entity_tokens[0],))

    elif entity_type != "URL":
        for i in range(len(entity_tokens) - 1, 1, -1):
            ngrams = nltk.ngrams(entity_tokens, i)
            for new_entity in ngrams:
                new_entities.append(new_entity)
        if entity_type in ['LOCATION', 'ORGANIZATION', 'NNP', 'NN']:
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en])
        if len(entity_tokens) > 1:
            new_entities.extend([(ne,) for ne in entity_tokens if not ne.isnumeric() and ne.lower() not in stop_words_en])
            new_entities.extend([(ne,) for ne in entity_lemmas if ne not in entity_tokens and not ne.isnumeric() and ne.lower() not in stop_words_en])
    return new_entities


def link_entity(entity, try_subentities=True):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity: list of entity tokens
    :param try_subentities:
    :return: list of KB ids
    """
    entity_tokens, entity_type = entity
    linkings = wdaccess.query_wikidata(wdaccess.entity_query(" ".join(entity_tokens)))
    if entity_type not in {"NN"} or not linkings:
        entity_variants = possible_variants(entity_tokens, entity_type)
        entity_variants = [" ".join(s) for s in entity_variants]
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(entity_variants), starts_with=None)
    if try_subentities and not linkings: # or (len(entity_tokens) == 1 and entity_type not in {"NN"}):
        subentities = possible_subentities(entity_tokens, entity_type)
        subentities = [" ".join(s) for s in subentities]
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(subentities), starts_with=None)
    linkings = [l.get("e20", "").replace(wdaccess.WIKIDATA_ENTITY_PREFIX, "") for l in linkings if l]
    linkings = sorted(linkings, key=lambda k: int(k[1:]))
    linkings = linkings[:entity_linking_p.get("max.entity.options", 3)]
    return linkings


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())