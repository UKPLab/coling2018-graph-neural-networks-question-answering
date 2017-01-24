import nltk
import re

from wikidata import wdaccess

entity_linking_p = {
    "max.entity.options": 3
}

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
roman_nums_pattern = re.compile("^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def possible_subentities(entity_tokens, entity_type):
    """
    Retrive all possible sub-entities of the given entity. Short title tokens are also capitalized.

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of sub-entities.
    >>> possible_subentities(["Nfl", "Redskins"], "ORGANIZATION")
    [('NFL',), ('Nfl',), ('Redskins',)]
    >>> possible_subentities(["senators"], "NN")
    [('Senators',), ('senator',)]
    >>> possible_subentities(['the', 'current', 'senators'], 'NN')
    [('the', 'current'), ('current', 'senators'), ('THE',), ('The', 'Current', 'Senators'), ('the', 'current', 'senator'), ('the',), ('current',), ('senators',), ('senator',)]
    >>> possible_subentities(["awards"], "NN")
    [('Awards',), ('award',)]
    >>> possible_subentities(["star", "wars"], "NN")
    [('Star', 'Wars'), ('star', 'war'), ('star',), ('wars',), ('war',)]
    >>> possible_subentities(["Grand", "Bahama", "Island"], "LOCATION")
    [('Grand', 'Bahama'), ('Bahama', 'Island'), ('Grand',), ('Bahama',), ('Island',)]
    >>> possible_subentities(["Dmitri", "Mendeleev"], "PERSON")
    [('Mendeleev',), ('Dmitri',)]
    >>> possible_subentities(["Dmitrii", "Ivanovich",  "Mendeleev"], "PERSON")
    [('Dmitrii', 'Mendeleev'), ('Mendeleev',), ('Dmitrii',)]
    >>> possible_subentities(["Victoria"], "PERSON")
    []
    >>> possible_subentities(["Jfk"], "NNP")
    [('JFK',)]
    >>> possible_subentities(['the', 'president', 'after', 'jfk'], 'NN')
    [('the', 'president', 'after'), ('president', 'after', 'jfk'), ('the', 'president'), ('president', 'after'), ('after', 'jfk'), ('THE',), ('JFK',), ('The', 'President', 'After', 'Jfk'), ('the',), ('president',), ('after',), ('jfk',)]
    >>> possible_subentities(['Jj', 'Thomson'], 'PERSON')
    [('J. J.', 'Thomson'), ('Thomson',), ('Jj',)]
    >>> possible_subentities(['J', 'J', 'Thomson'], 'URL')
    [['J.', 'J.', 'Thomson']]
    >>> possible_subentities(['Natalie', 'Portman'], 'URL')
    []
    >>> possible_subentities(['W', 'Bush'], 'PERSON')
    [('W.', 'Bush'), ('Bush',), ('W',)]
    >>> possible_subentities(["Us"], "LOCATION")
    [('US',)]
    >>> possible_subentities(['Atlanta', 'Texas'], "LOCATION")
    [('Atlanta',), ('Texas',)]
    >>> possible_subentities(['Atlanta', 'United', 'States'], "LOCATION")
    [('Atlanta', 'United'), ('United', 'States'), ('Atlanta',), ('United',), ('States',)]
    >>> possible_subentities(['Names', 'Of', 'Walt', 'Disney'], 'ORGANIZATION')
    [('Names', 'Of', 'Walt'), ('Of', 'Walt', 'Disney'), ('Names', 'Of'), ('Of', 'Walt'), ('Walt', 'Disney'), ('OF',), ('Names',), ('Of',), ('Walt',), ('Disney',)]
    >>> possible_subentities(['Timothy', 'Mcveigh'], 'PERSON')
    [('Timothy', 'McVeigh'), ('Mcveigh',), ('Timothy',)]
    >>> possible_subentities(['Mcdonalds'], 'URL')
    [('McDonalds',)]
    >>> possible_subentities(['Super', 'Bowl', 'Xliv'], 'NNP')
    [('Super', 'Bowl'), ('Bowl', 'Xliv'), ('Super',), ('Bowl',), ('Xliv',), ('Super', 'Bowl', 'XLIV')]
    >>> possible_subentities(['2009'], 'CD')
    []
    """
    new_entities = []
    entity_lemmas = []
    if entity_type is "NN":
        entity_lemmas = [lemmatizer.lemmatize(n) for n in entity_tokens]

    if entity_type is "PERSON":
        if len(entity_tokens) > 2:
            new_entities.append((entity_tokens[0], entity_tokens[-1]))
        if len(entity_tokens) > 1:
            if len(entity_tokens[0]) < 3:
                new_entities.append((" ".join([c.upper() + "." for c in entity_tokens[0]]),) + tuple(entity_tokens[1:]))
            if any(t.startswith("Mc") for t in entity_tokens):
                new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
            new_entities.extend([(entity_tokens[-1],), (entity_tokens[0],)])
    elif entity_type == "URL":
        new_entity = [t + "." if len(t) == 1 else t for t in entity_tokens]
        if new_entity != entity_tokens:
            new_entities.append(new_entity)
        if any(t.startswith("Mc") for t in entity_tokens):
            new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
    else:
        if entity_type != "URL":
            for i in range(len(entity_tokens) - 1, 1, -1):
                ngrams = nltk.ngrams(entity_tokens, i)
                for new_entity in ngrams:
                    new_entities.append(new_entity)
        if entity_type in ['LOCATION', 'ORGANIZATION', 'NNP', 'NN']:
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 4])
        # if entity_type in ['LOCATION', 'URL'] and len(entity_tokens) > 1:
        #     new_entities.extend([tuple(entity_tokens[:i]) + (entity_tokens[i] + ",", ) + tuple(entity_tokens[i+1:]) for i in range(len(entity_tokens)-1)])
        if entity_type in ['NN']:
            new_entities.append(tuple([ne.title() for ne in entity_tokens]))
            if entity_lemmas != entity_tokens:
                new_entities.append(tuple(entity_lemmas))
        if len(entity_tokens) > 1:
            new_entities.extend([(ne,) for ne in entity_tokens])
            new_entities.extend([(ne,) for ne in entity_lemmas if ne not in entity_tokens])
    if any(roman_nums_pattern.match(ne.upper()) for ne in entity_tokens):
        new_entities.append(tuple([ne.upper() if roman_nums_pattern.match(ne.upper()) else ne for ne in entity_tokens]))
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
    if (try_subentities and not linkings) or (len(entity_tokens) == 1 and entity_type not in {"NN"}):
        subentities = possible_subentities(entity_tokens, entity_type)
        while subentities:
            linkings += wdaccess.query_wikidata(wdaccess.entity_query(" ".join(subentities.pop(0))))
    linkings = [l.get("e20", "") for l in linkings if l]
    linkings = sorted(linkings, key=lambda k: int(k[1:]))
    linkings = linkings[:entity_linking_p.get("max.entity.options", 3)]
    return linkings


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())