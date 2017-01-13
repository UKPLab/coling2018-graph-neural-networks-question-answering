import nltk

from wikidata import wdaccess


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def possible_subentities(entity_tokens, entity_type):
    """
    Retrive all possible sub-entities of the given entity. Short title tokens are also capitalized.

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of sub-entities.
    >>> possible_subentities(["Nfl", "Redskins"], "ORGANIZATION")
    [('NFL',), ('Nfl',), ('Redskins',)]
    >>> possible_subentities(["senator"], "NN")
    [('Senator',)]
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
    >>> possible_subentities(["Jfk"], "NNP")
    [('JFK',)]
    >>> possible_subentities(["Us"], "LOCATION")
    [('US',)]
    >>> possible_subentities(['Names', 'Of', 'Walt', 'Disney'], 'ORGANIZATION')
    [('Names', 'Of', 'Walt'), ('Of', 'Walt', 'Disney'), ('Names', 'Of'), ('Of', 'Walt'), ('Walt', 'Disney'), ('OF',), ('WALT',), ('Names',), ('Of',), ('Walt',), ('Disney',)]
    """
    new_entities = []
    entity_lemmas = []
    if entity_type is "NN":
        entity_lemmas = [lemmatizer.lemmatize(n) for n in entity_tokens]
    if entity_type is "PERSON":
        if len(entity_tokens) > 2:
            new_entities.append((entity_tokens[0], entity_tokens[-1]))
        if len(entity_tokens) > 1:
            new_entities.extend([(entity_tokens[-1],), (entity_tokens[0],)])
    else:
        for i in range(len(entity_tokens) - 1, 1, -1):
            ngrams = nltk.ngrams(entity_tokens, i)
            for new_entity in ngrams:
                new_entities.append(new_entity)
        if entity_type in ['LOCATION', 'ORGANIZATION', 'NNP']:
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 5])
        if entity_type in ['NN']:
            new_entities.append(tuple([ne.title() for ne in entity_tokens]))
            if entity_lemmas != entity_tokens:
                new_entities.append(tuple(entity_lemmas))
        if len(entity_tokens) > 1:
            new_entities.extend([(ne,) for ne in entity_tokens])
            new_entities.extend([(ne,) for ne in entity_lemmas if ne not in entity_tokens])
    return new_entities


def link_entity(entity, try_subentities=True, max_options=3):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity: list of entity tokens
    :param try_subentities:
    :return: list of KB ids
    """
    entity_tokens, entity_type = entity
    linkings = wdaccess.query_wikidata(wdaccess.entity_query(" ".join(entity_tokens)))
    if entity_type is not 'URL':
        if (try_subentities and not linkings) or len(entity_tokens) == 1:
            subentities = possible_subentities(entity_tokens, entity_type)
            while subentities:
                linkings += wdaccess.query_wikidata(wdaccess.entity_query(" ".join(subentities.pop(0))))
    linkings = [l.get("e20", "") for l in linkings if l]
    linkings = sorted(linkings, key=lambda k: int(k[1:]))
    linkings = linkings[:max_options]
    return linkings