import nltk
import re

from wikidata import wdaccess

entity_linking_p = {
    "max.entity.options": 3
}

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
roman_nums_pattern = re.compile("^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
labels_blacklist = wdaccess.load_blacklist(wdaccess.RESOURCES_FOLDER + "labels_blacklist.txt")
stop_words_en = set(nltk.corpus.stopwords.words('english'))


def possible_variants(entity_tokens, entity_type):
    """
    Construct all possible variants of the given entity,

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of entity variants
    >>> possible_variants(['the', 'current', 'senators'], 'NN')
    [('The', 'Current', 'Senators'), ('the', 'current', 'senator'), ('current', 'senators')]
    >>> possible_variants(['the', 'senator'], 'NN')
    [('The', 'Senator'), ('senator',)]
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
    [('the', 'president', 'after', 'JFK'), ('The', 'President', 'after', 'Jfk'), ('president', 'jfk')]
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
    [('Names', 'of', 'Walt', 'Disney'), ('Names', 'of', 'walt', 'disney'), ('US', 'Names', 'of', 'Walt', 'Disney')]
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
    >>> possible_variants(['St', 'Louis', 'Rams'], 'ORGANIZATION')
    [('ST', 'Louis', 'Rams'), ('St.', 'Louis', 'Rams'), ('St', 'louis', 'rams'), ('US', 'St', 'Louis', 'Rams')]
    >>> possible_variants(['united', 'states', 'of', 'america'], 'LOCATION')
    [('United', 'States', 'of', 'America')]
    >>> possible_variants(['character', 'did'], 'NN')
    [('Character', 'did'), ('character',)]
    >>> possible_variants(['Wright', 'Brothers'], 'ORGANIZATION')
    [('Wright', 'brothers')]
    >>> possible_variants(['University', 'Of', 'Leeds'], 'ORGANIZATION')
    [('University', 'of', 'Leeds'), ('University', 'of', 'leeds')]
    >>> possible_variants(['Navy'], 'ORGANIZATION')
    [('US', 'Navy')]
    >>> possible_variants(['Us', 'Army'], 'ORGANIZATION')
    [('US', 'Army'), ('Us', 'army')]
    >>> possible_variants(['House', 'Of', 'Representatives'], 'ORGANIZATION')
    [('House', 'of', 'Representatives'), ('House', 'of', 'representatives'), ('US', 'House', 'of', 'Representatives')]
    >>> possible_variants(['Michael', 'J', 'Fox'], 'PERSON')
    [('Michael', 'J.', 'Fox')]
    >>> possible_variants(['M.C.', 'Escher'], 'PERSON')
    [('M. C.', 'Escher')]
    """
    new_entities = []
    entity_lemmas = []
    if entity_type is "NN":
        entity_lemmas = [lemmatizer.lemmatize(n) for n in entity_tokens]
    if entity_type is "PERSON":
        if entity_tokens[-1].lower() == "junior":
            entity_tokens[-1] = "Jr"
        if len(entity_tokens) > 1:
            entity_tokens_no_dots = [t.replace(".","") for t in entity_tokens]
            if any(len(t) < 3 and t.lower() not in {"jr", "st"} for t in entity_tokens_no_dots):
                new_entities.append(tuple([" ".join([c.upper() + "." for c in t]) if len(t) < 3 and t.lower() not in {"jr", "st"} else t for t in entity_tokens_no_dots]))
            if any(t.startswith("Mc") for t in entity_tokens):
                new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
            if entity_tokens[-1].lower() == "jr":
                new_entities.append(tuple(entity_tokens[:-1] + [entity_tokens[-1] + "."]))
                new_entities.append(tuple(entity_tokens[:-2] + [entity_tokens[-2] + ","] + [entity_tokens[-1] + "."]))
            if entity_tokens[0].lower() == "st":
                new_entities.append(tuple(entity_tokens[:-1] + [entity_tokens[-1] + "."]))
                new_entities.append(tuple(entity_tokens[:-2] + [entity_tokens[-2] + ","] + [entity_tokens[-1] + "."]))
    elif entity_type == "URL":
        new_entity = [t + "." if len(t) == 1 else t for t in entity_tokens]
        if new_entity != entity_tokens:
            new_entities.append(new_entity)
        if any(t.startswith("Mc") for t in entity_tokens):
            new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
    else:
        upper_cased = [ne.upper() if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en else ne for ne in entity_tokens]
        if upper_cased != entity_tokens:
            new_entities.append(tuple(upper_cased))
        if "St" in entity_tokens or "st" in entity_tokens:
            new_entities.append(tuple([ne + "." if ne in {'St', 'st'} else ne for ne in entity_tokens]))
        proper_title = [ne.title() if ne.lower() not in stop_words_en or i == 0 else ne.lower() for i, ne in enumerate(entity_tokens)]
        if proper_title != entity_tokens:
            new_entities.append(tuple(proper_title))
        if entity_type in ['ORGANIZATION'] and len(entity_tokens) > 1:
            new_entities.append(tuple([entity_tokens[0].title()] + [ne.lower() for ne in entity_tokens[1:]]))
        if entity_type in ['NN']:
            if entity_lemmas != entity_tokens:
                new_entities.append(tuple(entity_lemmas))
            no_stop_title = [ne for ne in entity_tokens if ne.lower() not in stop_words_en]
            if no_stop_title != entity_tokens:
                new_entities.append(tuple(no_stop_title))
        if entity_type in {'ORGANIZATION'} and all(w not in {t.lower() for t in entity_tokens} for w in {'us', 'university', 'company', 'brothers', 'computer', 'united', 'states'}):
            new_entities.append(("US", ) + tuple(proper_title))
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
    [('the', 'current'), ('current', 'senators'), ('current',), ('senators',), ('senator',), ('Current',), ('Senators',)]
    >>> possible_subentities(["awards"], "NN")
    []
    >>> possible_subentities(["star", "wars"], "NN")
    [('star',), ('wars',), ('war',), ('Star',), ('Wars',)]
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
    [('the', 'president', 'after'), ('president', 'after', 'jfk'), ('the', 'president'), ('president', 'after'), ('after', 'jfk'), ('JFK',), ('president',), ('jfk',), ('President',), ('Jfk',)]
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
    [('dalmatians',), ('dalmatian',), ('Dalmatians',)]
    >>> possible_subentities(['Martin', 'Luther', 'King', 'Jr'], 'PERSON')
    [('Martin', 'Luther', 'King'), ('Martin',)]
    >>> possible_subentities(['romanian', 'people'], 'NN')
    [('romanian',), ('Romanian',)]
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
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en | labels_blacklist])
        if len(entity_tokens) > 1:
            new_entities.extend([(ne,) for ne in entity_tokens if not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
            new_entities.extend([(ne,) for ne in entity_lemmas if ne not in entity_tokens and not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
            if entity_type in {'NN'}:
                new_entities.extend([(ne.title(),) for ne in entity_tokens if not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
    return new_entities


def link_entity(entity, try_subentities=True):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity: list of entity tokens
    :param try_subentities:
    :return: list of KB ids
    >>> link_entity((['Martin', 'Luther', 'King', 'Junior'], 'PERSON'))
    ['Q8027', 'Q6776048']
    >>> link_entity((['movies', 'does'], 'NN'))
    []
    >>> link_entity((['lord', 'of', 'the', 'rings'], 'NN'))
    ['Q15228', 'Q127367', 'Q267982']
    >>> link_entity((["Chile"], 'NNP'))
    ['Q298', 'Q272795', 'Q1045129']
    >>> link_entity((["Bela", "Fleck"], 'NNP'))
    ['Q561390']
    >>> link_entity((["thai"], 'NN'))
    ['Q869', 'Q9217', 'Q42732']
    >>> link_entity((['romanian', 'people'], 'NN'))
    ['Q218', 'Q7913']
    >>> sorted(link_entity((['college'], 'NN')))
    ['Q189004', 'Q23002039']
    >>> link_entity((['House', 'Of', 'Representatives'], 'ORGANIZATION'))
    ['Q11701', 'Q233262', 'Q320256']
    >>> sorted(link_entity((['senator', 'of', 'the', 'state'], 'NN')))
    ['Q13217683', 'Q15686806']
    >>> link_entity((['Michael', 'J', 'Fox'], 'PERSON'))
    ['Q395274']
    """
    entity_tokens, entity_type = entity
    if " ".join(entity_tokens) in labels_blacklist or all(e.lower() in stop_words_en | labels_blacklist for e in entity_tokens):
        return []
    if any(t in wdaccess.entity_map for t in entity_tokens):
        return [e for t in entity_tokens for e in wdaccess.entity_map.get(t, [])][:entity_linking_p.get("max.entity.options", 3)]
    linkings = wdaccess.query_wikidata(wdaccess.multi_entity_query([" ".join(entity_tokens)]), starts_with=None)
    if entity_type not in {"NN"} or not linkings:
        entity_variants = possible_variants(entity_tokens, entity_type)
        entity_variants = [" ".join(s) for s in entity_variants]
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(entity_variants), starts_with=None)
    if try_subentities and not linkings: # or (len(entity_tokens) == 1 and entity_type not in {"NN"}):
        subentities = possible_subentities(entity_tokens, entity_type)
        subentities = [" ".join(s) for s in subentities]
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(subentities), starts_with=None)
    linkings = post_process_entity_linkings(linkings)
    return linkings


def post_process_entity_linkings(linkings):
    linkings = {l.get("e20", "").replace(wdaccess.WIKIDATA_ENTITY_PREFIX, "") for l in linkings if l}
    linkings = [l for l in linkings if l not in wdaccess.entity_blacklist]
    linkings = sorted(linkings, key=lambda k: int(k[1:]))
    linkings = linkings[:entity_linking_p.get("max.entity.options", 3)]
    return linkings


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
