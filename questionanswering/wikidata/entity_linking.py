import nltk
from nltk.metrics import distance
import re
import numpy as np

import utils
from wikidata import wdaccess

entity_linking_p = {
    "max.entity.options": 3
}

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
roman_nums_pattern = re.compile("^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
# labels_blacklist = utils.load_blacklist(utils.RESOURCES_FOLDER + "labels_blacklist.txt")
labels_blacklist = set()
entity_blacklist = utils.load_blacklist(utils.RESOURCES_FOLDER + "entity_blacklist.txt")
stop_words_en = set(nltk.corpus.stopwords.words('english'))
entity_map = utils.load_entity_map(utils.RESOURCES_FOLDER + "manual_entity_map.tsv")


def possible_variants(entity_tokens, entity_type):
    """
    Construct all possible variants of the given entity,

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of entity variants
    >>> possible_variants(['the', 'current', 'senators'], 'NN')
    [('The', 'Current', 'Senators'), ('the', 'current', 'senator'), ('The', 'Current', 'Senator'), ('current', 'senators')]
    >>> possible_variants(['the', 'senator'], 'NN')
    [('The', 'Senator'), ('senator',)]
    >>> possible_variants(["awards"], "NN")
    [('Awards',), ('award',)]
    >>> possible_variants(["senators"], "NN")
    [('Senators',), ('senator',)]
    >>> possible_variants(["star", "wars"], "NN")
    [('Star', 'Wars'), ('star', 'war'), ('Star', 'War')]
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
    [('102', 'Dalmatians'), ('102', 'dalmatian'), ('102', 'Dalmatian')]
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
    >>> possible_variants(['chancellors', 'of', 'Germany'], 'NN')
    [('Chancellors', 'of', 'Germany'), ('chancellor', 'of', 'germany'), ('Chancellor', 'of', 'Germany'), ('chancellors', 'Germany')]
    >>> possible_variants(['Canadians'], 'NNP')
    [('canadian',), ('Canadian',)]
    """
    new_entities = []
    entity_lemmas = []
    if entity_type in {'NN', 'NNP'}:
        entity_lemmas = _lemmatize_tokens(entity_tokens)
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
        if entity_type in {'ORGANIZATION'} and len(entity_tokens) > 1:
            new_entities.append(tuple([entity_tokens[0].title()] + [ne.lower() for ne in entity_tokens[1:]]))
        if entity_type in {'NN', 'NNP'}:
            if [l.lower() for l in entity_lemmas] != [t.lower() for t in entity_tokens]:
                new_entities.append(tuple(entity_lemmas))
                if len(entity_lemmas) > 1 or entity_type is "NNP":
                    proper_title = [ne.title() if ne.lower() not in stop_words_en or i == 0 else ne.lower() for i, ne in enumerate(entity_lemmas)]
                    if proper_title != entity_tokens:
                        new_entities.append(tuple(proper_title))
            no_stop_title = [ne for ne in entity_tokens if ne.lower() not in stop_words_en]
            if no_stop_title != entity_tokens:
                new_entities.append(tuple(no_stop_title))
        if entity_type not in {'PERSON', 'URL'} and all(w not in {t.lower() for t in entity_tokens} for w in {'us', 'united', 'america', 'usa', 'u.s.'}):
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
        entity_lemmas = _lemmatize_tokens(entity_tokens)

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


def _lemmatize_tokens(entity_tokens):
    return [lemmatizer.lemmatize(n.lower()) for n in entity_tokens]


def link_entity(entity, try_subentities=True):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity: list of entity tokens
    :param try_subentities:
    :return: list of KB ids
    >>> link_entity((['Martin', 'Luther', 'King', 'Junior'], 'PERSON'))
    [[('Q8027', 'Martin Luther King, Jr.'), ('Q6776048', 'Martin Luther King, Jr.')]]
    >>> link_entity((['movies', 'does'], 'NN'))
    [[('Q11424', 'film'), ('Q1179487', 'Movies'), ('Q6926907', 'Movies')]]
    >>> link_entity((['lord', 'of', 'the', 'rings'], 'NN'))
    [[('Q15228', 'The Lord of the Rings'), ('Q127367', 'The Lord of the Rings: The Fellowship of the Ring'), ('Q131074', 'The Lord of the Rings')]]
    >>> link_entity((['state'], 'NN'))
    [[('Q7275', 'state'), ('Q230855', 'state of physical system'), ('Q599031', 'state of information system')]]
    >>> link_entity((["Chile"], 'NNP'))
    [[('Q298', 'Chilito'), ('Q1045129', '4636 Chile'), ('Q272795', 'Tacna')]]
    >>> link_entity((["Bela", "Fleck"], 'NNP'))
    [[('Q561390', 'Béla Fleck')]]
    >>> link_entity((["thai"], 'NN'))
    [[('Q869', 'Thailand'), ('Q9217', 'Thai'), ('Q42732', 'Thai')]]
    >>> link_entity((['romanian', 'people'], 'NN'))
    [[('Q33659', 'People'), ('Q3238275', 'Homo sapiens sapiens'), ('Q2472587', 'people')], [('Q218', 'Romania'), ('Q7913', 'Romanian')]]
    >>> link_entity((['college'], 'NN'))
    [[('Q189004', 'college'), ('Q1459186', 'college'), ('Q728520', 'College')]]
    >>> link_entity((['House', 'Of', 'Representatives'], 'ORGANIZATION'))
    [[('Q11701', 'United States House of Representatives'), ('Q233262', 'House of Representatives'), ('Q320256', 'House of Representatives')]]
    >>> link_entity((['senator', 'of', 'the', 'state'], 'NN'))
    [[('Q13217683', 'senator'), ('Q15686806', 'senator')]]
    >>> link_entity((['Michael', 'J', 'Fox'], 'PERSON'))
    [[('Q395274', 'Michael J. Fox')]]
    >>> link_entity((['Eowyn'], 'PERSON'))
    [[('Q716565', 'Éowyn'), ('Q10727030', 'Eowyn')]]
    >>> link_entity((['Jackie','Kennedy'], 'PERSON'))
    [[('Q165421', 'Jacqueline Kennedy Onassis'), ('Q9696', 'John F. Kennedy'), ('Q34821', 'Kennedy family')]]
    >>> link_entity((['JFK'], 'NNP'))
    [[('Q8685', 'John F. Kennedy International Airport'), ('Q9696', 'John F. Kennedy'), ('Q741823', 'JFK')]]
    >>> link_entity((['Kennedy'], 'PERSON'))
    [[('Q9696', 'John F. Kennedy'), ('Q34821', 'Kennedy family'), ('Q67761', 'Kennedy')]]
    >>> link_entity((['Indian', 'company'], 'NN'))
    [[('Q102538', 'company'), ('Q225093', 'Company'), ('Q681815', 'The Company')], [('Q668', 'India'), ('Q1091034', 'Indian'), ('Q3111799', 'Indian')]]
    >>> link_entity((['Indian'], 'LOCATION'))
    [[('Q668', 'India'), ('Q1091034', 'Indian'), ('Q3111799', 'Indian')]]
    >>> link_entity((['supervisor', 'of', 'Albert', 'Einstein'], 'NN'))
    [[('Q937', 'Albert Einstein'), ('Q1168822', 'house of Albert of Luynes'), ('Q152245', 'Albert, Prince Consort')], [('Q903385', 'clinical supervision'), ('Q1240788', 'Supervisor'), ('Q363802', 'doctoral advisor')]]
    >>> link_entity((['Obama'], "PERSON"))
    [[('Q76', 'Barack Obama'), ('Q41773', 'Obama'), ('Q5280414', 'Obama')]]
    >>> link_entity((['Canadians'], 'NNP'))
    [[('Q16', 'Canada'), ('Q44676', 'Canadian English'), ('Q1196645', 'Canadians')]]
    >>> link_entity((['president'], 'NN'))
    [[('Q30461', 'president'), ('Q11696', 'President of the United States of America'), ('Q1255921', 'president')]]
    """
    entity_tokens, entity_type = entity
    if " ".join(entity_tokens) in labels_blacklist or all(e.lower() in stop_words_en | labels_blacklist for e in entity_tokens):
        return []
    entity_variants = possible_variants(entity_tokens, entity_type)
    subentities = possible_subentities(entity_tokens, entity_type)
    linkings = wdaccess.query_wikidata(wdaccess.multi_entity_query([" ".join(entity_tokens)]), starts_with=None)
    map_keys = {" ".join(t).lower() for t in [entity_tokens] + entity_variants + subentities}
    if any(t in entity_map for t in map_keys):
        linkings += [{'e2': e, 'label': l, 'labelright': t} for t in map_keys for e, l in entity_map.get(t, [])][:entity_linking_p.get("max.entity.options", 3)]
    # if entity_type not in {"NN"} or not linkings:
    entity_variants = {" ".join(s) for s in entity_variants}
    linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(entity_variants), starts_with=None)
    if try_subentities and not linkings: # or (len(entity_tokens) == 1 and entity_type not in {"NN"}):
        subentities = {" ".join(s) for s in subentities}
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(subentities), starts_with=None)
    linkings = post_process_entity_linkings(entity_tokens, linkings)
    return linkings


def post_process_entity_linkings(entity_tokens, linkings):
    """
    :param entity_tokens: list of entity tokens as appear in the sentence
    :param linkings: possible linkings
    :return: sorted linkings
    >>> post_process_entity_linkings(['writers', 'studied'], wdaccess.query_wikidata(wdaccess.multi_entity_query({" ".join(s) for s in possible_subentities(['writers', 'studied'], "NN")}), starts_with=None))
    [[('Q36180', 'writer'), ('Q25183171', 'Writers'), ('Q28389', 'screenwriter')]]
    """
    linkings = {(l.get("e2", "").replace(wdaccess.WIKIDATA_ENTITY_PREFIX, ""), l.get("label", ""), l.get("labelright", "")) for l in linkings if l}
    linkings = [l for l in linkings if l[0] not in entity_blacklist]
    grouped_linkings = []
    for linkings in [g[1] for g in group_entities_by_overlap(linkings)]:
        linkings = [l[:2] for l in linkings]
        linkings = [l + (lev_distance(" ".join(entity_tokens), l[1], costs=(1, 0, 2)),) for l in linkings]
        linkings = {l + (np.log(int(l[0][1:])),) for l in linkings if l[0].startswith("Q")}
        linkings = sorted(linkings, key=lambda k: (k[-2] + k[-1], int(k[0][1:])))
        linkings = linkings[:entity_linking_p.get("max.entity.options", 3)]
        linkings = [l[:2] for l in linkings]
        grouped_linkings.append(linkings)
    return grouped_linkings


def group_entities_by_overlap(entities):
    """
    Groups entities by token overlap ignoring case.

    :param entities: list of entities as tokens
    :return: a list of lists of entities
    # >>> group_entities_by_overlap([('star',), ('wars',), ('war',), ('Star',), ('Wars',)])
    # [({'star'}, [('star',), ('Star',)]), ({'wars', 'war'}, [('wars',), ('war',), ('Wars',)])]
    # >>> group_entities_by_overlap([('the', 'president', 'after'), ('president', 'after', 'jfk'), ('the', 'president'), ('president', 'after'), ('after', 'jfk'), ('JFK',), ('president',), ('jfk',), ('President',), ('Jfk',)]) == \
    # [({'after', 'the', 'jfk', 'president'}, [('the', 'president', 'after'), ('president', 'after', 'jfk'), ('the', 'president'), ('president', 'after'), ('after', 'jfk'), ('JFK',), ('president',), ('jfk',), ('President',), ('Jfk',)])]
    # True
    >>> group_entities_by_overlap([('Q36180', 'writer', "writer"), ('Q25183171', 'Writers', "writer"), ('Q28389', 'screenwriter', "writer")])
    [({'writer'}, [('Q36180', 'writer', 'writer'), ('Q25183171', 'Writers', 'writer'), ('Q28389', 'screenwriter', 'writer')])]
    >>> group_entities_by_overlap([('Q36180', 'star', "star"), ('Q25183171', 'Star Wars', "star wars"), ('Q28389', 'Star Wars saga', "star wars")])
    [({'wars', 'star', 'war'}, [('Q36180', 'star', 'star'), ('Q25183171', 'Star Wars', 'star wars'), ('Q28389', 'Star Wars saga', 'star wars')])]
    >>> group_entities_by_overlap([('Q36180', 'star', "star"), ('Q25183171', 'war', "war"), ('Q28389', 'The Wars', "wars")])
    [({'star'}, [('Q36180', 'star', 'star')]), ({'wars', 'war'}, [('Q25183171', 'war', 'war'), ('Q28389', 'The Wars', 'wars')])]
    """
    groupings = []
    for e in sorted(entities, key=lambda el: len(el[1]), reverse=True):
        tokens = {t for t in e[2].lower().split()}
        tokens.update(set(_lemmatize_tokens(tokens)))
        i = 0
        while len(groupings) > i >= 0:
            k, entities = groupings[i]
            if len(tokens & k) > 0:
                entities.append(e)
                k.update(tokens)
                i = -1
            else:
                i += 1
        if i == len(groupings):
            groupings.append((tokens, [e]))
    return groupings


def lev_distance(s1, s2, costs=(1, 1, 1)):
    """
    Levinstein distance with adjustable costs

    :param s1: first string
    :param s2: second string
    :param costs: a tuple of costs: (remove, add, substitute)
    :return: a distance as an integer number
    >>> lev_distance("Obama", "Barack Obama") == distance.edit_distance("Obama", "Barack Obama")
    True
    >>> lev_distance("Chili", "Tacna") == distance.edit_distance("Chili", "Tacna")
    True
    >>> lev_distance("Lord of the Rings", "lord of the rings") == distance.edit_distance("Lord of the Rings", "lord of the rings")
    True
    >>> lev_distance("Lord of the Rings", "") == distance.edit_distance("Lord of the Rings", "")
    True
    >>> lev_distance("Chili", "Tabac", costs=(1,1,2)) == distance.edit_distance("Chili", "Tabac", substitution_cost=2)
    True
    >>> lev_distance("Obama", "Barack Obama", costs=(1,0,1))
    0
    >>> lev_distance("Obama", "Barack Obama", costs=(0,2,1))
    14
    >>> lev_distance("Obama II", "Barack Obama", costs=(1,0,1))
    3
    >>> lev_distance("Chile", "Tacna", costs=(2,1,2))
    10
    >>> lev_distance("Chile", "Chilito", costs=(2,1,2))
    4
    """

    len1 = len(s1)
    len2 = len(s2)
    a_cost, b_cost, c_cost = costs
    lev = np.zeros((len1+1, len2+1), dtype='int16')
    if a_cost > 0:
        lev[:, 0] = list(range(0, len1*a_cost+1, a_cost))
    if b_cost > 0:
        lev[0] = list(range(0, len2*b_cost+1, b_cost))
    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            c1 = s1[i]
            c2 = s2[j]
            a = lev[i, j+1] + a_cost  # skip character in s1 -> remove
            b = lev[i+1, j] + b_cost  # skip character in s2 -> add
            c = lev[i, j] + (c_cost if c1 != c2 else 0) # substitute
            lev[i+1][j+1] = min(a, b, c)
    return lev[-1, -1]


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
