import nltk
import itertools
from nltk.metrics import distance
import re
import numpy as np

import utils
from wikidata import wdaccess

v_structure_markers = utils.load_blacklist(utils.RESOURCES_FOLDER + "v_structure_markers.txt")

entity_linking_p = {
    "max.entity.options": 3,
    "entity.options.to.retrieve": 10,
    "min.num.links": 0,
    "respect.case": False,
    "overlaping.nn.ne": False,
    "lev.costs": (1, 0, 2),
    "always.include.subentities": False
}

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
roman_nums_pattern = re.compile("^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
# labels_blacklist = utils.load_blacklist(utils.RESOURCES_FOLDER + "labels_blacklist.txt")
labels_blacklist = set()
entity_blacklist = utils.load_blacklist(utils.RESOURCES_FOLDER + "entity_blacklist.txt")
stop_words_en = set(nltk.corpus.stopwords.words('english'))
entity_map = utils.load_entity_map(utils.RESOURCES_FOLDER + "manual_entity_map.tsv")

np_grammar = r"""
    NP:
    {<JJ|RB|CD|VBG|VBN|DT>*(<NNP|NN|NNS|NNPS>+)(<RB|CD>|<VBG|VBN|VBZ><RB>?)?(<IN|CC>(<PRP\$|DT><NN|NNS|NNP|NNPS>+|<NNP|NNPS>+))?}
    """
np_parser = nltk.RegexpParser(np_grammar)


def extract_entities_from_tagged(annotated_tokens, tags):
    """
    The method takes a list of tokens annotated with the Stanford NE annotation scheme and produces a list of entites.

    :param annotated_tokens: list of tupels where the first element is a token and the second is the annotation
    :return: list of entities each represented by the corresponding token ids

    Tests:
    >>> extract_entities_from_tagged([('what', 'O'), ('character', 'O'), ('did', 'O'), ('natalie', 'PERSON'), ('portman', 'PERSON'), ('play', 'O'), ('in', 'O'), ('star', 'O'), ('wars', 'O'), ('?', 'O')], tags={'PERSON'})
    [['natalie', 'portman']]
    >>> extract_entities_from_tagged([('Who', 'O'), ('was', 'O'), ('john', 'PERSON'), ('noble', 'PERSON')], tags={'PERSON'})
    [['john', 'noble']]
    >>> extract_entities_from_tagged([(w, 'NE' if t != 'O' else 'O') for w, t in [('Who', 'O'), ('played', 'O'), ('Aragorn', 'PERSON'), ('in', 'O'), ('the', 'ORG'), ('Hobbit', 'ORG'), ('?', 'O')]], tags={'NE'})
    [['Aragorn'], ['the', 'Hobbit']]
    """
    vertices = []
    current_vertex = []
    for i, (w, t) in enumerate(annotated_tokens):
        if t in tags:
            current_vertex.append(w)
        elif len(current_vertex) > 0:
            vertices.append(current_vertex)
            current_vertex = []
    if len(current_vertex) > 0:
        vertices.append(current_vertex)
    return vertices


def extract_entities(tokens_ne_pos):
    """
    Extract entities from the NE tags and POS tags of a sentence. Regular nouns are lemmatized to get rid of plurals.

    :param tokens_ne_pos: list of POS and NE tags.
    :return: list of entities in the order: NE>NNP>NN
    >>> extract_entities([('who', 'O', 'WP'), ('are', 'O', 'VBP'), ('the', 'O', 'DT'), ('current', 'O', 'JJ'), ('senators', 'O', 'NNS'), ('from', 'O', 'IN'), ('missouri', 'LOCATION', 'NNP'), ('?', 'O', '.')])
    [(['Missouri'], 'LOCATION'), (['the', 'current', 'senators'], 'NN')]
    >>> extract_entities([('what', 'O', 'WDT'), ('awards', 'O', 'NNS'), ('has', 'O', 'VBZ'), ('louis', 'PERSON', 'NNP'), ('sachar', 'PERSON', 'NNP'), ('won', 'O', 'NNP'), ('?', 'O', '.')])
    [(['Louis', 'Sachar'], 'PERSON'), (['Won'], 'NNP'), (['awards', 'has'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('president', 'O', 'NN'), ('after', 'O', 'IN'), ('jfk', 'O', 'NNP'), ('died', 'O', 'VBD'), ('?', 'O', '.')])
    [(['the', 'president', 'after', 'jfk'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('natalie', 'PERSON', 'NN'), ('likes', 'O', 'VBP')])
    [(['Natalie'], 'PERSON')]
    >>> extract_entities([('what', 'O', 'WDT'), ('character', 'O', 'NN'), ('did', 'O', 'VBD'), ('john', 'O', 'NNP'), \
    ('noble', 'O', 'NNP'), ('play', 'O', 'VB'), ('in', 'O', 'IN'), ('lord', 'O', 'NNP'), ('of', 'O', 'IN'), ('the', 'O', 'DT'), ('rings', 'O', 'NNS'), ('?', 'O', '.')])
    [(['John', 'Noble'], 'NNP'), (['character'], 'NN'), (['lord', 'of', 'the', 'rings'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['plays', 'O', 'VBZ'], ['lois', 'PERSON', 'NNP'], ['lane', 'PERSON', 'NNP'], ['in', 'O', 'IN'], ['superman', 'O', 'NNP'], ['returns', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Lois', 'Lane'], 'PERSON'), (['superman', 'returns'], 'NN')]
    >>> extract_entities([('the', 'O', 'DT'), ('empire', 'O', 'NN'), ('strikes', 'O', 'VBZ'), ('back', 'O', 'RB'), ('is', 'O', 'VBZ'), ('the', 'O', 'DT'), ('second', 'O', 'JJ'), ('movie', 'O', 'NN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('star', 'O', 'NN'), ('wars', 'O', 'NNS'), ('franchise', 'O', 'VBP')])
    [(['the', 'empire', 'strikes', 'back'], 'NN'), (['the', 'second', 'movie', 'in', 'the', 'star', 'wars'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['played', 'O', 'VBD'], ['cruella', 'LOCATION', 'NNP'], ['deville', 'LOCATION', 'NNP'], ['in', 'O', 'IN'], ['102', 'O', 'CD'], ['dalmatians', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Cruella', 'Deville'], 'LOCATION'), (['102', 'dalmatians'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['was', 'O', 'VBD'], ['the', 'O', 'DT'], ['winner', 'O', 'NN'], ['of', 'O', 'IN'], ['the', 'O', 'DT'], ['2009', 'O', 'CD'], ['nobel', 'O', 'NNP'], ['peace', 'O', 'NNP'], ['prize', 'O', 'NNP'], ['?', 'O', '.']])
    [(['The', '2009', 'Nobel', 'Peace', 'Prize'], 'NNP'), (['the', 'winner'], 'NN'), (['2009'], 'CD')]
    >>> extract_entities([['who', 'O', 'WP'], ['is', 'O', 'VBZ'], ['the', 'O', 'DT'], ['senator', 'O', 'NN'], ['of', 'O', 'IN'], ['connecticut', 'LOCATION', 'NNP'], ['2010', 'O', 'CD'], ['?', 'O', '.']])
    [(['Connecticut'], 'LOCATION'), (['the', 'senator'], 'NN'), (['2010'], 'CD')]
    >>> extract_entities([['Which', 'O', 'WDT'],['actors', 'O', 'NNS'],['play', 'O', 'VBP'],['in', 'O', 'IN'],['Big', 'O', 'JJ'],['Bang', 'O', 'NNP'],['Theory', 'O', 'NNP'],['?', 'O', '.']])
    [(['Big', 'Bang', 'Theory'], 'NNP'), (['actors'], 'NN')]
    >>> extract_entities([('Who', 'O', 'WP'), ('was', 'O', 'VBD'), ('on', 'O', 'IN'), ('the', 'O', 'DT'), ('Apollo', 'O', 'NNP'), ('11', 'O', 'CD'), ('mission', 'O', 'NN'), ('?', 'O', '.')])
    [(['The', 'Apollo', '11'], 'NNP'), (['mission'], 'NN')]
    >>> entity_linking_p['overlaping.nn.ne'] = True
    >>> extract_entities([['Who', 'O', 'WP'], ['is', 'O', 'VBZ'], ['the', 'O', 'DT'], ['king', 'O', 'NN'], ['of', 'O', 'IN'], ['the', 'O', 'DT'],['Netherlands', 'LOCATION', 'NNP'], ['?', 'O', '.']])
    [(['Netherlands'], 'LOCATION'), (['the', 'king', 'of', 'the', 'Netherlands'], 'NN')]
    >>> extract_entities([['Who', 'O', 'WP'], ['wrote', 'O', 'VBD'], ['the', 'O', 'DT'], ['song', 'O', 'NN'], ['Hotel', 'O', 'NNP'], ['California', 'LOCATION', 'NNP'], ['?', 'O', '.']])
    [(['California'], 'LOCATION'), (['the', 'song', 'Hotel', 'California'], 'NN')]
    >>> extract_entities((['Give', 'O', 'VB'], ['me', 'O', 'PRP'], ['all', 'O', 'DT'], ['federal', 'O', 'JJ'], ['chancellors', 'O', 'NNS'], ['of', 'O', 'IN'], ['Germany', 'LOCATION', 'NNP'], ['.', 'O', '.']))
    [(['Germany'], 'LOCATION'), (['all', 'federal', 'chancellors', 'of', 'Germany'], 'NN')]
    """
    persons = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['PERSON'])
    locations = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['LOCATION'])
    orgs = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['ORGANIZATION'])

    chunks = np_parser.parse([(w, t) for w, _, t in tokens_ne_pos] if entity_linking_p.get("overlaping.nn.ne", False) else [(w, t if p == "O" else "O") for w, p, t in tokens_ne_pos])
    nps = [el for el in chunks if type(el) == nltk.tree.Tree and el.label() == "NP"]
    # nns = extract_entities_from_tagged([(w, t) for w, _, t in tokens_ne_pos], ['NN', 'NNS'])
    # nnps = extract_entities_from_tagged([(w, t) for w, _, t in tokens_ne_pos], ['NNP', 'NNPS'])
    nnps = [[w for w, _ in el.leaves()] for el in nps if all(t not in {'NN', 'NNS'} for _, t in el.leaves())]
    nns = [[w for w, _ in el.leaves()] for el in nps if any(t in {'NN', 'NNS'} for _, t in el.leaves())]
    cds = [cd for cd in extract_entities_from_tagged([(w, t) for w, _, t in tokens_ne_pos], ['CD']) if len(cd[0]) == 4]
    # cds = [[w for w, _ in el.leaves()] for el in chunks if type(el) == nltk.tree.Tree and el.label() == "CD"]

    # sentence = " ".join([w for w, _, _ in tokens_ne_pos])
    # ne_vertices = [(k.split(), 'URL') for k in manual_entities if k in sentence]
    ne_vertices = [(ne, 'PERSON') for ne in persons] + [(ne, 'LOCATION') for ne in locations] + [(ne, 'ORGANIZATION') for ne in orgs]
    vertices = []
    for nn in nnps:
        if not ne_vertices or not all(n in v for n in nn for v, _ in ne_vertices):
            ne_vertices.append((nn, 'NNP'))
    for nn in nns:
        if not ne_vertices or not all(n in v for n in nn for v, _ in ne_vertices):
            vertices.append((nn, 'NN'))
    vertices.extend([(cd, 'CD') for cd in cds])
    if not entity_linking_p.get("respect.case", False):
        ne_vertices = [([w.title() if w.islower() else w for w in ne], pos) for ne, pos in ne_vertices]
    return ne_vertices + vertices


def possible_variants(entity_tokens, entity_type):
    """
    Construct all possible variants of the given entity,

    :param entity_tokens: a list of entity tokens
    :param entity_type:  type of the entity
    :return: a list of entity variants
    >>> possible_variants(['the', 'current', 'senators'], 'NN')
    [('The', 'Current', 'Senators'), ('the', 'current', 'senator'), ('The', 'Current', 'Senator'), ('current', 'senators'), ('Current', 'Senators'), ('current', 'senator'), ('Current', 'Senator'), ('US', 'Current', 'Senators')]
    >>> possible_variants(['the', 'senator'], 'NN')
    [('The', 'Senator'), ('senator',), ('Senator',), ('US', 'Senator')]
    >>> possible_variants(["awards"], "NN")
    [('Awards',), ('award',), ('US', 'Awards')]
    >>> possible_variants(["senators"], "NN")
    [('Senators',), ('senator',), ('US', 'Senators')]
    >>> possible_variants(["star", "wars"], "NN")
    [('Star', 'Wars'), ('star', 'war'), ('Star', 'War'), ('US', 'Star', 'Wars')]
    >>> possible_variants(["rings"], "NN")
    [('Rings',), ('ring',), ('US', 'Rings')]
    >>> possible_variants(["Jfk"], "NNP")
    [('JFK',), ('US', 'Jfk')]
    >>> possible_variants(['the', 'president', 'after', 'jfk'], 'NN')
    [('the', 'president', 'after', 'JFK'), ('The', 'President', 'after', 'Jfk'), ('president', 'jfk'), ('President', 'Jfk'), ('US', 'President', 'after', 'Jfk')]
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
    [('US', 'Super', 'Bowl', 'Xliv'), ('Super', 'Bowl', 'XLIV')]
    >>> possible_variants(['2009'], 'CD')
    []
    >>> possible_variants(['102', 'dalmatians'], 'NN')
    [('102', 'Dalmatians'), ('102', 'dalmatian'), ('102', 'Dalmatian'), ('US', '102', 'Dalmatians')]
    >>> possible_variants(['Martin', 'Luther', 'King', 'Jr'], 'PERSON')
    [('Martin', 'Luther', 'King', 'Jr.'), ('Martin', 'Luther', 'King,', 'Jr.')]
    >>> possible_variants(['St', 'Louis', 'Rams'], 'ORGANIZATION')
    [('ST', 'Louis', 'Rams'), ('St.', 'Louis', 'Rams'), ('St', 'louis', 'rams'), ('US', 'St', 'Louis', 'Rams')]
    >>> possible_variants(['St', 'Martin'], 'PERSON')
    [('St.', 'Martin')]
    >>> possible_variants(['united', 'states', 'of', 'america'], 'LOCATION')
    [('United', 'States', 'of', 'America')]
    >>> possible_variants(['character', 'did'], 'NN')
    [('Character', 'did'), ('character',), ('Character',), ('US', 'Character', 'did')]
    >>> possible_variants(['Wright', 'Brothers'], 'ORGANIZATION')
    [('Wright', 'brothers'), ('US', 'Wright', 'Brothers')]
    >>> possible_variants(['University', 'Of', 'Leeds'], 'ORGANIZATION')
    [('University', 'of', 'Leeds'), ('University', 'of', 'leeds'), ('US', 'University', 'of', 'Leeds')]
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
    [('Chancellors', 'of', 'Germany'), ('chancellor', 'of', 'Germany'), ('Chancellor', 'of', 'Germany'), ('chancellors', 'Germany'), ('Chancellors', 'Germany'), ('chancellor', 'Germany'), ('Chancellor', 'Germany'), ('US', 'Chancellors', 'of', 'Germany')]
    >>> possible_variants(['Canadians'], 'NNP')
    [('Canadian',), ('US', 'Canadians')]
    >>> possible_variants(['movies', 'does'], 'NN')
    [('Movies', 'does'), ('movie', 'doe'), ('Movie', 'Doe'), ('movies',), ('Movies',), ('movie',), ('Movie',), ('US', 'Movies', 'does')]
    >>> entity_linking_p["respect.case"] = True
    >>> possible_variants(['Canadians'], 'NNP')
    [('Canadian',), ('US', 'Canadians')]
    >>> possible_variants(['canadians'], 'NNP')
    [('canadian',), ('US', 'canadians')]
    >>> entity_linking_p["respect.case"] = False
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
            if not entity_linking_p.get("respect.case", False) and any(t.startswith("Mc") for t in entity_tokens):
                new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
            if entity_tokens[-1].lower() == "jr":
                new_entities.append(tuple(entity_tokens_no_dots[:-1] + [entity_tokens_no_dots[-1] + "."]))
                new_entities.append(tuple(entity_tokens_no_dots[:-2] + [entity_tokens_no_dots[-2] + ","] + [entity_tokens_no_dots[-1] + "."]))
            if entity_tokens[0].lower() == "st":
                new_entities.append(tuple([entity_tokens_no_dots[0] + "."] + entity_tokens_no_dots[1:]))
    elif entity_type == "URL":
        new_entity = [t + "." if len(t) == 1 else t for t in entity_tokens]
        if new_entity != entity_tokens:
            new_entities.append(new_entity)
        if any(t.startswith("Mc") for t in entity_tokens):
            new_entities.append(tuple([t if not t.startswith("Mc") or len(t) < 3 else t[:2] + t[2].upper() + t[3:] for t in entity_tokens]))
    else:
        if not entity_linking_p.get("respect.case", False):
            upper_cased = [ne.upper() if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en else ne for ne in entity_tokens]
            if upper_cased != entity_tokens:
                new_entities.append(tuple(upper_cased))
            proper_title = _get_proper_casing(entity_tokens)
            if proper_title != entity_tokens:
                new_entities.append(tuple(proper_title))
        if "St" in entity_tokens or "st" in entity_tokens:
            new_entities.append(tuple([ne + "." if ne in {'St', 'st'} else ne for ne in entity_tokens]))
        if entity_type in {'ORGANIZATION'} and len(entity_tokens) > 1:
            new_entities.append(tuple([entity_tokens[0].title()] + [ne.lower() for ne in entity_tokens[1:]]))
        if entity_type in {'NN', 'NNP'}:
            if [l.lower() for l in entity_lemmas] != [t.lower() for t in entity_tokens]:
                new_entities.append(tuple(entity_lemmas))
                if not entity_linking_p.get("respect.case", False) and (len(entity_lemmas) > 1 or entity_type is "NNP"):
                    proper_title = _get_proper_casing(entity_lemmas)
                    if proper_title != entity_tokens and proper_title != entity_lemmas:
                        new_entities.append(tuple(proper_title))
            no_stop_title = [ne for ne in entity_tokens if ne.lower() not in stop_words_en]
            if no_stop_title != entity_tokens:
                new_entities.append(tuple(no_stop_title))
                if not entity_linking_p.get("respect.case", False):
                    proper_title = _get_proper_casing(no_stop_title)
                    if proper_title != entity_tokens and proper_title != no_stop_title:
                        new_entities.append(tuple(proper_title))
            no_stop_title_lemmas = _lemmatize_tokens(no_stop_title)
            if no_stop_title_lemmas != no_stop_title and no_stop_title_lemmas != entity_tokens and no_stop_title_lemmas != entity_lemmas:
                new_entities.append(tuple(no_stop_title_lemmas))
                if not entity_linking_p.get("respect.case", False):
                    proper_title = _get_proper_casing(no_stop_title_lemmas)
                    if proper_title != entity_tokens and proper_title != no_stop_title_lemmas:
                        new_entities.append(tuple(proper_title))
        if entity_type not in {'PERSON', 'URL', 'CD'} and all(w not in {t.lower() for t in entity_tokens} for w in {'us', 'united', 'america', 'usa', 'u.s.'}):
            proper_title = _get_proper_casing([ne for i, ne in enumerate(entity_tokens) if ne.lower() not in stop_words_en or i > 0])
            new_entities.append(("US", ) + tuple(proper_title if not entity_linking_p.get("respect.case", False) else entity_tokens))
    if not entity_linking_p.get("respect.case", False) and any(roman_nums_pattern.match(ne.upper()) for ne in entity_tokens):
        new_entities.append(tuple([ne.upper() if roman_nums_pattern.match(ne.upper()) else ne for ne in entity_tokens]))
    return new_entities


def _get_proper_casing(tokens):
    return [ne.title() if ne.lower() not in stop_words_en or i == 0 else ne.lower() for i, ne in enumerate(tokens)]


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
    [('the', 'current'), ('The', 'Current'), ('current', 'senators'), ('Current', 'Senators'), ('current', 'senator'), ('Current', 'Senator'), ('current',), ('senators',), ('senator',), ('Current',), ('Senators',)]
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
    [('the', 'president', 'after'), ('The', 'President', 'after'), ('president', 'after', 'jfk'), ('President', 'after', 'Jfk'), ('the', 'president'), ('The', 'President'), ('president', 'after'), ('President', 'after'), ('after', 'jfk'), ('After', 'Jfk'), ('JFK',), ('president',), ('jfk',), ('President',), ('Jfk',)]
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
    [('Names', 'Of', 'Walt'), ('Names', 'of', 'Walt'), ('Of', 'Walt', 'Disney'), ('Names', 'Of'), ('Names', 'of'), ('Of', 'Walt'), ('Walt', 'Disney'), ('Names',), ('Walt',), ('Disney',)]
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
    [('romanian',), ('people',), ('Romanian',), ('People',)]
    >>> possible_subentities(['all', 'federal', 'chancellors', 'of', 'Germany'], 'NN')
    [('all', 'federal', 'chancellors', 'of'), ('All', 'Federal', 'Chancellors', 'of'), ('all', 'federal', 'chancellor', 'of'), ('All', 'Federal', 'Chancellor', 'of'), ('federal', 'chancellors', 'of', 'Germany'), ('Federal', 'Chancellors', 'of', 'Germany'), ('federal', 'chancellor', 'of', 'Germany'), ('Federal', 'Chancellor', 'of', 'Germany'), ('all', 'federal', 'chancellors'), ('All', 'Federal', 'Chancellors'), ('all', 'federal', 'chancellor'), ('All', 'Federal', 'Chancellor'), ('federal', 'chancellors', 'of'), ('Federal', 'Chancellors', 'of'), ('federal', 'chancellor', 'of'), ('Federal', 'Chancellor', 'of'), ('chancellors', 'of', 'Germany'), ('Chancellors', 'of', 'Germany'), ('chancellor', 'of', 'Germany'), ('Chancellor', 'of', 'Germany'), ('all', 'federal'), ('All', 'Federal'), ('federal', 'chancellors'), ('Federal', 'Chancellors'), ('federal', 'chancellor'), ('Federal', 'Chancellor'), ('chancellors', 'of'), ('Chancellors', 'of'), ('chancellor', 'of'), ('Chancellor', 'of'), ('of', 'Germany'), ('Of', 'Germany'), ('federal',), ('chancellors',), ('Germany',), ('chancellor',), ('Federal',), ('Chancellors',), ('Germany',)]
    >>> entity_linking_p["respect.case"] = True
    >>> possible_subentities(['Romanian', 'people'], 'NN');
    [('Romanian',), ('people',)]
    >>> entity_linking_p["respect.case"] = False
    """
    if len(entity_tokens) == 1:
        return []
    new_entities = []
    entity_lemmas = []
    if entity_type in {"NN", "NNP"}:
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
            lemma_ngrams = list(nltk.ngrams(entity_lemmas, i))
            for j, new_entity in enumerate(ngrams):
                new_entities.append(new_entity)
                proper_title = _get_proper_casing(new_entity)
                if not entity_linking_p.get("respect.case", False) and proper_title != entity_tokens and proper_title != list(new_entity):
                    new_entities.append(tuple(proper_title))
                if len(entity_lemmas) > 0 and lemma_ngrams[j] != new_entity:
                    new_entities.append(lemma_ngrams[j])
                    proper_title = _get_proper_casing(lemma_ngrams[j])
                    if not entity_linking_p.get("respect.case", False) and proper_title != entity_tokens and proper_title != list(lemma_ngrams[j]):
                        new_entities.append(tuple(proper_title))
        if not entity_linking_p.get("respect.case", False) and entity_type in ['LOCATION', 'ORGANIZATION', 'NNP', 'NN']:
            new_entities.extend([(ne.upper(),) for ne in entity_tokens if len(ne) < 4 and ne.upper() != ne and ne.lower() not in stop_words_en | labels_blacklist])
        if len(entity_tokens) > 1:
            new_entities.extend([(ne,) for ne in entity_tokens if not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
            new_entities.extend([(ne,) for ne in entity_lemmas if ne not in entity_tokens and not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
            if not entity_linking_p.get("respect.case", False) and entity_type in {'NN'}:
                new_entities.extend([(ne.title(),) for ne in entity_tokens if not ne.isnumeric() and ne.lower() not in stop_words_en | labels_blacklist])
    return new_entities


def _lemmatize_tokens(entity_tokens):
    """

    :param entity_tokens:
    :return:
    >>> _lemmatize_tokens(['House', 'Of', 'Representatives'])
    ['House', 'Of', 'Representative']
    """
    lemmas = [lemmatizer.lemmatize(n.lower()) for n in entity_tokens]
    lemmas = [l.title() if entity_tokens[i].istitle() else l for i, l in enumerate(lemmas)]
    return lemmas


def link_entities_in_graph(ungrounded_graph, joint_diambiguation=True):
    """
    Link all free entities in the graph.

    :param ungrounded_graph: graph as a dictionary with 'entities'
    :return: graph with entity linkings in the 'entities' array
    # >>> link_entities_in_graph({'entities': [(['Norway'], 'LOCATION'), (['oil'], 'NN')], 'tokens': ['where', 'does', 'norway', 'get', 'their', 'oil', '?']})['entities'] == \
    # [{'linkings': [('Q2480177', 'Norway')], 'type': 'LOCATION', 'tokens': ['Norway']}, {'linkings': [('Q1130872', 'Oil')], 'type': 'NN', 'tokens': ['oil']}]
    # True
    >>> 'Q223757' in [e['linkings'][0][0] for e in  link_entities_in_graph({'entities': [(['Bella'], 'PERSON'), (['Twilight'], 'NNP')], 'tokens': ['who', 'plays', 'bella', 'on', 'twilight', '?']})['entities']]
    True
    >>> link_entities_in_graph({'entities': [(['Bella'], 'PERSON'), (['2012'], 'CD')], 'tokens': ['who', 'plays', 'bella', 'on', 'twilight', '?']})['entities'] ==\
    [{'tokens': ['2012'], 'linkings': [], 'type': 'CD'}, {'tokens': ['Bella'], 'linkings': [('Q52533', 'Bella, Basilicata'), ('Q97065', 'Bella Fromm'), ('Q112242', 'Bella Alten')], 'type': 'NNP'}]
    True
    >>> 'Q15862' in [e['linkings'][0][0] for e in link_entities_in_graph({'entities': [(['first', 'Queen', 'album'], 'NN')], 'tokens': "What was the first Queen album ?".split()})['entities']]
    True
    """
    entities = _link_entities_in_sentence(ungrounded_graph.get('entities', []), ungrounded_graph.get('tokens', []))
    if joint_diambiguation:
        entities = jointly_disambiguate_entities(entities, entity_linking_p.get("min.num.links", 0))
    for e in entities:
        # If there are many linkings we take the top N, since they are still ordered by ids/lexical_overlap
        e['linkings'] = [(l.get('kbID'), l.get('label')) for l in e.get('linkings', [])]
        e['linkings'] = e['linkings'][:entity_linking_p.get("max.entity.options", 3)]
    ungrounded_graph['entities'] = entities
    return ungrounded_graph


def _link_entities_in_sentence(fragments, sentence_tokens):
    linkings = []
    entities = []
    discovered_entity_ids = set()
    if all(len(e) == 3 for e in fragments):
        return fragments
    for fragment in fragments:
        if len(fragment) == 2:
            if fragment[1] == "CD":
                entities.append({"tokens": fragment[0], "type": fragment[1], "linkings":[]})
            else:
                _linkings = _link_entity(fragment)
                _linkings = [l for l in _linkings if l.get("e2") not in discovered_entity_ids]
                discovered_entity_ids.update({l.get("e2") for l in _linkings})
                if len(_linkings) > 0:
                    for l in _linkings:
                        l['fragment'] = fragment[0]
                    _grouped_linkings = group_entities_by_overlap(_linkings)
                    for _, _linkings in _grouped_linkings:
                        _linkings = post_process_entity_linkings(_linkings)
                        linkings.extend(_linkings)
        elif len(fragment) == 3:
            entities.append(fragment)
    grouped_linkings = group_entities_by_overlap(linkings)
    for tokens, _linkings in grouped_linkings:
        if len(_linkings) > 0:
            # _linkings = post_process_entity_linkings(_linkings)
            _linkings = sorted(_linkings, key=lambda l: (l.get('lev', 0) + l.get('id_rank', 0), int(l.get('kbID', "")[1:])))
            entities.append({"linkings": _linkings, "type": 'NNP', 'tokens': _linkings[0]['fragment']})

    # if any(w in set(sentence_tokens) for w in v_structure_markers):
    #     for entity in [e for e in entities if e.get("type") != "CD" and len(e.get('tokens', [])) == 1 and "linkings" in e]:
    #         for film_id in [l.get('kbID') for e in entities for l in e.get("linkings", []) if e != entity]:
    #             character_linkings = wdaccess.query_wikidata(wdaccess.character_query(" ".join(entity.get('tokens',[])), film_id), starts_with=None)
    #             character_linkings = post_process_entity_linkings(character_linkings, entity.get("tokens"))
    #             entity['linkings'] = [l for l in character_linkings if l.get("e2") not in discovered_entity_ids] + entity.get("linkings", [])
    return entities


def jointly_disambiguate_entities(entities, min_num_links=0):
    """
    This method jointly disambiguates ambiguous entities. For each entity it selects the linkings the have
    the most connections to any linking of the other entities. Note that it can still select multiple linkinigs
    for a single entity if the have the same number of connections. Right now, it doesn't also guarantee that selected
    linkings of different entities agree, i.e. have connections between them. Although this is mostly true in practice.

    :param entities: a list of entities as dictionaries, that have 'linkings'
    :return: a list of entities as dictionaries
    :param min_num_links: filter out entities that don't have a linking that has equal or more links than specified
    >>> jointly_disambiguate_entities([{'linkings': [{'kbID': 'Q20','label': 'Norway'}, {'kbID': 'Q944765','label': 'Norway'}, {'kbID': 'Q1913264','label': 'Norway'}], 'tokens': ['Norway'], 'type': 'LOCATION'}, {'linkings': [{'kbID': 'Q42962','label': 'oil'}, {'kbID': 'Q1130872', 'label': 'Oil'}], 'tokens': ['oil'], 'type': 'NN'}]) ==\
    [{'linkings': [{'links': 0, 'kbID': 'Q20', 'label': 'Norway'}, {'links': 0, 'kbID': 'Q944765', 'label': 'Norway'}, {'links': 0, 'kbID': 'Q1913264', 'label': 'Norway'}], 'type': 'LOCATION', 'tokens': ['Norway']}, {'linkings': [{'links': 0, 'kbID': 'Q42962', 'label': 'oil'}, {'links': 0, 'kbID': 'Q1130872', 'label': 'Oil'}], 'type': 'NN', 'tokens': ['oil']}]
    True
    >>> jointly_disambiguate_entities([{'linkings': [{'kbID': 'Q20','label': 'Norway'}, {'kbID': 'Q944765','label': 'Norway'}, {'kbID': 'Q1913264','label': 'Norway'}], 'tokens': ['Norway'], 'type': 'NNP'}]) == \
    [{'linkings': [{'links': 0, 'kbID': 'Q20', 'label': 'Norway'}, {'links': 0, 'kbID': 'Q944765', 'label': 'Norway'}, {'links': 0, 'kbID': 'Q1913264', 'label': 'Norway'}], 'type': 'NNP', 'tokens': ['Norway']}]
    True
    >>> jointly_disambiguate_entities([{'linkings': [{'kbID': 'Q223757','label': 'Bella Swan'}, {'kbID': 'Q52533','label': 'Bella, Basilicata'}, {'kbID': 'Q156571','label': '695 Bella'}], 'tokens': ['Bella'], 'type': 'PERSON'}, {'linkings': [{'kbID': 'Q44523','label': 'Twilight'}, {'kbID': 'Q160071','label': 'Twilight'}, {'kbID': 'Q189378','label': 'Twilight'}], 'tokens': ['Twilight'], 'type': 'NNP'}]) == \
    [{'linkings': [{'links': 3, 'kbID': 'Q223757', 'label': 'Bella Swan'}], 'type': 'PERSON', 'tokens': ['Bella']}, {'linkings': [{'links': 1, 'kbID': 'Q44523', 'label': 'Twilight'}, {'links': 1, 'kbID': 'Q160071', 'label': 'Twilight'}, {'links': 1, 'kbID': 'Q189378', 'label': 'Twilight'}], 'type': 'NNP', 'tokens': ['Twilight']}]
    True
    >>> jointly_disambiguate_entities([{'linkings': [{'kbID': 'Q139','label': 'queen'}, {'kbID': 'Q116','label': 'monarch'}, {'kbID': 'Q15862','label': 'Queen'}], 'tokens': ['first', 'Queen', 'album'], 'type': 'NN'}, {'linkings': [{'kbID': 'Q146378','label': 'Album'}, {'kbID': 'Q482994','label': 'album'}, {'kbID': 'Q1173065','label': 'album'}], 'tokens': ['first', 'Queen', 'album'], 'type': 'NN'}, {'linkings': [{'kbID': 'Q154898','label': 'First'}, {'kbID': 'Q3746013','label': 'First'}, {'kbID': 'Q5452237','label': 'First'}], 'tokens': ['first', 'Queen', 'album'], 'type': 'NN'}]) == \
    [{'linkings': [{'links': 1, 'kbID': 'Q15862', 'label': 'Queen'}], 'type': 'NN', 'tokens': ['first', 'Queen', 'album']}, {'linkings': [{'links': 2, 'kbID': 'Q482994', 'label': 'album'}], 'type': 'NN', 'tokens': ['first', 'Queen', 'album']}, {'linkings': [{'links': 1, 'kbID': 'Q3746013', 'label': 'First'}], 'type': 'NN', 'tokens': ['first', 'Queen', 'album']}]
    True
    """
    _count_links_between_entities(entities)
    filtered_entities = []
    global_max_links = np.max([0] + [l.get('links') for e in entities for l in e.get('linkings', [])])
    for e in entities:
        if e.get("type") != "CD" and len(e.get('linkings', [])) > 0:
            max_links = np.max([l.get('links') for l in e['linkings']])
            if max_links >= min_num_links or global_max_links == 0:
                e['linkings'] = [l for l in e['linkings'] if l.get('links') == max_links]
                filtered_entities.append(e)
        else:
            filtered_entities.append(e)
    return filtered_entities


def _count_links_between_entities(entities):
    for e in entities:
        for l in e.get('linkings', []):
            l['links'] = 0
    entity_pairs = list(itertools.combinations([e for e in entities if e.get("type") != "CD"], 2))
    if not (len(entity_pairs) == 0 or all(len(e.get("linkings", [])) < 2 for e in entities)):
        for e1, e2 in entity_pairs:
            for l1 in e1['linkings']:
                for l2 in e2['linkings']:
                    have_link = wdaccess.verify_grounding(
                        {'edgeSet': [{"rightkbID": l1.get('kbID'), "leftkbID": l2.get('kbID')}]})
                    if not have_link:
                        have_link = wdaccess.verify_grounding(
                            {'edgeSet': [{"rightkbID": l1.get('kbID')}, {"rightkbID": l2.get('kbID')}]})
                    if have_link:
                        l1['links'] = l1.get('links', 0) + 1
                        l2['links'] = l2.get('links', 0) + 1


def link_entity(entity):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity: list of entity tokens
    :return: list of KB ids
    >>> entity_linking_p["entity.options.to.retrieve"] = 3
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
    [[('Q716565', 'Éowyn'), ('Q10727030', 'Eowyn'), ('Q16910118', 'Eowyn Ivey')]]
    >>> link_entity((['Jackie','Kennedy'], 'PERSON'))
    [[('Q165421', 'Jacqueline Kennedy Onassis'), ('Q9696', 'John F. Kennedy'), ('Q34821', 'Kennedy family')]]
    >>> link_entity((['JFK'], 'NNP'))
    [[('Q8685', 'John F. Kennedy International Airport'), ('Q9696', 'John F. Kennedy'), ('Q741823', 'JFK')]]
    >>> link_entity((['Kennedy'], 'PERSON'))[0][0]
    ('Q9696', 'John F. Kennedy')
    >>> link_entity((['Indian', 'company'], 'NN'))
    [[('Q102538', 'company'), ('Q225093', 'Company'), ('Q681815', 'The Company')], [('Q668', 'India'), ('Q1091034', 'Indian'), ('Q3111799', 'Indian')]]
    >>> link_entity((['Indian'], 'LOCATION'))
    [[('Q668', 'India'), ('Q1091034', 'Indian'), ('Q3111799', 'Indian')]]
    >>> [linking[0] for linking in link_entity((['supervisor', 'of', 'Albert', 'Einstein'], 'NN'))]
    [('Q937', 'Albert Einstein'), ('Q903385', 'clinical supervision')]
    >>> link_entity((['Obama'], "PERSON"))
    [[('Q76', 'Barack Obama'), ('Q13133', 'Michelle Obama'), ('Q41773', 'Obama')]]
    >>> link_entity((['Canadians'], 'NNP'))
    [[('Q16', 'Canada'), ('Q44676', 'Canadian English'), ('Q1196645', 'Canadians')]]
    >>> link_entity((['president'], 'NN'))
    [[('Q30461', 'president'), ('Q11696', 'President of the United States of America'), ('Q1255921', 'president')]]
    >>> link_entity((['all', 'federal', 'chancellors', 'of', 'Germany'], 'NN'))
    [[('Q4970706', 'Federal Chancellor of Germany'), ('Q56022', 'Chancellor of Germany'), ('Q183', 'Germany')]]
    """
    linkings = _link_entity(entity)
    grouped_linkings = []
    for _, _linkings in group_entities_by_overlap(linkings):
        _linkings = post_process_entity_linkings(_linkings, entity[0])
        grouped_linkings.append([(l.get('kbID'), l.get('label')) for l in _linkings])
    return grouped_linkings


def _link_entity(entity):
    """
    :param entity: a tuple where the first element is a list of tokens and the second element is either a part of speech tag or a NE tag
    :return: a list of linkings as dictionaries where the "e2" field contains the entity id
    :rtype list
    """
    entity_tokens, entity_type = entity
    if " ".join(entity_tokens) in labels_blacklist or all(e.lower() in stop_words_en | labels_blacklist for e in entity_tokens):
        return []
    entity_variants = possible_variants(entity_tokens, entity_type)
    subentities = possible_subentities(entity_tokens, entity_type)
    linkings = wdaccess.query_wikidata(wdaccess.multi_entity_query([" ".join(entity_tokens)]), starts_with=None)
    map_keys = {" ".join(t).lower() for t in [entity_tokens] + entity_variants + subentities}
    if any(t in entity_map for t in map_keys):
        linkings += [{'e2': e, 'label': l, 'labelright': t} for t in map_keys for e, l in entity_map.get(t, [])][:entity_linking_p.get("entity.options.to.retrieve", 3)]
    # if entity_type not in {"NN"} or not linkings:
    entity_variants = {" ".join(s) for s in entity_variants}
    linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(entity_variants), starts_with=None)
    if entity_linking_p.get("always.include.subentities", False) or not linkings: # or (len(entity_tokens) == 1 and entity_type not in {"NN"}):
        subentities = {" ".join(s) for s in subentities}
        linkings += wdaccess.query_wikidata(wdaccess.multi_entity_query(subentities), starts_with=None)
    return linkings


def post_process_entity_linkings(linkings, entity_fragment=None):
    """
    :param linkings: possible linkings as a list of dictionaries
    :param entity_fragment: list of entity tokens as appear in the sentence (optional, either that or linkings should have a key element "fragment")
    :return: sorted linkings
    >>> post_process_entity_linkings(wdaccess.query_wikidata(wdaccess.multi_entity_query({" ".join(s) for s in possible_subentities(['writers', 'studied'], "NN")}), starts_with=None), ['writers', 'studied']) == \
    [{'labelright': 'writer', 'label': 'screenwriter', 'lev': 9, 'e2': 'http://www.wikidata.org/entity/Q28389', 'kbID': 'Q28389', 'id_rank': 10.253757025176343}, {'labelright': 'writer', 'label': 'writer', 'lev': 9, 'e2': 'http://www.wikidata.org/entity/Q36180', 'kbID': 'Q36180', 'id_rank': 10.496261758949286}, {'labelright': 'writer', 'label': 'Déborah Puig-Pey Stiefel', 'lev': 8, 'e2': 'http://www.wikidata.org/entity/Q27942639', 'kbID': 'Q27942639', 'id_rank': 17.145664359730738}, {'labelright': 'Writers', 'label': 'Writers', 'lev': 9, 'e2': 'http://www.wikidata.org/entity/Q25183171', 'kbID': 'Q25183171', 'id_rank': 17.041686511931928}]
    True
    """
    # linkings = {(l.get("e2", "").replace(wdaccess.WIKIDATA_ENTITY_PREFIX, ""), l.get("label", ""), l.get("labelright", ""), l.get("fragment", "")) for l in linkings if l}
    assert all('fragment' in l for l in linkings) or entity_fragment
    discovered_entity_ids = set()
    _linkings = []
    for l in linkings:
        l['kbID'] = l.get("e2", "").replace(wdaccess.WIKIDATA_ENTITY_PREFIX, "")
        if l['kbID'] not in discovered_entity_ids:
            _linkings.append(l)
            discovered_entity_ids.add(l['kbID'])

    _linkings = [l for l in _linkings if l.get("kbID") not in entity_blacklist and l.get("kbID", "").startswith("Q")]
    # linkings = [l[:2] for l in linkings]
    for l in _linkings:
        l['lev'] = lev_distance(" ".join(entity_fragment if entity_fragment else l.get('fragment', [])), l.get('label', ""), costs=entity_linking_p.get("lev.costs", (1,1,2)))
        l['id_rank'] = np.log(int(l['kbID'][1:]))
    _linkings = sorted(_linkings, key=lambda l: (l['lev'] + l['id_rank'], int(l['kbID'][1:])))
    _linkings = _linkings[:entity_linking_p.get("entity.options.to.retrieve", 3)]
    return _linkings


def group_entities_by_overlap(entities):
    """
    Groups entities by token overlap ignoring case.

    :param entities: list of entities as tokens
    :return: a list of lists of entities
    >>> sorted(group_entities_by_overlap([{'e2':'Q36180', 'label':'writer', 'labelright':"writer"}, {'e2':'Q25183171', 'label':'Writers', 'labelright':"writer"}, {'e2':'Q28389', 'label':'screenwriter', 'labelright':"writer"}])) == \
    [({'writer'}, [{'e2': 'Q28389', 'labelright': 'writer', 'label': 'screenwriter'}, {'e2': 'Q25183171', 'labelright': 'writer', 'label': 'Writers'}, {'e2': 'Q36180', 'labelright': 'writer', 'label': 'writer'}])]
    True
    >>> sorted(group_entities_by_overlap([{'e2': 'Q36180', 'label':'star', 'labelright':"star"}, {'e2':'Q25183171', 'label':'Star Wars', 'labelright':"star wars"}, {'e2':'Q28389', 'label':'Star Wars saga', 'labelright':"star wars"}])) == \
     [({'wars', 'star', 'war'}, [{'labelright': 'star wars', 'e2': 'Q28389', 'label': 'Star Wars saga'}, {'labelright': 'star wars', 'e2': 'Q25183171', 'label': 'Star Wars'}, {'labelright': 'star', 'e2': 'Q36180', 'label': 'star'}])]
    True
    >>> sorted(group_entities_by_overlap([{'e2':'Q36180', 'label':'star', 'labelright':"star"}, {'e2':'Q25183171', 'label':'war', 'labelright':"war"}, {'e2':'Q28389', 'label':'The Wars', 'labelright':"wars"}])) == \
    [({'war', 'wars'}, [{'e2':'Q28389', 'label':'The Wars', 'labelright':"wars"}, {'e2':'Q25183171', 'label':'war', 'labelright':"war"}]), ({'star'}, [{'e2':'Q36180', 'label':'star', 'labelright':"star"}])]
    True
    """
    groupings = []
    for e in sorted(entities, key=lambda el: len(el.get('label','')), reverse=True):
        tokens = {t for t in e.get('labelright', '').lower().split()}
        tokens.update(set(_lemmatize_tokens(list(tokens))))
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
