import itertools
import nltk
import copy
import re

import utils


def graph_has_temporal(g):
    """
    Test if there are temporal relation in the graph.

    :param g: graph as a dictionary
    :return: True if graph has temporal relations, False otherwise
    """
    return any(any(edge.get(p) == 'time' for p in {'argmax', 'argmin', 'type'}) or 'num' in edge for edge in g.get('edgeSet', []))


def if_graph_adheres(g, allowed_extensions=set()):
    """
    Test if the given graphs only uses the allowed extensions.

    :param g: graphs a dictionary with an edgeSet
    :param allowed_extensions: a set of allowed extensions
    :return: True if graph uses only allowed extensions, false otherwise
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}]}, allowed_extensions=set())
    True
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'v-structure'}]}, allowed_extensions=set())
    False
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','argmin': 'time'}]}, allowed_extensions=set())
    False
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'time'}]}, allowed_extensions={'temporal'})
    True
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland']}, {'kbID':'P31v'}]}, allowed_extensions=set())
    False
    """
    allowed_extensions = set(allowed_extensions)
    if 'v-structure' not in allowed_extensions and 'v-structure' in {e.get('type', "direct") for e in g.get('edgeSet', [])}:
        return False
    if 'temporal' not in allowed_extensions and graph_has_temporal(g):
        return False
    if 'hopUp' not in allowed_extensions and any('hopUp' in e for e in g.get('edgeSet', [])):
        return False
    if 'hopDown' not in allowed_extensions and any('hopDown' in e for e in g.get('edgeSet', [])):
        return False
    if 'qualifier_rel' not in allowed_extensions and any(e.get('kbID', "").endswith('q') for e in g.get('edgeSet', [])):
        return False
    if 'multi_rel' not in allowed_extensions and len(g.get('edgeSet', [])) > 1:
        return False
    if 'filter' not in allowed_extensions and any('filter' in e for e in g.get('edgeSet', [])):
        return False
    return True


def get_property_str_representation(edge, property2label, use_placeholder=False):
    """
    Construct a string representation of a label using the property to label mapping.

    :param edge: edge to translate
    :param property2label: property id to label mapping
    :param use_placeholder: if an entity should be included or just a placeholder
    :return: a string representation of an edge
    >>> get_property_str_representation({'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}, {'P17': "country"})
    'country Iceland'
    >>> get_property_str_representation({'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}, {'P17': "country"}, use_placeholder=True)
    'country <e>'
    >>> get_property_str_representation({'argmax':'time','type': 'time'}, {'P17': "country"}, use_placeholder=True)
    '<argmax> <e>'
    >>> get_property_str_representation({'kbID': 'P31v',   'right': ['Australia'], \
   'rightkbID': 'Q408', 'type': 'reverse'}, {'P31': "instance of"}, use_placeholder=True)
    '<e> instance of'
    >>> get_property_str_representation({'argmin': 'time', 'kbID': 'P17s',  'right': ['IC'], \
    'rightkbID': 'Q189', 'canonical_right': 'Iceland', 'type': 'direct'}, {'P17': "country"}, use_placeholder=False)
    '<argmin> country Iceland'
    >>> get_property_str_representation({'hopUp': 'P131v', 'kbID': 'P69s',  'right': ['Missouri'], \
    'rightkbID': 'Q189', 'type': 'direct'}, {'P69': "educated at", 'P131': 'located in'}, use_placeholder=True)
    'educated at <x> located in <e>'
    >>> get_property_str_representation({'hopUp': 'P17v', 'kbID': 'P140v',  'right': ['Russia'], \
    'rightkbID': 'Q159', 'type': 'reverse'}, {'P17': "country", 'P140': 'religion'}, use_placeholder=True)
    '<x> country <e> religion'
    >>> get_property_str_representation({'hopUp': None, 'kbID': 'P140v',  'right': ['Russia'], \
    'rightkbID': 'Q159', 'type': 'reverse'}, {'P17': "country", 'P140': 'religion'}, use_placeholder=True)
    '<e> religion'
    >>> get_property_str_representation({'kbID': 'P453q', 'right': ['Natalie', 'Portman'], 'type': 'reverse', 'hopUp': 'P161v', 'rightkbID': 'Q37876', 'canonical_right': 'Natalie Portman'},  {'P453': "character role", "P161": "cast member" }, use_placeholder=True)
    '<x> cast member <e> character role'
    >>> get_property_str_representation({'canonical_right': 'Indiana', 'hopUp': 'P1001v', 'kbID': 'P39v', 'type': 'direct'}, {'P39': 'position held', 'P1001':'applies to territorial jurisdiction'}, use_placeholder=True)
    'position held <x> applies to territorial jurisdiction <e>'
    >>> get_property_str_representation({'canonical_right': 'Facebook','hopDown': 'P17v','kbID': 'P150v','type': 'reverse'}, {'P17':'country', 'P150':'contains administrative territorial entity'}, use_placeholder=True)
    '<e> country <x> contains administrative territorial entity'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct'}, {'P361': 'part of'}, use_placeholder=True)
    'part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Meg Griffin', 'kbID': 'P161v', 'type': 'v-structure'}, {'P161': "cast member"}, use_placeholder=True)
    '<v> cast member <e>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'argmax':'time'}, {'P361': 'part of'}, use_placeholder=True)
    '<argmax> part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'filter':'importance'}, {'P361': 'part of'}, use_placeholder=True)
    '<filter> part of <e> part of <x>'
    >>> get_property_str_representation({'kbID': 'P69s',  'right': ['Missouri'], \
    'rightkbID': 'Q189', 'type': 'direct', 'num': '2012'}, {'P69': "educated at", 'P131': 'located in'}, use_placeholder=True)
    'educated at <e> <num>'
    """
    e_type = edge.get('type', 'direct')
    if e_type == "time":
        property_label = ""
    else:
        property_label = property2label.get(edge.get('kbID', '')[:-1], utils.unknown_el)
    if e_type == 'v-structure':
        property_label = "<v> " + property_label
    e_arg = '<argmax> ' if 'argmax' in edge else '<argmin> ' if 'argmin' in edge else ""
    num = "<num>" if "num" in edge else ""
    if 'filter' in edge and edge['filter'] == 'importance':
        e_arg += "<filter> "
    hopUp_label, hopDown_label = '', ''
    if 'hopUp' in edge and edge['hopUp']:
        hopUp_label = "<x> {} ".format(property2label.get(edge['hopUp'][:-1], utils.unknown_el))
    elif 'hopDown' in edge and edge['hopDown']:
        hopDown_label = " {} <x>".format(property2label.get(edge['hopDown'][:-1], utils.unknown_el))
    entity_name = "<e>" if use_placeholder \
        else edge["canonical_right"] if "canonical_right" in edge \
        else " ".join(edge.get('right', []))
    property_label = ("{2}{3}{1}{4} {0} {5}" if e_type in {'reverse', 'time'} else "{2}{0} {3}{1}{4} {5}").format(
        property_label, entity_name, e_arg, hopUp_label, hopDown_label, num)
    property_label = property_label.strip()
    return property_label


def add_string_representations_to_edges(g, property2label, use_placeholder=False):
    """
    To each edge in the graph add its string representation as a label.

    :param g: graph as a distionary with an 'edgeSet'
    :param property2label: properties to labels mapping
    :param use_placeholder: if a placeholder should be used for entities or not
    :return: the orginal graph
    >>> add_string_representations_to_edges({'edgeSet': [{'kbID':'P17v', 'right': ['Nfl', 'Redskins'], 'type':'reverse'}], 'tokens': ['where', 'are', 'the', 'nfl', 'redskins', 'from', '?']}, {'P17': "country"})['edgeSet'][0]['label']
    'Nfl Redskins country'
    """
    for edge in g.get('edgeSet', []):
        edge_label = get_property_str_representation(edge, property2label, use_placeholder)
        edge['label'] = edge_label
    return g


def replace_entities(g):
    """
    Replaces an entity participating in the first relation in the graph with <e>

    :param g: graph as a dictionary
    :return: the original graph modified
    >>> replace_entities({'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', 'nfl', 'redskins', 'from', '?']}) == {'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', '<e>', 'from', '?']}
    True
    >>> replace_entities({'edgeSet': [{'canonical_right': 'Vasco Núñez de Balboa', 'right': ['Vasco', 'Nunez', 'De', 'Balboa'], 'kbID': 'P106v', 'type': 'reverse'},\
   {'canonical_right': 'Reise', 'hopUp': 'P279v', 'kbID': 'P425v', 'type': 'direct', 'right': ['journey']}], \
   'tokens': ['what', 'was', 'vasco', 'nunez', 'de', 'balboa', 'original', 'purpose', 'of', 'his', 'journey', '?']})['tokens']
    ['what', 'was', '<e>', 'original', 'purpose', 'of', 'his', 'journey', '?']
    >>> replace_entities({'edgeSet': [{'right': ['House', 'Of', "Representatives"]}], 'tokens': "what is the upper house of the house of representatives ?".split()})['tokens']
    ['what', 'is', 'the', 'upper', 'house', 'of', 'the', '<e>', '?']
    """
    tokens = g.get('tokens', [])
    edge = get_graph_first_edge(g)
    entity = [t.lower() for t in edge.get('right', [])]
    new_tokens = []
    entity_pos = 0
    for i, t in enumerate(tokens):
        if entity_pos == len(entity) or t != entity[entity_pos]:
            if entity_pos > 0:
                if entity_pos == len(entity):
                    new_tokens.append("<e>")
                else:
                    new_tokens.extend(entity[:entity_pos])
                entity_pos = 0
            new_tokens.append(t)
        else:
            entity_pos += 1
    g['tokens'] = new_tokens
    return g


def normalize_tokens(g):
    """
    Normalize a tokens of the graph by setting it to lower case and removing any numbers.

    :param g: graph to normalize
    :return: graph with normalized tokens
    >>> normalize_tokens({'tokens':["Upper", "Case"]})
    {'tokens': ['upper', 'case']}
    >>> normalize_tokens({'tokens':["He", "started", "in", "1995"]})
    {'tokens': ['he', 'started', 'in', '0']}
    """
    tokens = g.get('tokens', [])
    g['tokens'] = [re.sub(r"\d+", "0", t.lower()) for t in tokens]
    return g


def get_graph_first_edge(g):
    """
    Get the first edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :return: an edge as a dictionary
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}], 'entities': []}) == {'right':[4,5,6]}
    True
    >>> get_graph_first_edge({})
    {}
    >>> get_graph_first_edge({'edgeSet':[]})
    {}
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}, {'right':[8]}], 'entities': []}) == {'right':[4,5,6]}
    True
    """
    return g["edgeSet"][0] if 'edgeSet' in g and g["edgeSet"] else {}


def get_graph_last_edge(g):
    """
    Get the last edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :return: an edge as a dictionary
    """
    return g["edgeSet"][-1] if 'edgeSet' in g and g["edgeSet"] else {}


def copy_graph(g):
    """
    Create a copy of the given graph.

    :param g: input graph as dictionary
    :return: a copy of the graph
    >>> copy_graph({'edgeSet': [{'right':[4,5,6]}], 'entities': [], 'tokens':[]}) == {'edgeSet': [{'right':[4,5,6]}], 'entities': [], 'tokens':[]}
    True
    >>> copy_graph({}) == {'edgeSet':[], 'entities':[]}
    True
    """
    new_g = {'edgeSet': copy.deepcopy(g.get('edgeSet', [])),
             'entities': copy.copy(g.get('entities', []))}
    if 'tokens' in g:
        new_g['tokens'] = g.get('tokens', [])
    if 'filter' in g:
        new_g['filter'] = g['filter']
    return new_g

np_grammar = r"""
    NP:
    {(<NN|NNS>+|<NNP|NNPS>+)<IN|CC>(<PRP\$|DT><NN|NNS>+|<NNP|NNPS>+)}
    {<JJ|RB|CD>*(<NNS|NN>+|<NNP|NNPS>+)<NNP|NN|NNS|NNPS>*}
    {<NNP|NN|NNS|NNPS>+}
    """
np_parser = nltk.RegexpParser(np_grammar)

manual_entities = set(utils.load_entity_map(utils.RESOURCES_FOLDER + "manual_entity_map.tsv").keys())


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
    [(['Missouri'], 'LOCATION'), (['current', 'senators'], 'NN')]
    >>> extract_entities([('what', 'O', 'WDT'), ('awards', 'O', 'NNS'), ('has', 'O', 'VBZ'), ('louis', 'PERSON', 'NNP'), ('sachar', 'PERSON', 'NNP'), ('won', 'O', 'NNP'), ('?', 'O', '.')])
    [(['Louis', 'Sachar'], 'PERSON'), (['Won'], 'NNP'), (['awards'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('president', 'O', 'NN'), ('after', 'O', 'IN'), ('jfk', 'O', 'NNP'), ('died', 'O', 'VBD'), ('?', 'O', '.')])
    [(['president', 'after', 'jfk'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('natalie', 'PERSON', 'NN'), ('likes', 'O', 'VBP')])
    [(['Natalie'], 'PERSON')]
    >>> extract_entities([('what', 'O', 'WDT'), ('character', 'O', 'NN'), ('did', 'O', 'VBD'), ('john', 'O', 'NNP'), \
    ('noble', 'O', 'NNP'), ('play', 'O', 'VB'), ('in', 'O', 'IN'), ('lord', 'O', 'NNP'), ('of', 'O', 'IN'), ('the', 'O', 'DT'), ('rings', 'O', 'NNS'), ('?', 'O', '.')])
    [(['John', 'Noble'], 'NNP'), (['character'], 'NN'), (['lord', 'of', 'the', 'rings'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['plays', 'O', 'VBZ'], ['lois', 'PERSON', 'NNP'], ['lane', 'PERSON', 'NNP'], ['in', 'O', 'IN'], ['superman', 'O', 'NNP'], ['returns', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Lois', 'Lane'], 'PERSON'), (['superman', 'returns'], 'NN')]
    >>> extract_entities([('the', 'O', 'DT'), ('empire', 'O', 'NN'), ('strikes', 'O', 'VBZ'), ('back', 'O', 'RB'), ('is', 'O', 'VBZ'), ('the', 'O', 'DT'), ('second', 'O', 'JJ'), ('movie', 'O', 'NN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('star', 'O', 'NN'), ('wars', 'O', 'NNS'), ('franchise', 'O', 'VBP')])
    [(['empire'], 'NN'), (['movie', 'in', 'the', 'star', 'wars'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['played', 'O', 'VBD'], ['cruella', 'LOCATION', 'NNP'], ['deville', 'LOCATION', 'NNP'], ['in', 'O', 'IN'], ['102', 'O', 'CD'], ['dalmatians', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Cruella', 'Deville'], 'LOCATION'), (['102', 'dalmatians'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['was', 'O', 'VBD'], ['the', 'O', 'DT'], ['winner', 'O', 'NN'], ['of', 'O', 'IN'], ['the', 'O', 'DT'], ['2009', 'O', 'CD'], ['nobel', 'O', 'NNP'], ['peace', 'O', 'NNP'], ['prize', 'O', 'NNP'], ['?', 'O', '.']])
    [(['2009', 'Nobel', 'Peace', 'Prize'], 'NNP'), (['winner'], 'NN'), (['2009'], 'CD')]
    >>> extract_entities([['who', 'O', 'WP'], ['is', 'O', 'VBZ'], ['the', 'O', 'DT'], ['senator', 'O', 'NN'], ['of', 'O', 'IN'], ['connecticut', 'LOCATION', 'NNP'], ['2010', 'O', 'CD'], ['?', 'O', '.']])
    [(['Connecticut'], 'LOCATION'), (['senator'], 'NN'), (['2010'], 'CD')]
    >>> extract_entities([['Which', 'O', 'WDT'],['actors', 'O', 'NNS'],['play', 'O', 'VBP'],['in', 'O', 'IN'],['Big', 'O', 'JJ'],['Bang', 'O', 'NNP'],['Theory', 'O', 'NNP'],['?', 'O', '.']])
    [(['Big', 'Bang', 'Theory'], 'NNP'), (['actors'], 'NN')]
    """
    persons = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['PERSON'])
    locations = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['LOCATION'])
    orgs = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['ORGANIZATION'])

    chunks = np_parser.parse([(w, t if p == "O" else "O") for w, p, t in tokens_ne_pos])
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
    ne_vertices = [([w.title() for w in ne], pos) for ne, pos in ne_vertices]
    return ne_vertices + vertices


def construct_graphs(tokens, entities):
    """
    Deprecated
    """
    entity_powerset = itertools.chain.from_iterable(itertools.combinations(entities, n) for n in range(1, len(entities)+1))
    graphs = []
    for entity_set in entity_powerset:
        g = {'edgeSet': [], 'tokens': tokens}
        for entity in entity_set:
            g['edgeSet'].append({'right':entity})
        graphs.append(g)
    return graphs


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())


