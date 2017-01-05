import itertools
import nltk
import copy
import re


def replace_first_entity(g):
    """
    Replaces an entity participating in the first relation in the graph with <e>

    :param g: graph as a dictionary
    :return: the original graph modified
    >>> replace_first_entity({'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', 'nfl', 'redskins', 'from', '?']}) == {'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', '<e>', 'from', '?']}
    True
    """
    tokens = g.get('tokens', [])
    edge = get_graph_first_edge(g)
    entity = {t.lower() for t in edge.get('right', [])}
    new_tokens = []
    previous_is_entity = False
    for i, t in enumerate(tokens):
        if t not in entity:
            if previous_is_entity:
                new_tokens.append("<e>")
                previous_is_entity = False
            new_tokens.append(t)
        else:
            previous_is_entity = True
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
    >>> get_graph_first_edge({'edgeSet': [{'left':[0], 'right':[4,5,6]}], 'entities': []}) == {'left':[0], 'right':[4,5,6]}
    True
    >>> get_graph_first_edge({})
    {}
    >>> get_graph_first_edge({'edgeSet':[]})
    {}
    >>> get_graph_first_edge({'edgeSet': [{'left':[0], 'right':[4,5,6]}, {'left':[0], 'right':[8]}], 'entities': []}) == {'left':[0], 'right':[4,5,6]}
    True
    """
    return g["edgeSet"][0] if 'edgeSet' in g and g["edgeSet"] else {}


def copy_graph(g):
    """
    Create a copy of the given graph.

    :param g: input graph as dictionary
    :return: a copy of the graph
    >>> copy_graph({'edgeSet': [{'left':[0], 'right':[4,5,6]}], 'entities': []}) == {'edgeSet': [{'left':[0], 'right':[4,5,6]}], 'entities': [], 'tokens':[]}
    True
    >>> copy_graph({}) == {'tokens':[], 'edgeSet':[], 'entities':[]}
    True
    """
    new_g = {'tokens': g.get('tokens', []),
             'edgeSet': copy.deepcopy(g.get('edgeSet', [])),
             'entities': copy.copy(g.get('entities', []))}
    return new_g


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

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def extract_entities(tokens_ne_pos):
    """
    Extract entities from the NE tags and POS tags of a sentence. Regular nouns are lemmatized to get rid of plurals.

    :param tokens_ne_pos: list of POS and NE tags.
    :return: list of entities in the order: NE>NNP>NN
    >>> extract_entities([('who', 'O', 'WP'), ('are', 'O', 'VBP'), ('the', 'O', 'DT'), ('current', 'O', 'JJ'), ('senators', 'O', 'NNS'), ('from', 'O', 'IN'), ('missouri', 'LOCATION', 'NNP'), ('?', 'O', '.')])
    [(['Missouri'], 'LOCATION'), (['senator'], 'NN')]
    >>> extract_entities([('what', 'O', 'WDT'), ('awards', 'O', 'NNS'), ('has', 'O', 'VBZ'), ('louis', 'PERSON', 'NNP'), ('sachar', 'PERSON', 'NNP'), ('won', 'O', 'NNP'), ('?', 'O', '.')])
    [(['Louis', 'Sachar'], 'PERSON'), (['award'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('president', 'O', 'NN'), ('after', 'O', 'IN'), ('jfk', 'O', 'NNP'), ('died', 'O', 'VBD'), ('?', 'O', '.')])
    [(['Jfk'], 'NNP'), (['president'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('natalie', 'PERSON', 'NN'), ('likes', 'O', 'VBP')])
    [(['Natalie'], 'PERSON')]
    """
    persons = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['PERSON'])
    locations = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['LOCATION'])
    orgs = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['ORGANIZATION'])

    nns = extract_entities_from_tagged([(w, t) for w, _, t in tokens_ne_pos], ['NN', 'NNS'])
    nnps = extract_entities_from_tagged([(w, t) for w, _, t in tokens_ne_pos], ['NNP', 'NNPS'])
    ne_vertices = [(ne, 'PERSON') for ne in persons] + [(ne, 'LOCATION') for ne in locations] + [(ne, 'ORGANIZATION') for ne in orgs]
    vertices = []
    for nn in nnps:
        if not any(n in v for n in nn for v, _ in vertices + ne_vertices):
            ne_vertices.append((nn, 'NNP'))
    for nn in nns:
        if not any(n in v for n in nn for v, _ in vertices + ne_vertices):
            nn = [lemmatizer.lemmatize(n) for n in nn]
            vertices.append((nn, 'NN'))
    ne_vertices = [([w.title() for w in ne], pos) for ne, pos in ne_vertices]
    return ne_vertices + vertices


def construct_graphs(tokens, entities):
    entity_powerset = itertools.chain.from_iterable(itertools.combinations(entities, n) for n in range(1, len(entities)+1))
    graphs = []
    for entity_set in entity_powerset:
        g = {'edgeSet': [], 'tokens': tokens}
        for entity in entity_set:
            g['edgeSet'].append({'left':[0], 'right':entity})
        graphs.append(g)
    return graphs


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())

