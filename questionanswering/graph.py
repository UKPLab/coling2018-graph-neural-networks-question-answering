import networkx as nx
import itertools


def extract_entities(annotated_tokens):
    """
    The method takes a list of tokens annotated with the Stanford NE annotation scheme and produces a list of entites.

    :param annotated_tokens: list of tupels where the first element is a token and the second is the annotation
    :return: list of entities each represented by the corresponding token ids

    Tests:
    >>> extract_entities([('what', 'O'), ('character', 'O'), ('did', 'O'), ('natalie', 'PERSON'), ('portman', 'PERSON'), ('play', 'O'), ('in', 'O'), ('star', 'O'), ('wars', 'O'), ('?', 'O')])
    [[3, 4]]

    >>> extract_entities([('Who', 'O'), ('was', 'O'), ('john', 'PERSON'), ('noble', 'PERSON')])
    [[2, 3]]

    """
    vertices = []
    current_vertex = []
    for i, (w, t) in enumerate(annotated_tokens):
        if t != 'O':
            current_vertex.append(i)
        elif len(current_vertex) > 0:
            vertices.append(current_vertex)
            current_vertex = []
    if len(current_vertex) > 0:
        vertices.append(current_vertex)
    return vertices


actions = ['add_entity']


def construct_graphs(tokens, entities):
    entity_powerset = itertools.chain.from_iterable(itertools.combinations(entities, n) for n in range(1, len(entities)+1))
    graphs = []
    for entity_set in entity_powerset:
        g = {'edgeSet': []}
        for entity in entity_set:
            g['edgeSet'].append(([0], entity))
        graphs.append(g)
    return graphs


class SGraph:
    """

    """

    def __init__(self):
        self._tokens = []
        self._vertices = []
        self._edges = []


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())

