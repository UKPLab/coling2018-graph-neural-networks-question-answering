import copy

from construction import graph
from wikidata import entity_linking


def last_relation_subentities(g):
    """
    Takes a graph with an existing relation and suggests a set of graphs with the same relation but one of the entities
     is a sub-span of the original entity.

    :param g: a graph with an non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_subentities({'edgeSet': [], 'entities': [(['grand', 'bahama', 'island'], 'LOCATION')], 'tokens': ['what', 'country', 'is', 'the', 'grand', 'bahama', 'island', 'in', '?']})
    []
    >>> len(last_relation_subentities({'edgeSet': [{'right': (['grand', 'bahama', 'island'], 'LOCATION')}], 'entities': [], 'tokens': ['what', 'country', 'is', 'the', 'grand', 'bahama', 'island', 'in', '?']}))
    5
    >>> last_relation_subentities({'edgeSet': [{'right':(['Jfk'], 'NNP')}], 'entities': []}) == [{'edgeSet': [{'right': ['JFK']}], 'entities': []}]
    True
    """
    if len(g.get('edgeSet', [])) == 0 or len(g['edgeSet'][-1]['right']) < 1:
        return []
    new_graphs = []
    right_entity = g['edgeSet'][-1]['right']
    for new_entity in entity_linking.possible_subentities(*right_entity):
        new_g = graph.copy_graph(g)
        new_g['edgeSet'][-1]['right'] = list(new_entity)
        new_graphs.append(new_g)
    return new_graphs


def last_relation_hop_up(g):
    """
    Takes a graph with an existing relation and an intermediate variable by performing a hop-up for the second entity.

    :param g: a graph with an non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_hop_up({'edgeSet': [], 'entities': [[4, 5, 6]]})
    []
    >>> last_relation_hop_up({'edgeSet': [{'right':[4,5,6]}], 'entities': []}) == [{'edgeSet': [{'right':[4,5,6], 'hopUp': None}], 'entities': []}]
    True
    >>> last_relation_hop_up({'edgeSet': [{'right':[4,5,6], 'hopUp': None}], 'entities': []})
    []
    >>> last_relation_hop_up({'edgeSet': [{'right':["Bahama"], "rightkbID":"Q6754"}], 'entities': []}) == [{'edgeSet': [{'right':["Bahama"], "rightkbID":"Q6754", 'hopUp': None}], 'entities': []}]
    True
    """
    if len(g.get('edgeSet', [])) == 0 or 'hopUp' in g['edgeSet'][-1] or g['edgeSet'][-1].get('type') in {'time'}:
        return []
    new_g = graph.copy_graph(g)
    new_g['edgeSet'][-1]['hopUp'] = None
    return [new_g]


def add_entity_and_relation(g):
    """
    Takes a graph with a non-empty list of free entities and adds a new relations with the one of the free entities, thus
    removing it from the list.

    :param g: a graph with a non-empty 'entities' list
    :return: a list of suggested graphs
    >>> add_entity_and_relation({'edgeSet': [], 'entities': []})
    []
    >>> add_entity_and_relation({'edgeSet': [], 'entities': [(["Natalie", "Portman"], 'PERSON')]}) == [{'edgeSet': [{'right': ['Natalie', 'Portman'], 'rightkbID': 'Q37876'}], 'entities': []}]
    True
    """
    if len(g.get('entities', [])) == 0:
        return []
    entities = copy.copy(g.get('entities', []))
    new_graphs = []
    while entities:
        entity = entities.pop(0)
        linkings = entity_linking.link_entity(entity)
        for linking in linkings:
            new_g = graph.copy_graph(g)
            new_g['entities'] = entities[:]
            new_edge = {'right': entity[0], 'rightkbID': linking}
            new_g['edgeSet'].append(new_edge)
            new_graphs.append(new_g)

    return new_graphs


def last_relation_temporal(g):
    """
    Adds a temporal argmax to the last relation in the graph, that is only the latest/earliest entity is returned as the answer.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'right':[8]}], 'entities': []}) == [{'edgeSet': [{'right':[2]}, {'right':[8], 'argmax': 'time'}], 'entities': []}, {'edgeSet': [{'right':[2]}, {'right':[8], 'argmin': 'time'}], 'entities': []}]
    True
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'right':[8], 'argmin':'time'}], 'entities': []})
    []
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'type':'time'}], 'entities': []})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or graph.graph_has_temporal(g):
        return []
    new_graphs = []
    for t in ARG_TYPES:
        new_g = graph.copy_graph(g)
        new_g['edgeSet'][-1][t] = "time"
        new_graphs.append(new_g)
    return new_graphs


def add_temporal_relation(g):
    """
    Adds a temporal argmax relation in the graph, that is only the latest/earliest entity is returned as the answer.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'right':[8]}], 'entities': []}) == \
     [{'edgeSet': [{'right':[2]}, {'right':[8]}, {'type':'time', 'argmax':'time'}], 'entities': []}, {'edgeSet': [{'right':[2]}, {'right':[8]}, {'type':'time', 'argmin':'time'}], 'entities': []}]
    True
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'right':[8], 'argmin':'time'}], 'entities': []})
    []
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'type':'time'}], 'entities': []})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or graph.graph_has_temporal(g):
        return []
    new_graphs = []
    for t in ARG_TYPES:
        new_g = graph.copy_graph(g)
        new_edge = {'type': 'time', t: 'time'}
        new_g['edgeSet'].append(new_edge)
        new_graphs.append(new_g)
    return new_graphs

# This division of actions is relevant for grounding with gold answers:
# - Restrict action limit the set of answers and should be applied
#   to a graph that has groundings
RESTRICT_ACTIONS = {add_entity_and_relation, last_relation_temporal, add_temporal_relation}
# - Expand actions change graph to extract another set of answers and should be
#   applied to a graph that has empty denotation
EXPAND_ACTIONS = {last_relation_hop_up}  # Expand actions

# This division is relevant for constructing all possible groundings without gold answers:
# - WikiData actions need to be grounded in Wikidata in order to construct the next graph
WIKIDATA_ACTIONS = {add_entity_and_relation, last_relation_hop_up}
# - Non linking options just add options to the graph structure without checking if it is possible in WikiData.
#   Hop-up is always possible anyway, temporal is possible most of the time.
NON_LINKING_ACTIONS = {last_relation_temporal, add_temporal_relation}

ARG_TYPES = ['argmax', 'argmin']


def expand(g):
    """
    Expand the coverage of the given graph by constructing version that has more possible/other denotations.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[["Barack", "Obama"]]})
    []
    >>> expand({"edgeSet":[{"right":["Barack", "Obama"]}]}) == [{'edgeSet': [{'hopUp': None, 'right': ['Barack', 'Obama']}], 'entities': []}]
    True
    """
    if "edgeSet" not in g:
        return []
    available_expansions = EXPAND_ACTIONS
    return_graphs = [el for f in available_expansions for el in f(g)]
    return return_graphs


def restrict(g):
    """
    Restrict the set of possible graph denotations by adding new constraints that should be fullfilled by the linking.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[(['Barack', 'Obama'], "PERSON")]}) == [{'entities': [], 'edgeSet': [{'right': ['Barack', 'Obama'], 'rightkbID': 'Q76'}], 'tokens': ['Who', 'is', 'Barack', 'Obama', '?']}]
    True
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "edgeSet":[{"left":[0], "right":[2,3]}]})
    []
    """
    if "entities" not in g:
        return []
    available_restrictions = RESTRICT_ACTIONS
    return_graphs = [el for f in available_restrictions for el in f(g)]
    return return_graphs


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())