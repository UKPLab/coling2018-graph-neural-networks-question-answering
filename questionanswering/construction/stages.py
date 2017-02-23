import copy

from construction import graph
from wikidata import entity_linking

argmax_markers = {"last", "latest", "currency", "money", "football", "team"}
argmin_markers = {"first", "oldest"}
argmax_time_markers = {"president", "2012"}
argmin_time_markers = {"first", "oldest"}


def last_relation_hop(g):
    """
    Takes a graph with an existing relation and an intermediate variable by performing a hop-up for the second entity.

    :param g: a graph with an non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_hop({'edgeSet': [], 'entities': [[4, 5, 6]]})
    []
    >>> last_relation_hop({'edgeSet': [{'right':[4,5,6]}], 'entities': []}) == [{'edgeSet': [{'right':[4,5,6], 'hopUp': None}], 'entities': []}, {'edgeSet': [{'right':[4,5,6], 'hopDown': None}], 'entities': []}]
    True
    >>> last_relation_hop({'edgeSet': [{'right':[4,5,6], 'hopUp': None}], 'entities': []})
    []
    >>> last_relation_hop({'edgeSet': [{'right':["Bahama"], "rightkbID":"Q6754"}], 'entities': []}) == [{'edgeSet': [{'right':["Bahama"], "rightkbID":"Q6754", 'hopUp': None}], 'entities': []}, {'edgeSet': [{'right':["Bahama"], "rightkbID":"Q6754", 'hopDown': None}], 'entities': []}]
    True
    >>> last_relation_hop({'edgeSet': [{'right':[4,5,6], 'argmax':'time'}], 'entities': []})
    []
    >>> last_relation_hop({'edgeSet': [{'right':[4,5,6], 'num':['2012']}], 'entities': []})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or any(hop in g['edgeSet'][-1] for hop in {'hopUp', 'hopDown'}) or graph.graph_has_temporal(g):
        return []
    new_graphs = []
    for hop in HOP_TYPES:
        new_g = graph.copy_graph(g)
        new_g['edgeSet'][-1][hop] = None
        new_graphs.append(new_g)
    return new_graphs


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
    >>> add_entity_and_relation({'edgeSet': [], 'entities': [(['2012'], 'CD'), (["Natalie", "Portman"], 'PERSON')]}) == [{'edgeSet': [{'right': ['Natalie', 'Portman'], 'rightkbID': 'Q37876'}], 'entities': [(['2012'], 'CD')]}]
    True
    >>> add_entity_and_relation({'edgeSet': [], 'entities': [(['2012'], 'CD')]})
    []
    """
    if len(g.get('entities', [])) == 0:
        return []
    entities = copy.copy(g.get('entities', []))
    skipped = []
    new_graphs = []
    while entities:
        entity = entities.pop(0)
        if entity[1] == 'CD':
            skipped.append(entity)
        else:
            if len(entity.get("linkings",[])) > 0:
                linkings = entity['linkings']
                for kbID, label in linkings:
                    new_g = graph.copy_graph(g)
                    new_g['entities'] = entities[:] + skipped
                    new_g['edgeSet'].append({'right': entity[0], 'rightkbID': kbID, 'canonical_right': label})
                    new_graphs.append(new_g)
    return new_graphs


def last_relation_numeric(g):
    """
    Adds a numeric restriction to the last relation in the graph.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_numeric({'edgeSet': [{'right':[2]}, {'right':[8]}], 'entities': [(["Natalie", "Portman"], 'PERSON'), (['2012'], 'CD')]}) == \
    [{'edgeSet': [{'right':[2]}, {'right':[8], 'num': ['2012']}], 'entities': [(["Natalie", "Portman"], 'PERSON')]}]
    True
    >>> last_relation_numeric({'edgeSet': [{'right':[2]}, {'right':[8], 'argmin':'time'}], 'entities': [(['2012'], 'CD')]})
    []
    >>> last_relation_numeric({'edgeSet': [{'right':[2]}, {'right':[8], 'num':'2009'}], 'entities': [(['2012'], 'CD')]})
    []
    >>> last_relation_numeric({'edgeSet': [{'right':[2]}], 'entities': []})
    []
    >>> last_relation_numeric({'edgeSet': [{'right':[2]}], 'entities': [(["Natalie", "Portman"], 'PERSON')]})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or graph.graph_has_temporal(g):
        return []
    if len(g.get('entities', [])) == 0 or not any(e[1] == 'CD' for e in g['entities'] if len(e) > 1):
        return []
    entities = copy.copy(g.get('entities', []))
    cd_entities = [e[0] for e in entities if e[1] == 'CD']
    if len(cd_entities) == 0:
        return []
    cd_entity = cd_entities[0]
    entities = [e for e in entities if e[0] != cd_entity]
    new_g = graph.copy_graph(g)
    new_g['entities'] = entities
    new_g['edgeSet'][-1]['num'] = cd_entity
    return [new_g]


def last_relation_temporal(g):
    """
    Adds a temporal argmax to the last relation in the graph, that is only the latest/earliest entity is returned as the answer.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'right':[8]}], 'entities': [], 'tokens':['what','currency']}) == \
    [{'edgeSet': [{'right':[2]}, {'right':[8], 'argmax': 'time'}], 'entities': [], 'tokens':['what','currency']}]
    True
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'right':[8], 'argmin':'time'}], 'entities': []})
    []
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'right':[8], 'num':['2009']}], 'entities': []})
    []
    >>> last_relation_temporal({'edgeSet': [{'right':[2]}, {'type':'time'}], 'entities': []})
    []
    >>> add_temporal_relation({'edgeSet': [{'kbID': 'P161v','type': 'direct'}],'tokens': ['where','was','<e>','assassinated','?']})
    []
    >>> add_temporal_relation({'edgeSet': [{'kbID': 'P161v','type': 'direct'}],'tokens': ['where','was', 'first', '<e>','?']}) == \
    [{'entities': [], 'edgeSet': [{'kbID': 'P161v', 'type': 'direct'}, {'argmin': 'time', 'type': 'time'}], 'tokens': ['where', 'was', 'first', '<e>', '?']}]
    True
    """
    if len(g.get('edgeSet', [])) == 0 or graph.graph_has_temporal(g):
        return []
    new_graphs = []
    consider_types = set()
    if any(t in argmax_markers for t in g.get('tokens',[])):
        consider_types.add('argmax')
    if any(t in argmin_markers for t in g.get('tokens',[])):
        consider_types.add('argmin')
    for t in consider_types.intersection(ARG_TYPES):
        new_g = graph.copy_graph(g)
        new_g['edgeSet'][-1][t] = "time"
        new_graphs.append(new_g)
    return new_graphs


def add_temporal_relation(g):
    """
    Adds a temporal argmax relation in the graph, that is only the latest/earliest entity is returned as the answer.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'right':[8]}], 'entities': [], 'tokens':['who', 'president']}) == \
     [{'edgeSet': [{'right':[2]}, {'right':[8]}, {'type':'time', 'argmax':'time'}], 'entities': [], 'tokens':['who', 'president']}]
    True
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'right':[8], 'argmin':'time'}], 'entities': []})
    []
    >>> add_temporal_relation({'edgeSet': [{'right':[2]}, {'type':'time'}], 'entities': []})
    []
    >>> add_temporal_relation({'edgeSet': [{'kbID': 'P161v','type': 'direct'}],'tokens': ['where','was','<e>','assassinated','?']})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or graph.graph_has_temporal(g):
        return []
    new_graphs = []
    consider_types = set()
    if any(t in argmax_time_markers for t in g.get('tokens',[])):
        consider_types.add('argmax')
    if any(t in argmin_time_markers for t in g.get('tokens',[])):
        consider_types.add('argmin')
    for t in consider_types.intersection(ARG_TYPES):
        new_g = graph.copy_graph(g)
        new_edge = {'type': 'time', t: 'time'}
        new_g['edgeSet'].append(new_edge)
        new_graphs.append(new_g)
    return new_graphs

# This division of actions is relevant for grounding with gold answers:
# - Restrict action limit the set of answers and should be applied
#   to a graph that has groundings
RESTRICT_ACTIONS = [add_entity_and_relation, last_relation_temporal, add_temporal_relation, last_relation_numeric]
# - Expand actions change graph to extract another set of answers and should be
#   applied to a graph that has empty denotation
EXPAND_ACTIONS = [last_relation_hop]  # Expand actions

# This division is relevant for constructing all possible groundings without gold answers:
# - WikiData actions need to be grounded in Wikidata in order to construct the next graph
WIKIDATA_ACTIONS = {add_entity_and_relation, last_relation_hop}
# - Non linking options just add options to the graph structure without checking if it is possible in WikiData.
#   Hop-up is always possible anyway, temporal is possible most of the time.
NON_LINKING_ACTIONS = {last_relation_temporal, add_temporal_relation, last_relation_numeric}

ARG_TYPES = {'argmax', 'argmin'}
HOP_TYPES = {'hopUp', 'hopDown'}


def expand(g):
    """
    Expand the coverage of the given graph by constructing version that has more possible/other denotations.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[["Barack", "Obama"]]})
    []
    >>> expand({"edgeSet":[{"right":["Barack", "Obama"]}]}) == [{'edgeSet': [{'hopUp': None, 'right': ['Barack', 'Obama']}], 'entities': []}, {'edgeSet': [{'hopDown': None, 'right': ['Barack', 'Obama']}], 'entities': []}]
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
