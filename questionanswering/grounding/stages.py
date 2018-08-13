from copy import copy

from questionanswering.construction.sentence import Sentence
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering.grounding.graph_queries import QUESTION_VAR, LONG_LEG_RELATIONS

DENOTATION_CLASS_EDGE = Edge(leftentityid=QUESTION_VAR, relationid='iclass')

arg_relations = {"MIN": ["P582", "P585", "P577"],
                 "MAX": ["P580", "P585", "P577"]}
year_relations = {"P585q", "P580q", "P582q"}
argmax_markers = {"last", "latest"}
argmin_markers = {"first", "oldest"}


def with_denotation_class_edge(g: SemanticGraph):
    """
    Adds an implicit graph otherwise a copy of the original graph is returned.
    
    :param g: a graph with a non-empty 'entities' list
    :return: a list of suggested graphs
    >>> with_denotation_class_edge(SemanticGraph())
    [SemanticGraph([Edge(0, ?qvar-iclass->None)])]
import grounding.graph_queries    >>> add_denotation_class_edge(SemanticGraph([Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, relationid='iclass')]))
    [SemanticGraph([Edge(0, ?qvar-iclass->None)])]
    """
    if any(edge.relationid == 'iclass' for edge in g.edges):
        return g
    new_g = copy(g)
    new_g.edges.append(copy(DENOTATION_CLASS_EDGE))
    return new_g


def add_entity_and_relation(g: SemanticGraph, leg_length=1, fixed_relations=None):
    """
    Takes a graph with a non-empty list of free entities and adds a new relations with the one of the free entities,
    thus removing it from the list.

    :param g: a graph with a non-empty 'entities' list
    :return: a list of suggested graphs
    >>> add_entity_and_relation(SemanticGraph())
    []
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{'linkings':[("Q37876", "Natalie Portman"), ("Q872356", "Portman")], 'tokens':["Portman"], 'type':'PERSON'}]))
    [SemanticGraph([Edge(0, ?qvar-None->Q37876)], 0), SemanticGraph([Edge(0, Q37876-None->?qvar)], 0), SemanticGraph([Edge(0, ?qvar-None->Q872356)], 0), SemanticGraph([Edge(0, Q872356-None->?qvar)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{'linkings':[("Q6", "city")], 'tokens':["city"], 'type':'NN'}, {'linkings':[("Q37876", "Natalie Portman")], 'tokens':["Portman"], 'type':'PERSON'}]))
    [SemanticGraph([Edge(0, ?qvar-None->Q37876)], 1), SemanticGraph([Edge(0, Q37876-None->?qvar)], 1), SemanticGraph([Edge(0, ?qvar-class->Q6)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{"linkings": [("2012", ["2012"])] ,"tokens": ['2012'], "type": 'YEAR'}]), leg_length=2)
    [SemanticGraph([Edge(0, ?qvar-None->?m02012), Edge(1, ?m02012-None->2012)], 0), SemanticGraph([Edge(0, ?m02012-None->2012), Edge(1, ?m02012-None->?qvar)], 0)]
    >>> add_entity_and_relation(SemanticGraph(edges=[Edge(leftentityid=QUESTION_VAR, rightentityid='Q37876')], free_entities=[{'linkings':[("Q6", "city")], 'tokens':["city"], 'type':'NN'}]))
    [SemanticGraph([Edge(0, ?qvar-None->Q37876), Edge(1, ?qvar-class->Q6)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{"type": "NNP", "linkings": [("Q1079", "Breaking Bad")]}], tokens=["Who", "played", "Gus", "Fring", "on", "Breaking", "Bad", "?"]))
    [SemanticGraph([Edge(0, ?qvar-None->Q1079)], 0), SemanticGraph([Edge(0, Q1079-None->?qvar)], 0), SemanticGraph([Edge(0, None-None->?qvar~None->Q1079)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{"type": "NNP", "linkings": [("Q1079", "Breaking Bad")]}], tokens=["Who", "played", "Gus", "Fring", "on", "Breaking", "Bad", "?"]), leg_length=2)  # doctest: +ELLIPSIS
    [... SemanticGraph([Edge(0, None-None->?qvar~None->?m0Q1079), Edge(1, ?m0Q1079-None->Q1079)], 0), SemanticGraph([Edge(0, None-None->?qvar~None->?m0Q1079), Edge(1, Q1079-None->?m0Q1079)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{'linkings':[("Q76", "Obama")], 'type':'PERSON'}]), leg_length=2)  # doctest: +ELLIPSIS
    [... SemanticGraph([Edge(0, Q76-None->?m0Q76), Edge(1, ?m0Q76-None->?qvar)], 0), SemanticGraph([Edge(0, ?m0Q76-None->Q76), Edge(1, ?m0Q76-None->?qvar)], 0)]
    >>> add_entity_and_relation(Sentence(input_text="where is London ?").graphs[0].graph)
    [SemanticGraph([Edge(0, ?qvar-class->Q618123)], 0)]
    >>> add_entity_and_relation(SemanticGraph(free_entities=[{'linkings':[("Q76", "Obama")], 'type':'PERSON'}]), leg_length=2, fixed_relations=['P31', 'P27'])  # doctest: +ELLIPSIS
    [SemanticGraph([Edge(0, ?qvar-None->?m0Q76), Edge(1, ?m0Q76-P31->Q76)], 0), SemanticGraph([Edge(0, ?qvar-None->?m0Q76), Edge(1, Q76-P31->?m0Q76)], 0),...]
    """
    if len(g.free_entities) == 0:
        return []
    new_graphs = []
    # Put the common nouns to the end
    entities_to_consider = [e for e in g.free_entities if e.get("type") not in {'NN'}] \
                           + [e for e in g.free_entities if e.get("type") in {'NN'}]
    while entities_to_consider:
        entity = entities_to_consider.pop(0)
        if len(entity.get("linkings", [])) > 0:
            linkings = [l for l in entity['linkings'] if l[0]]
            for kbID, label in linkings:
                new_legs = []
                if entity.get("type") == 'NN':
                    if len(g.edges) > 0:
                        new_legs = [(Edge(leftentityid=QUESTION_VAR, relationid='class', rightentityid=kbID),)]
                else:
                    if fixed_relations:
                        for r in fixed_relations:
                            new_legs += [(Edge(leftentityid=QUESTION_VAR, relationid=r, rightentityid=kbID),),
                                         (Edge(rightentityid=QUESTION_VAR, relationid=r, leftentityid=kbID),)]
                    else:
                        new_legs = [(Edge(leftentityid=QUESTION_VAR, rightentityid=kbID),),
                                    (Edge(rightentityid=QUESTION_VAR, leftentityid=kbID),)]
                    if any(t.lower().startswith("play") or t.lower().startswith("voice") for t in g.tokens):
                        new_legs.append((Edge(rightentityid=QUESTION_VAR, qualifierentityid=kbID),))
                    if leg_length > 1 and entity.get("type") == 'YEAR':
                        new_legs = []
                    else:
                        for leg_len in range(leg_length - 1):
                            next_leg_extension = []
                            for leg in new_legs:
                                intermediate_node = f"?m{leg_len}{kbID}"
                                last_edge = leg[-1]

                                head_to_tail = (Edge(leftentityid=last_edge.leftentityid,
                                                     rightentityid=intermediate_node if last_edge.leftentityid else last_edge.rightentityid,
                                                     qualifierentityid=intermediate_node if last_edge.qualifierentityid else None),
                                                Edge(leftentityid=intermediate_node,
                                                     rightentityid=last_edge.rightentityid if last_edge.leftentityid else last_edge.qualifierentityid))
                                head_to_head = (copy(head_to_tail[0]), copy(head_to_tail[1]))
                                if QUESTION_VAR in head_to_head[0].nodes():
                                    head_to_head[1].invert()
                                    head_to_head[1].relationid = last_edge.relationid
                                    head_to_tail[1].relationid = last_edge.relationid
                                else:
                                    head_to_head[0].invert()
                                    head_to_head[0].relationid = last_edge.relationid
                                    head_to_tail[0].relationid = last_edge.relationid
                                next_leg_extension.extend([head_to_tail, head_to_head])
                            new_legs = next_leg_extension
                new_legs = [leg for leg in new_legs if not any(e.leftentityid is not None and e.leftentityid.isdigit() for e in leg)]
                for leg in new_legs:
                    new_g = copy(g)
                    new_g.free_entities = entities_to_consider[:]
                    new_g.edges.extend(leg)
                    new_graphs.append(new_g)
    return new_graphs


def last_edge_numeric_constraint(g: SemanticGraph):
    """
    Adds a numeric restriction to the last added edge in the graph.

    :param g: a graph with a non-empty list of edges
    :return: a list of suggested graphs
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76")], free_entities=[{'linkings':[("Q37876", "Natalie Portman")], 'tokens':["Portman"], 'type':'PERSON'}, {'linkings': [('2012', '2012')], 'type': 'YEAR', 'tokens': ['2012']}]))
    [SemanticGraph([Edge(0, ?qvar-None->Q76~None->2012)], 1)]
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid='2009')], free_entities=[{'linkings': [('2012', '2012')], 'type': 'YEAR', 'tokens': ['2012']}]))
    []
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid='2009')]))
    []
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid='2009')], free_entities=[{'linkings':[("Q37876", "Natalie Portman")], 'tokens':["Portman"], 'type':'PERSON'}]))
    []
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76")]))
    [SemanticGraph([Edge(0, ?qvar-None->Q76~None->MIN)], 0), SemanticGraph([Edge(0, ?qvar-None->Q76~None->MAX)], 0)]
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid='MIN')]))
    []
    >>> last_edge_numeric_constraint(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid='MIN'), Edge(leftentityid=QUESTION_VAR, rightentityid="Q5")]))
    []
    """
    if len(g.edges) == 0 or g.edges[-1].qualifierentityid is not None:
        return []
    new_graphs = []
    entities_to_consider = [e for e in g.free_entities if e.get("type") == 'YEAR']
    skipped = [e for e in g.free_entities if e.get("type") != 'YEAR']
    if not entities_to_consider and not any(e.temporal for e in g.edges) and QUESTION_VAR in g.edges[-1].nodes():
        add_args = []
        sorting = 'MIN'
        if len(argmin_markers & set(g.tokens)) > 0:
            add_args = arg_relations['MIN']
            sorting = 'MIN'
        elif len(argmax_markers & set(g.tokens)) > 0:
            add_args = arg_relations['MAX']
            sorting = 'MAX'
        for rel in add_args:
            new_g = copy(g)
            new_g.free_entities = skipped[:]
            edge_to_modify = new_g.edges[-1]
            edge_to_modify.qualifierentityid = sorting
            edge_to_modify.qualifierrelationid = rel
            new_graphs.append(new_g)
    while entities_to_consider:
        entity = entities_to_consider.pop(0)
        if entity.get('linkings'):
            for rel in year_relations:
                new_g = copy(g)
                new_g.free_entities = skipped[:] + entities_to_consider[:]
                edge_to_modify = new_g.edges[-1]
                edge_to_modify.qualifierentityid = entity['linkings'][0][0]
                edge_to_modify.qualifierrelationid = rel
                new_graphs.append(new_g)
    return new_graphs


def add_relation(g: SemanticGraph):
    """
    Adds a new constraint edge to the graph.

    :param g: a semantic graph with a non-empty edge list
    :return: a list of new graphs
    >>> add_relation(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76")], tokens=['first']))
    [SemanticGraph([Edge(0, ?qvar-None->Q76), Edge(1, ?qvar-None->MIN)], 0), SemanticGraph([Edge(0, ?qvar-None->Q76), Edge(1, ?qvar-None->MAX)], 0)]
    >>> add_relation(SemanticGraph())
    []
    >>> add_relation(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76", qualifierentityid="MAX")]))
    []
    """
    new_graphs = []
    if len(g.edges) > 0 and not any(e.temporal for e in g.edges):
        add_args = []
        sorting = 'MIN'
        if len(argmin_markers & set(g.tokens)) > 0:
            add_args = arg_relations['MIN']
            sorting = 'MIN'
        elif len(argmax_markers & set(g.tokens)) > 0:
            add_args = arg_relations['MAX']
            sorting = 'MAX'
        for rel in add_args:
            new_g = copy(g)
            new_g.edges.append(Edge(leftentityid=QUESTION_VAR, rightentityid=sorting, relationid=rel))
            new_graphs.append(new_g)
    return new_graphs


# This division of actions is relevant for grounding with gold answers:
# - Restrict action     limit the set of answers and should be applied
#   to a graph that has groundings
ACTIONS = [last_edge_numeric_constraint,
           lambda x: add_entity_and_relation(x, leg_length=1),
           add_relation,
           lambda x: add_entity_and_relation(x, leg_length=2, fixed_relations=LONG_LEG_RELATIONS),
           lambda x: add_entity_and_relation(x, leg_length=2)]


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
