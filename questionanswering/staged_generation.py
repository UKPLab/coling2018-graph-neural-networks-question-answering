import copy
from wikidata_access import *
from evaluation import *
from webquestions_io import *
import logging

logging.basicConfig(level=logging.DEBUG)


def get_available_expansions(g):
    """
    Get a list of methods that can be applied on the given graph to expand its denotation.

    :param g: a graph object
    :return: list of methods that take graph as an only argument
    >>> get_available_expansions({'edgeSet':[]})
    []
    >>> hop_up in get_available_expansions({'edgeSet':[{'left':[0], 'right':[2,3]}]})
    True
    """
    if len(g['edgeSet']) > 0 and 'hopUp' not in g['edgeSet'][-1]:
        return [hop_up]
    return []


def get_available_restrictions(g):
    """
    Get a list of methods that can be applied on the given graph to expand restrict denotation.

    :param g: a graph object
    :return: list of methods that take graph as an only argument
    >>> get_available_restrictions({'entities':[], 'edgeSet':[{},{}]})
    []
    >>> add_entity_and_relation in get_available_restrictions({'entities':[[2,3]]})
    True
    """
    if len(g['entities']) > 0:
        return [add_entity_and_relation]
    return []


def remove_token_from_entity(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': []}
    for edge in g.get('edgeSet', []):
        new_edge = copy.copy(edge)
        new_edge['right'] = new_edge['right'][:-1]
        new_g['edgeSet'].append(new_edge)
    return new_g


def hop_up(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': copy.deepcopy(g['edgeSet'])}
    if len(new_g['edgeSet']) > 0:
        new_g['edgeSet'][-1]['hopUp'] = 1
    return new_g


def add_entity_and_relation(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': copy.deepcopy(g.get('edgeSet', []))}
    entities_left = g['entities']

    if len(entities_left) > 0:
        entity = entities_left[0]
        new_edge = {'left': [0], 'right': entity}
        new_g['edgeSet'].append(new_edge)

    new_g['entities'] = entities_left[1:] if len(entities_left) > 1 else []
    return new_g


EXPAND_ACTIONS = [hop_up]
RESTRICT_ACTIONS = [add_entity_and_relation]


def expand(g):
    """
    Expand the coverage of the given graph by constructing version that has more possible/other denotations.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[[2, 3]]})
    []
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "edgeSet":[{"left":[0], "right":[2,3]}]}) == [{'tokens': ['Who', 'is', 'Barack', 'Obama', '?'], 'edgeSet': [{'left': [0], 'hopUp': 1, 'right': [2, 3]}]}]
    True
    """
    if "edgeSet" not in g:
        return []
    available_expansions = get_available_expansions(g)
    return_graphs = [f(g) for f in available_expansions]
    return return_graphs


def restrict(g):
    """
    Restrict the set of possible graph denotations by adding new constraints that should be fullfilled by the linking.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[[2, 3]]}) == [{'edgeSet': [{'left': [0], 'right': [2, 3]}], 'entities': [], 'tokens': ['Who', 'is', 'Barack', 'Obama', '?']}]
    True
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "edgeSet":[{"left":[0], "right":[2,3]}]})
    []
    """
    if "entities" not in g:
        return []
    available_restrictions = get_available_restrictions(g)
    return_graphs = [f(g) for f in available_restrictions]
    return return_graphs


def generate_with_gold(ungrounded_graph, question_obj):
    """
    Generate all possible groundings that produce positive f-score starting with the given ungrounded graph and
    using expand and restrict operations on its denotation.

    :param ungrounded_graph: the starting graph that should contain a list of tokens and a list of entities
    :param question_obj: a WebQuestions question encoded as a dictionary
    :return: a list of generated grounded graphs
    """
    pool = [ungrounded_graph]  # pool of possible parses
    generated_graphs = []

    while len(pool) > 0:
        g = pool.pop()
        suggested_graphs = restrict(g[0])
        chosen_graphs = ground_with_gold(question_obj, suggested_graphs)
        if len(chosen_graphs) == 0:
            suggested_graphs = [e_g for s_g in suggested_graphs for e_g in expand(s_g)]
            chosen_graphs = ground_with_gold(question_obj, suggested_graphs)
        if len(chosen_graphs) > 0:
            pool.extend(chosen_graphs)
        else:
            generated_graphs.append(g)

    return generated_graphs


def ground_with_gold(suggested_graphs, question_obj):
    """
    For each graph among the suggested_graphs find its groundings in the WikiData, then evaluate each suggested graph
    with each of its possible groundings and compare the denotations with the answers embedded in the question_obj.
    Return all groundings that produce an f-score > 0.0

    :param suggested_graphs: a list of ungrounded graphs
    :param question_obj: a WebQuestions question encoded as a dictionary
    :return: a list of graph groundings
    """
    suggested_graphs = [apply_grounding(p, s_g) for s_g in suggested_graphs for p in
                        query_wikidata(graph_to_query(s_g))]
    retrieved_answers = [query_wikidata(graph_to_query(s_g, return_var_values=True)) for s_g in suggested_graphs]
    retrieved_answers = [[r['e1'] for r in answer_set] for answer_set in retrieved_answers]
    retrieved_answers = [[e.lower() for a in answer_set for e in entity_map.get(a, [a]) ] for answer_set in retrieved_answers]

    evaluation_results = [retrieval_prec_rec_f1(get_answers_from_question(question_obj), retrieved_answers[i]) for i in
                          range(len(suggested_graphs))]
    chosen_graphs = [(suggested_graphs[i], evaluation_results[i], retrieved_answers[i])
                     for i in range(len(suggested_graphs)) if evaluation_results[i][2] > 0.0]
    return chosen_graphs


def apply_grounding(g, grounding):
    """
    Given a grounding obtained from WikiData apply it to the graph.
    Note: that the variable names returned by WikiData are important as they encode some grounding features.

    :param g: a single ungrounded graph
    :param grounding: a dictionary representing the grounding of relations and variables
    :return: a grounded graph
    >>> apply_grounding({'edgeSet':[{}]}, {'r0d':'P31v'}) == {'edgeSet': [{'type': 'direct', 'kbID': 'P31v'}]}
    True
    >>> apply_grounding({'edgeSet':[{}]}, {'r0v':'P31v'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v'}]}
    True
    >>> apply_grounding({'edgeSet':[{}, {}]}, {'r1d':'P39v', 'r0v':'P31v', 'e20': 'Q18'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'rightkbID': 'Q18'}, {'type': 'direct', 'kbID': 'P39v'}]}
    True
    >>> apply_grounding({'edgeSet':[]}, {})
    {'edgeSet': []}
    """
    grounded = copy.deepcopy(g)
    for i, edge in enumerate(grounded.get('edgeSet', [])):
        if "e2" + str(i) in grounding:
            edge['rightkbID'] = grounding["e2" + str(i)]
        if "r{}d".format(i) in grounding:
            edge['kbID'] = grounding["r{}d".format(i)]
            edge['type'] = 'direct'
        elif "r{}r".format(i) in grounding:
            edge['kbID'] = grounding["r{}r".format(i)]
            edge['type'] = 'reverse'
        elif "r{}v".format(i) in grounding:
            edge['kbID'] = grounding["r{}v".format(i)]
            edge['type'] = 'v-structure'

    return grounded


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

