import copy
from wikidata_access import *
from evaluation import *
from webquestions_io import *


def get_available_expansions(g):
    if len(g['edgeSet']) > 0 and 'hopUp' not in g['edgeSet'][-1]:
        return [hop_up]
    return []


def get_available_restrictions(g):
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
    new_g = {"tokens": g['tokens'], 'edgeSet': copy.deepcopy(g['edgeSet'])}
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

    :param g:
    :return:
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[[2, 3]]})

    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "edgeSet":[{"left":[0], "right":[2,3]}]})
    """

    available_expansions = get_available_expansions(g)
    return_graphs = [f(g) for f in available_expansions]
    return return_graphs


def restrict(g):
    available_restrictions = get_available_restrictions(g)
    return_graphs = [f(g) for f in available_restrictions]
    return return_graphs


def generate_with_gold(ungrounded_graph, question_obj):
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

    return pool


def ground_with_gold(suggested_graphs, question_obj):
    suggested_graphs = [apply_grounding(p, s_g) for s_g in suggested_graphs for p in
                        query_wikidata(graph_to_query(s_g))]
    retrieved_answers = [query_wikidata(graph_to_query(s_g, return_var_values=True)) for s_g in suggested_graphs]
    retrieved_answers = [r['e1'] for r in retrieved_answers]
    retrieved_answers = [e.lower() for a in retrieved_answers for e in entity_map.get(a, [a])]

    evaluation_results = [retrieval_prec_rec_f1(get_answers_from_question(question_obj), retrieved_answers[i]) for i in
                          range(len(suggested_graphs))]
    chosen_graphs = [(suggested_graphs[i], evaluation_results[i], retrieved_answers[i])
                     for i in range(len(suggested_graphs)) if evaluation_results[i][2] > 0.0]
    return chosen_graphs


def apply_grounding(g, grounding):
    grounded = copy.deepcopy(g)
    for i, edge in enumerate(grounded.get('edgeSet', [])):
        if "e2" + str(i) in grounding:
            edge['rightkbID'] = grounding["e2" + str(i)]
        if "r{}d".format(i) in grounding:
            edge['kbID'] = grounding["r" + str(i)]
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

