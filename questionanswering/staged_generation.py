import copy


def get_available_expansions(g):
    return EXPAND_ACTIONS


def get_available_restrictions(g):
    return RESTRICT_ACTIONS


def remove_token_from_entity(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': []}
    for edge in g.get('edgeSet', []):
        new_edge = copy.copy(edge)
        new_edge['right'] = new_edge['right'][:-1]
        new_g['edgeSet'].append(new_edge)
    return new_g


def hop_up(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': []}
    for edge in g.get('edgeSet', []):
        new_edge = copy.copy(edge)
        new_edge['hopUp'] = 1
        new_g['edgeSet'].append(new_edge)
    return new_g


def add_entity_and_relation(g, entities_left):
    new_g = {"tokens": g['tokens'], 'edgeSet': g['edgeSet'][:]}

    if len(entities_left) > 0:
        entity = entities_left[0]
        new_edge = {'left': [0], 'right': entity}
        new_g['edgeSet'].append(new_edge)
    return new_g, (entities_left[1:] if len(entities_left) > 1 else [])


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


def ground_with_gold(g, question_obj):
    pool = [] # pool of possible parses

    pool.append(g)
    while len(pool) > 0:
        g = pool.pop()
        suggested_graphs = restrict(g)
        suggested_graphs = [apply_grounding(p, s_g) for s_g in suggested_graphs for p in query_wikidata(graph_to_query(s_g))]
        chosen_graphs = [(s_g, ) + evaluate(query_wikidata(graph_to_query(s_g)), question_obj) for s_g in suggested_graphs]
        chosen_graphs = [(s_g, f1, answers) for s_g, f1, answers in chosen_graphs if f1 > 0.0]
        while len(chosen_graphs) == 0:
            suggested_graphs = [e_g for s_g in suggested_graphs for e_g in expand(s_g)]
            chosen_graphs = [(s_g, ) + evaluate(query_wikidata(graph_to_query(s_g)), question_obj) for s_g in suggested_graphs]
            chosen_graphs = [(s_g, f1, answers) for s_g, f1, answers in chosen_graphs if f1 > 0.0]
        pool.extend(chosen_graphs)

    return pool

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

