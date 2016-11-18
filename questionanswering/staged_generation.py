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


def add_entity_and_relations(g):
    new_g = {"tokens": g['tokens'], 'edgeSet': []}
    if len(g.get('entities', [])) > 0:
        entity = g['entities'][0]
        new_edge = {'left': [0], 'right': entity}
        new_g['edgeSet'].append(new_edge)
    return new_g


EXPAND_ACTIONS = [hop_up]
RESTRICT_ACTIONS = [add_entity_and_relations]


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

