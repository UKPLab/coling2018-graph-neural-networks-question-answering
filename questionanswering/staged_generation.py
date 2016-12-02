import copy
from wikidata_access import *
from evaluation import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def possible_subentities(entity_tokens):
    """
    Retrive all possible subentities of the given entity. Short title tokens are also capitalized.

    :param entity_tokens: a list of entity tokens
    :return: a list of subentities.
    >>> possible_subentities(["Nfl", "Redskins"])
    [('Nfl',), ('Redskins',), ('NFL',)]
    >>> possible_subentities(["senator"])
    []
    >>> possible_subentities(["Grand", "Bahama", "Island"])
    [('Grand', 'Bahama'), ('Bahama', 'Island'), ('Grand',), ('Bahama',), ('Island',)]
    """
    new_entities = []
    for i in range(len(entity_tokens) - 1, 0, -1):
        for new_entity in nltk.ngrams(entity_tokens, i):
            new_entities.append(new_entity)
    for new_entity in [(ne.upper(),) for ne in entity_tokens if len(ne) < 5 and ne.istitle()]:
        new_entities.append(new_entity)
    return new_entities


def last_relation_subentities(g):
    """
    Takes a graph with an existing relation and suggests a set of graphs with the same relation but one of the entities
     is a sub-span of the original entity.

    :param g: a graph with an non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_subentities({'edgeSet': [], 'entities': [['grand', 'bahama', 'island']], 'tokens': ['what', 'country', 'is', 'the', 'grand', 'bahama', 'island', 'in', '?']})
    []
    >>> len(last_relation_subentities({'edgeSet': [{'left':[0], 'right': ['grand', 'bahama', 'island']}], 'entities': [], 'tokens': ['what', 'country', 'is', 'the', 'grand', 'bahama', 'island', 'in', '?']}))
    5
    >>> last_relation_subentities({'edgeSet': [{'right':['Jfk']}], 'entities': []}) == [{'tokens': [], 'edgeSet': [{'right': ['JFK']}], 'entities': []}]
    True
    """
    if len(g.get('edgeSet', [])) == 0 or len(g['edgeSet'][-1]['right']) < 1:
        return []
    new_graphs = []
    right_entity = g['edgeSet'][-1]['right']
    for new_entity in possible_subentities(right_entity):
        new_g = {"tokens": g.get('tokens', []), 'edgeSet': copy.deepcopy(g['edgeSet']),
                 'entities': copy.copy(g.get('entities', []))}
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
    >>> last_relation_hop_up({'edgeSet': [{'left':[0], 'right':[4,5,6]}], 'entities': []}) == [{'edgeSet': [{'left':[0], 'right':[4,5,6], 'hopUp': None}], 'entities': [], 'tokens':[]}]
    True
    >>> last_relation_hop_up({'edgeSet': [{'left':[0], 'right':[4,5,6], 'hopUp': None}], 'entities': []})
    []
    >>> last_relation_hop_up({'edgeSet': [{'left':[0], 'right':["Bahama"], "rightkbID":"Q6754"}], 'entities': []}) == [{'edgeSet': [{'left':[0], 'right':["Bahama"], "rightkbID":"Q6754", 'hopUp': None}], 'entities': [], 'tokens':[]}]
    True
    """
    if len(g.get('edgeSet', [])) == 0 or 'hopUp' in g['edgeSet'][-1]:
        return []
    new_g = {"tokens": g.get('tokens', []), 'edgeSet': copy.deepcopy(g['edgeSet']),
             'entities': copy.copy(g.get('entities', []))}
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
    >>> add_entity_and_relation({'edgeSet': [], 'entities': [["Natalie", "Portman"]]}) == [{'tokens': [], 'edgeSet': [{'left': [0], 'right': ['Natalie', 'Portman'], 'rightkbID': 'Q37876'}], 'entities': []}]
    True
    """
    if len(g.get('entities', [])) == 0:
        return []
    entities = copy.copy(g.get('entities', []))
    linkings = []
    entity = None
    while entities and not linkings:
        entity = entities.pop(0)
        linkings = link_entity(entity)
    if not (linkings and entity):
        return []
    new_graphs = []
    for linking in linkings:
        new_g = {"tokens": g.get('tokens', []), 'edgeSet': copy.deepcopy(g.get('edgeSet', [])), 'entities': entities}
        new_edge = {'left': [0], 'right': entity, 'rightkbID': linking}
        new_g['edgeSet'].append(new_edge)
        new_graphs.append(new_g)

    return new_graphs


def last_relation_temporal(g):
    """
    Adds a temporal argmax to the last relation in the graph, that is only the latest entity is returned as the answer.

    :param g: a graph with a non-empty edgeSet
    :return: a list of suggested graphs
    >>> last_relation_temporal({'edgeSet': [{'left':[0], 'right':[2]}, {'left':[0], 'right':[8]}], 'entities': []}) == [{'edgeSet': [{'left':[0], 'right':[2]}, {'left':[0], 'right':[8], 'argmax': 'time'}], 'entities': [], 'tokens':[]}, {'edgeSet': [{'left':[0], 'right':[2]}, {'left':[0], 'right':[8], 'argmin': 'time'}], 'entities': [], 'tokens':[]}]
    True
    >>> last_relation_temporal({'edgeSet': [{'left':[0], 'right':[2]}, {'left':[0], 'right':[8], 'argmin':'time'}], 'entities': []})
    []
    """
    if len(g.get('edgeSet', [])) == 0 or any(t in g['edgeSet'][-1] for t in ARG_TYPES):
        return []
    new_graphs = []
    for t in ARG_TYPES:
        new_g = {"tokens": g.get('tokens', []), 'edgeSet': copy.deepcopy(g['edgeSet']),
                 'entities': copy.copy(g.get('entities', []))}
        new_g['edgeSet'][-1][t] = "time"
        new_graphs.append(new_g)
    return new_graphs


# This division of actions is relevant for grounding with gold answers:
# - Restrict action limit the set of answers and should be applied
#   to a graph that has groundings
RESTRICT_ACTIONS = [add_entity_and_relation, last_relation_temporal]
# - Expand actions change graph to extract another set of answers and should be
#   applied to a graph that has empty denotation
EXPAND_ACTIONS = [last_relation_hop_up]  # Expand actions

# This division is relevant for constructing all possible groundings without gold answers:
# - WikiData actions need to be grounded in Wikidata in order to construct the next graph
WIKIDATA_ACTIONS = [add_entity_and_relation]
# - Non linking options just add options to the graph structure without checking if it is possible in WikiData.
#   Hopup is alwasy possible anyway, temporal is possible most of the time.
NON_LINKING_ACTIONS = [last_relation_temporal, last_relation_hop_up]

ARG_TYPES = ['argmax', 'argmin']


def expand(g):
    """
    Expand the coverage of the given graph by constructing version that has more possible/other denotations.

    :param g: dict object representing the graph with "edgeSet" and "entities"
    :return: a list of new graphs that are modified copies
    >>> expand({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[["Barack", "Obama"]]})
    []
    >>> expand({"edgeSet":[{"left":[0], "right":["Barack", "Obama"]}]}) == [{'edgeSet': [{'left': [0], 'hopUp': None, 'right': ['Barack', 'Obama']}], 'tokens': [], 'entities': []}, {'edgeSet': [{'left': [0], 'right': ['OBAMA']}], 'tokens': [], 'entities': []}, {'edgeSet': [{'left': [0], 'right': ['Barack']}], 'tokens': [], 'entities': []}, {'edgeSet': [{'left': [0], 'right': ['Obama']}], 'tokens': [], 'entities': []}]
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
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "entities":[[2, 3]]}) == [{'edgeSet': [{'left': [0], 'right': [2, 3]}], 'entities': [], 'tokens': ['Who', 'is', 'Barack', 'Obama', '?']}]
    True
    >>> restrict({"tokens": ['Who', 'is', 'Barack', 'Obama', '?'], "edgeSet":[{"left":[0], "right":[2,3]}]})
    []
    """
    if "entities" not in g:
        return []
    available_restrictions = RESTRICT_ACTIONS
    return_graphs = [el for f in available_restrictions for el in f(g)]
    return return_graphs


def generate_with_gold(ungrounded_graph, gold_answers):
    """
    Generate all possible groundings that produce positive f-score starting with the given ungrounded graph and
    using expand and restrict operations on its denotation.

    :param ungrounded_graph: the starting graph that should contain a list of tokens and a list of entities
    :param gold_answers: list of gold answers for the encoded question
    :return: a list of generated grounded graphs
    """
    pool = [(ungrounded_graph, (0.0, 0.0, 0.0), [])]  # pool of possible parses
    generated_graphs = []
    iterations = 0
    while pool and (generated_graphs[-1][2] if len(generated_graphs) > 0 else 0.0) < 0.7:
        iterations += 1
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))

        if g[1][2] < 0.5:
            logger.debug("Restricting")
            restricted_graphs = restrict(g[0])
            logger.debug("Suggested graphs: {}".format(restricted_graphs))
            chosen_graphs = []
            suggested_graphs = restricted_graphs[:]
            while not chosen_graphs and suggested_graphs:
                s_g = suggested_graphs.pop(0)
                chosen_graphs = ground_with_gold([s_g], gold_answers)
                if not chosen_graphs:
                    logger.debug("Expanding")
                    expanded_graphs = expand(s_g)
                    logger.debug("Expanded graphs: {}".format(expanded_graphs))
                    chosen_graphs = ground_with_gold(expanded_graphs, gold_answers)
            if len(chosen_graphs) > 0:
                logger.debug("Extending the pool.")
                pool.extend(chosen_graphs)
            else:
                logger.debug("Extending the generated graph set: {}".format(g))
                generated_graphs.append(g)
        else:
            logger.debug("Extending the generated graph set.")
            generated_graphs.append(g)
    logger.debug("Iterations {}".format(iterations))
    return generated_graphs


def find_groundings(input_graphs):
    """
    Retrieve possible groundings of the give list of (partially ungrounded) graph.
    If a graph doesn't have free variables, itself is returned.

    :param input_graphs: a list of ungrounded graphs.
    :return: a list of possible grounded graphs.
    """
    grounded_graphs = []
    for s_g in input_graphs:
        if get_free_variables(s_g):
            grounded_graphs.extend([apply_grounding(s_g, p) for p in query_wikidata(graph_to_query(s_g))])
        else:
            grounded_graphs.append(s_g)

    logger.debug("Number of possible groundings: {}".format(len(grounded_graphs)))
    logger.debug("First one: {}".format(grounded_graphs[:1]))
    return grounded_graphs


def ground_with_gold(input_graphs, gold_answers):
    """
    For each graph among the suggested_graphs find its groundings in the WikiData, then evaluate each suggested graph
    with each of its possible groundings and compare the denotations with the answers embedded in the question_obj.
    Return all groundings that produce an f-score > 0.0

    :param input_graphs: a list of ungrounded graphs
    :param gold_answers: a set of gold answers
    :return: a list of graph groundings
    """
    grounded_graphs = find_groundings(input_graphs)
    retrieved_answers = [query_wikidata(graph_to_query(s_g, return_var_values=True)) for s_g in grounded_graphs]
    logger.debug(
        "Number of retrieved answer sets: {}. Example: {}".format(len(retrieved_answers), retrieved_answers[:1]))
    retrieved_answers = [label_query_results(answer_set) for answer_set in retrieved_answers]

    evaluation_results = [retrieval_prec_rec_f1_with_altlabels(gold_answers, retrieved_answers[i]) for i in
                          range(len(grounded_graphs))]
    chosen_graphs = [(grounded_graphs[i], evaluation_results[i], retrieved_answers[i])
                     for i in range(len(grounded_graphs)) if evaluation_results[i][2] > 0.0]
    logger.debug("Number of chosen groundings: {}".format(len(chosen_graphs)))
    return chosen_graphs


def generate_without_gold(ungrounded_graph,
                          wikidata_actions=WIKIDATA_ACTIONS, non_linking_actions=NON_LINKING_ACTIONS):
    """
    Generate all possible groundings of the given ungrounded graph
    using expand and restrict operations on its denotation.

    :param ungrounded_graph: the starting graph that should contain a list of tokens and a list of entities
    :param wikidata_actions: optional, list of actions to apply with grounding in WikiData
    :param non_linking_actions: optional, list of actions to apply without checking in WikiData
    :return: a list of generated grounded graphs
    """
    pool = [(ungrounded_graph, (0.0, 0.0, 0.0), [])]  # pool of possible parses
    generated_graphs = []
    iterations = 0
    while pool:
        if len(generated_graphs) % 50 == 0:
            logger.debug("Generated", len(generated_graphs))
            logger.debug("Pool", len(pool))
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))

        logger.debug("Constructing with WikiData")
        suggested_graphs = [el for f in wikidata_actions for el in f(g[0])]
        logger.debug("Suggested graphs: {}".format(suggested_graphs))
        chosen_graphs = ground_without_gold(suggested_graphs)

        logger.debug("Constructing without WikiData")
        extended_graphs = [(el, f, a) for s_g, f, a in chosen_graphs for f in non_linking_actions for el in f(s_g)]
        chosen_graphs.extend(extended_graphs)

        logger.debug("Extending the pool with {} graphs.".format(len(chosen_graphs)))
        pool.extend(chosen_graphs)
        logger.debug("Extending the generated with {} graphs.".format(len(chosen_graphs)))
        generated_graphs.extend(chosen_graphs)
        iterations += 1
    logger.debug("Iterations {}".format(iterations))
    return generated_graphs


def ground_without_gold(input_graphs):
    """

    :param input_graphs: a list of ungrounded graphs
    :return: a list of graph groundings
    """
    grounded_graphs = find_groundings(input_graphs)

    chosen_graphs = [(grounded_graphs[i], (0.0, 0.0, 0.0), [])
                     for i in range(len(grounded_graphs))]
    logger.debug("Number of chosen groundings: {}".format(len(chosen_graphs)))
    return chosen_graphs


def link_entity(entity_tokens):
    """
    Link the given list of tokens to an entity in a knowledge base. If none linkings is found try all combinations of
    subtokens of the given entity.

    :param entity_tokens: list of entity tokens
    :return: list of KB ids
    """
    linkings = query_wikidata(entity_query(" ".join(entity_tokens)))
    if not linkings:
        subentities = possible_subentities(entity_tokens)
        while not linkings and subentities:
            subentity_tokens = subentities.pop(0)
            linkings = query_wikidata(entity_query(" ".join(subentity_tokens)))
    linkings = [l.get("e20", "") for l in linkings if l]
    linkings = sorted(linkings, key=lambda k: int(k[1:]))
    return linkings


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
    >>> apply_grounding({'edgeSet':[{}]}, {'r0v':'P31v', 'hopup0v':'P131v'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'hopUp':'P131v'}]}
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
        if "hopup{}v".format(i) in grounding:
            edge['hopUp'] = grounding["hopup{}v".format(i)]
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
