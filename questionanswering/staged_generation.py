import copy

import evaluation
import graph
import stages
import wikidata_access
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


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
    while pool and (generated_graphs[-1][1][2] if len(generated_graphs) > 0 else 0.0) < 0.7:
        iterations += 1
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))

        if g[1][2] < 0.5:
            logger.debug("Restricting")
            restricted_graphs = stages.restrict(g[0])
            logger.debug("Suggested graphs: {}".format(restricted_graphs))
            chosen_graphs = []
            suggested_graphs = restricted_graphs[:]
            while not chosen_graphs and suggested_graphs:
                s_g = suggested_graphs.pop(0)
                chosen_graphs = ground_with_gold([s_g], gold_answers)
                if not chosen_graphs:
                    logger.debug("Expanding")
                    expanded_graphs = stages.expand(s_g)
                    logger.debug("Expanded graphs: {}".format(expanded_graphs))
                    chosen_graphs = ground_with_gold(expanded_graphs, gold_answers)
            if len(chosen_graphs) > 0:
                logger.debug("Extending the pool.")
                pool.extend(chosen_graphs)
            else:
                logger.debug("Extending the generated graph set: {}".format(g))
                generated_graphs.append(g)
        else:
            logger.debug("Extending the generated graph set: {}".format(g))
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
        if wikidata_access.get_free_variables(s_g):
            grounded_graphs.extend([apply_grounding(s_g, p) for p in wikidata_access.query_graph_groundings(s_g)])
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
    retrieved_answers = [wikidata_access.query_graph_denotations(s_g) for s_g in grounded_graphs]
    logger.debug(
        "Number of retrieved answer sets: {}. Example: {}".format(len(retrieved_answers), retrieved_answers[:1]))
    retrieved_answers = [wikidata_access.map_query_results(answer_set) for answer_set in retrieved_answers]

    evaluation_results = [evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers, retrieved_answers[i]) for i in
                          range(len(grounded_graphs))]
    chosen_graphs = [(grounded_graphs[i], evaluation_results[i], retrieved_answers[i])
                     for i in range(len(grounded_graphs)) if evaluation_results[i][2] > 0.0]
    if len(chosen_graphs) > 3:
        chosen_graphs = sorted(chosen_graphs, key=lambda x:x[1][2], reverse=True)[:3]
    logger.debug("Number of chosen groundings: {}".format(len(chosen_graphs)))
    return chosen_graphs


def generate_without_gold(ungrounded_graph,
                          wikidata_actions=stages.WIKIDATA_ACTIONS, non_linking_actions=stages.NON_LINKING_ACTIONS):
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
    grounded = graph.copy_graph(g)
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
