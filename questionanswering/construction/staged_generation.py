import itertools
import logging

import utils
from construction import stages, graph
from datasets import evaluation
from wikidata import entity_linking
from wikidata import wdaccess

generation_p = {
    'label.query.results': True,
    'logger': logging.getLogger(__name__),
    'replace.entities': True,
    'use.whitelist': False,
    'v.structure': True,
    'min.fscore.to.stop': 0.9
}

logger = generation_p['logger']
logger.setLevel(logging.ERROR)

v_structure_markers = utils.load_blacklist(utils.RESOURCES_FOLDER + "v_structure_markers.txt")


def generate_with_gold(ungrounded_graph, gold_answers):
    """
    Generate all possible groundings that produce positive f-score starting with the given ungrounded graph and
    using expand and restrict operations on its denotation.

    :param ungrounded_graph: the starting graph that should contain a list of tokens and a list of entities
    :param gold_answers: list of gold answers for the encoded question
    :return: a list of generated grounded graphs
    >>> max(g[1][2] if len(g) > 1 else 0.0 for g in generate_with_gold({'edgeSet': [], 'entities': [(['Nobel', 'Peace', 'Prize'], 'URL'), (['the', 'winner'], 'NN'), (['2009'], 'CD')]}, gold_answers=['barack obama']))
    1.0
    >>> max(g[1][2] if len(g) > 1 else 0.0 for g in generate_with_gold({'edgeSet': [], 'entities': [(['Texas', 'Rangers'], 'URL')], \
            'tokens': ['when', 'were', 'the', 'texas', 'rangers', 'started', '?']}, gold_answers=['1972']))
    1.0
    >>> max(g[1][2] if len(g) > 1 else 0.0 for g in generate_with_gold({'edgeSet': [], 'entities': [(['Chicago'], 'LOCATION')], \
    'tokens': ['what','is','the','zip','code','of','chicago','?']}, gold_answers=['60605', '60604', '60607', '60606', '60601', '60610', '60603', '60602', '60290', '60608'])) > 0.05
    True
    """
    ungrounded_graph = link_entities_in_graph(ungrounded_graph)
    pool = [(ungrounded_graph, (0.0, 0.0, 0.0), [])]  # pool of possible parses
    positive_graphs, negative_graphs = [], []
    iterations = 0
    while pool and (positive_graphs[-1][1][2] if len(positive_graphs) > 0 else 0.0) < generation_p['min.fscore.to.stop']:
        iterations += 1
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))
        master_g_fscore = g[1][2]
        if master_g_fscore < generation_p['min.fscore.to.stop']:
            logger.debug("Restricting")
            restricted_graphs = stages.restrict(g[0])
            restricted_graphs = [add_canonical_labels_to_entities(r_g) for r_g in restricted_graphs]
            logger.debug("Suggested graphs: {}".format(restricted_graphs))
            chosen_graphs = []
            suggested_graphs = restricted_graphs[:]
            bonus_round = False
            while (not chosen_graphs or bonus_round) and suggested_graphs:
                if bonus_round:
                    logger.debug("Bonus round!")
                bonus_round = False
                s_g = suggested_graphs.pop(0)
                temp_chosen, not_chosen_graphs = ground_with_gold([s_g], gold_answers, min_fscore=master_g_fscore)
                chosen_graphs += temp_chosen
                negative_graphs += not_chosen_graphs
                logger.debug("Chosen graphs length: {}".format(len(chosen_graphs)))
                if not chosen_graphs:
                    logger.debug("Expanding")
                    expanded_graphs = stages.expand(s_g)
                    logger.debug("Expanded graphs (10): {}".format(expanded_graphs[:10]))
                    temp_chosen, not_chosen_graphs = ground_with_gold(expanded_graphs, gold_answers, min_fscore=master_g_fscore)
                    chosen_graphs += temp_chosen
                    negative_graphs += not_chosen_graphs
                if chosen_graphs:
                    current_f1 = max(g[1][2] for g in chosen_graphs)
                    if current_f1 < 0.05:
                        bonus_round = True
                        master_g_fscore = current_f1
            if len(chosen_graphs) > 0:
                logger.debug("Extending the pool.")
                pool.extend(chosen_graphs)
                pool = sorted(pool, key=lambda x: x[1][2], reverse=True)
            else:
                logger.debug("Extending the generated graph set: {}".format(g))
                positive_graphs.append(g)
        else:
            logger.debug("Extending the generated graph set: {}".format(g))
            positive_graphs.append(g)
    logger.debug("Iterations {}".format(iterations))
    logger.debug("Negative {}".format(len(negative_graphs)))
    return positive_graphs + negative_graphs


def link_entities_in_graph(ungrounded_graph):
    """
    Link all free entities in the graph.

    :param ungrounded_graph: graph as a dictionary with 'entities'
    :return: graph with entity linkings in the 'entities' array
    >>> link_entities_in_graph({'entities': [(['Norway'], 'LOCATION'), (['oil'], 'NN')], 'tokens': ['where', 'does', 'norway', 'get', 'their', 'oil', '?']})['entities']
    [(['Norway'], 'LOCATION', [('Q20', 'Norway'), ('Q944765', 'Norway'), ('Q1913264', 'Norway')]), (['oil'], 'NN', [('Q42962', 'oil'), ('Q1130872', 'Oil'), ('Q7081283', 'Oil')])]
    >>> link_entities_in_graph({'entities': [(['Bella'], 'PERSON'), (['Twilight'], 'NNP')], 'tokens': ['who', 'plays', 'bella', 'on', 'twilight', '?']})['entities']
    [(['Bella'], 'PERSON', [[('Q223757', 'Bella Swan')], ('Q52533', 'Bella, Basilicata'), ('Q156571', '695 Bella')]), (['Twilight'], 'NNP', [('Q44523', 'Twilight'), ('Q160071', 'Twilight'), ('Q189378', 'Twilight')])]
    >>> link_entities_in_graph({'entities': [(['Bella'], 'PERSON'), (['2012'], 'CD')], 'tokens': ['who', 'plays', 'bella', 'on', 'twilight', '?']})['entities']
    [(['Bella'], 'PERSON', [('Q52533', 'Bella, Basilicata'), ('Q156571', '695 Bella'), ('Q231665', 'Belladonna')]), (['2012'], 'CD')]
    """
    entities = []
    if all(len(e) == 3 for e in ungrounded_graph.get('entities', [])):
        return ungrounded_graph
    for entity in ungrounded_graph.get('entities', []):
        if len(entity) == 2 and entity[1] != "CD":
            grouped_linkings = entity_linking.link_entity(entity)
            for linkings in grouped_linkings:
                entities.append(list(entity) + [linkings])
        else:
            entities.append(entity)
    if any(w in set(ungrounded_graph.get('tokens', [])) for w in v_structure_markers):
        for entity in [e for e in entities if e[1] == "PERSON" and len(e[0]) == 1 and len(e) == 3]:
            for film_id in [e_id for e in [e for e in entities if len(e) == 3] for e_id, l in e[2] if e != entity]:
                character_linkings = wdaccess.query_wikidata(wdaccess.character_query(" ".join(entity[0]), film_id), starts_with=None)
                character_linkings = entity_linking.post_process_entity_linkings(entity[0], character_linkings)
                entity[2] = character_linkings + entity[2]
                entity[2] = entity[2][:entity_linking.entity_linking_p.get("max.entity.options", 3)]
    entities = [tuple(e) for e in entities]
    ungrounded_graph['entities'] = entities
    return ungrounded_graph


def ground_with_gold(input_graphs, gold_answers, min_fscore=0.0):
    """
    For each graph among the suggested_graphs find its groundings in the WikiData, then evaluate each suggested graph
    with each of its possible groundings and compare the denotations with the answers embedded in the question_obj.
    Return all groundings that produce an f-score > 0.0

    :param input_graphs: a list of ungrounded graphs
    :param gold_answers: a set of gold answers
    :param min_fscore: lower bound on f-score for returned positive graphs
    :return: a list of graph groundings
    """
    logger.debug("Input graphs: {}".format(input_graphs))
    all_chosen_graphs, all_not_chosen_graphs = [], []
    input_graphs = input_graphs[:]
    while input_graphs and len(all_chosen_graphs) == 0:
        s_g = input_graphs.pop(0)
        chosen_graphs, not_chosen_graphs = ground_one_with_gold(s_g, gold_answers, min_fscore)
        all_chosen_graphs += chosen_graphs
        all_not_chosen_graphs += not_chosen_graphs
    all_chosen_graphs = sorted(all_chosen_graphs, key=lambda x: x[1][2], reverse=True)
    if len(all_chosen_graphs) > 3:
        all_chosen_graphs = all_chosen_graphs[:3]
    logger.debug("Number of chosen groundings: {}".format(len(all_chosen_graphs)))
    return all_chosen_graphs, all_not_chosen_graphs


def ground_one_with_gold(s_g, gold_answers, min_fscore):
    grounded_graphs = [apply_grounding(s_g, p) for p in find_groundings(s_g)]
    logger.debug("Number of possible groundings: {}".format(len(grounded_graphs)))
    logger.debug("First one: {}".format(grounded_graphs[:1]))
    retrieved_answers = [wdaccess.query_graph_denotations(s_g) for s_g in grounded_graphs]
    for i, s_g in enumerate(grounded_graphs):
        if len(retrieved_answers[i]) > 3:  # basically means there is no temporal relations there
            t_g = graph.copy_graph(s_g)
            t_g['filter'] = 'importance'
            grounded_graphs.append(t_g)
            retrieved_answers.append(wdaccess.filter_denotation_by_importance(retrieved_answers[i]))

    post_process_results = wdaccess.label_query_results if generation_p[
        'label.query.results'] else lambda x: [[r["e1"]] for r in x]
    retrieved_answers = [post_process_results(answer_set) for answer_set in retrieved_answers]
    retrieved_answers = [post_process_answers_given_graph(answer_set, grounded_graphs[i]) for i, answer_set in enumerate(retrieved_answers)]
    logger.debug(
        "Number of retrieved answer sets: {}. Example: {}".format(len(retrieved_answers),
                                                                  retrieved_answers[0][:10] if len(
                                                                      retrieved_answers) > 0 else []))
    evaluation_results = [evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers, retrieved_answers[i]) for i in
                          range(len(grounded_graphs))]
    chosen_graphs = [(grounded_graphs[i], evaluation_results[i], retrieved_answers[i])
                     for i in range(len(grounded_graphs)) if evaluation_results[i][2] > min_fscore]
    not_chosen_graphs = [(grounded_graphs[i], (0.0, 0.0, 0.0), len(retrieved_answers[i])) for i in range(len(grounded_graphs)) if evaluation_results[i][2] < 0.01]
    return chosen_graphs, not_chosen_graphs


def find_groundings(g):
    """
    Retrieve possible groundings for a given graph.

    :param g: the graph to ground
    :return: a list of graph groundings.
    >>> len(find_groundings({'edgeSet': [{'right': ['Percy', 'Jackson'], 'rightkbID': 'Q3899725', 'kbID': 'P674v', 'type': 'direct'}, {'rightkbID': 'Q571', 'right': ['book']}]}))
    1
    >>> len(find_groundings({'edgeSet': [{'right': ['Percy', 'Jackson'], 'rightkbID': 'Q3899725'}, {'rightkbID': 'Q571', 'right': ['book']}]}))
    1
    """
    query_results = []
    num_edges_to_ground = sum(1 for e in g.get('edgeSet', []) if not('type' in e and 'kbID' in e))
    if num_edges_to_ground == 1 and not any('hopUp' in e or 'hopDown' in e for e in g.get('edgeSet', []) if not('type' in e and 'kbID' in e)):
        query_results += wdaccess.query_graph_groundings(g)
    else:
        edge_type_combinations = list(itertools.product(*[['direct', 'reverse']]*num_edges_to_ground))
        for type_combindation in edge_type_combinations:
            t = graph.copy_graph(g)
            for i, edge in enumerate([e for e in t.get('edgeSet', []) if not('type' in e and 'kbID' in e)]):
                edge['type'] = type_combindation[i]
            query_results += wdaccess.query_graph_groundings(t)
    if generation_p['v.structure'] and num_edges_to_ground == 1 and any(w in set(g.get('tokens', [])) for w in v_structure_markers):
        t = graph.copy_graph(g)
        edge = [e for e in t.get('edgeSet', []) if not('type' in e and 'kbID' in e)][0]
        edge['type'] = 'v-structure'
        query_results += wdaccess.query_graph_groundings(t)
    return query_results


def ground_with_model(input_graphs, qa_model, min_score, beam_size=10):

    logger.debug("Input graphs: {}".format(len(input_graphs)))
    logger.debug("First input one: {}".format(input_graphs[:1]))

    grounded_graphs = [apply_grounding(s_g, p) for s_g in input_graphs for p in find_groundings(s_g)]
    if generation_p.get('use.whitelist', False):
        grounded_graphs = [g for g in grounded_graphs if all(e.get('type') in {'time', 'v-structure'} or e.get("kbID")[:-1] in wdaccess.property_whitelist for e in g.get('edgeSet', []))]
    logger.debug("Number of possible groundings: {}".format(len(grounded_graphs)))
    grounded_graphs = [graph.add_string_representations_to_edges(g, wdaccess.property2label, generation_p.get("replace.entities", False)) for g in grounded_graphs]
    direct_relations = {graph.get_graph_last_edge(g).get('kbID', "")[:-1]
                        for g in grounded_graphs if graph.get_graph_last_edge(g).get('type') in {'direct', 'reverse', 'v-structure'}
                        and graph.get_graph_last_edge(g).get('kbID', " ")[-1] not in 'qr'}
    grounded_graphs = [g for g in grounded_graphs if graph.get_graph_last_edge(g).get('kbID', " ")[-1] not in 'qr' or
                       graph.get_graph_last_edge(g).get('kbID', "")[:-1] not in direct_relations]
    logger.debug("Filter out unnecessary qualifiers: {}".format(len(grounded_graphs)))
    first_order_relations = {"{}-{}".format(graph.get_graph_last_edge(g).get('kbID'), graph.get_graph_last_edge(g).get('type'))
                        for g in grounded_graphs if graph.get_graph_last_edge(g).get('type') in {'direct', 'reverse', 'v-structure'} and
                             'hopUp' not in graph.get_graph_last_edge(g) and 'hopDown' not in graph.get_graph_last_edge(g)}
    grounded_graphs = [g for g in grounded_graphs if ('hopUp' not in graph.get_graph_last_edge(g) and 'hopDown' not in graph.get_graph_last_edge(g)) or
                       ("{}-{}".format(graph.get_graph_last_edge(g).get('kbID'), graph.get_graph_last_edge(g).get('type')) not in first_order_relations)]
    logger.debug("Filter out unnecessary hops: {}".format(len(grounded_graphs)))
    logger.debug("First one: {}".format(grounded_graphs[:1]))
    if len(grounded_graphs) == 0:
        return []
    tokens = grounded_graphs[0]['tokens']
    model_scores = qa_model.scores_for_instance((tokens, grounded_graphs))
    logger.debug("model_scores: {}".format(model_scores))
    assert len(model_scores) == len(grounded_graphs)
    all_chosen_graphs = [(grounded_graphs[i], model_scores[i])
                     for i in range(len(grounded_graphs)) if model_scores[i] > min_score]

    all_chosen_graphs = sorted(all_chosen_graphs, key=lambda x: x[1], reverse=True)
    if len(all_chosen_graphs) > beam_size:
        all_chosen_graphs = all_chosen_graphs[:beam_size]
    logger.debug("Number of chosen groundings: {}".format(len(all_chosen_graphs)))
    return all_chosen_graphs


def generate_with_model(ungrounded_graph, qa_model, beam_size=10):
    ungrounded_graph = link_entities_in_graph(ungrounded_graph)
    pool = [(ungrounded_graph, -1.0)]  # pool of possible parses
    generated_graphs = []
    iterations = 0
    while pool:
        iterations += 1
        g = pool.pop(0)
        logger.debug("Pool length: {}, Graph: {}".format(len(pool), g))
        master_score = g[1]
        logger.debug("Restricting")
        restricted_graphs = stages.restrict(g[0])
        restricted_graphs = [add_canonical_labels_to_entities(r_g) for r_g in restricted_graphs]
        logger.debug("Suggested graphs:{}, {}".format(len(restricted_graphs), restricted_graphs))
        suggested_graphs = restricted_graphs[:]
        suggested_graphs += [e_g for s_g in suggested_graphs for e_g in stages.expand(s_g)]
        logger.debug("Suggested (expanded) graphs: {}, {}".format(len(suggested_graphs), suggested_graphs))
        chosen_graphs = ground_with_model(suggested_graphs, qa_model, min_score=master_score, beam_size=beam_size)
        logger.debug("Chosen graphs length: {}".format(len(chosen_graphs)))
        if len(chosen_graphs) > 0:
            logger.debug("Extending the pool.")
            pool.extend(chosen_graphs)
            logger.debug("Extending the generated graph set: {}".format(len(chosen_graphs)))
            generated_graphs.extend(chosen_graphs)
    logger.debug("Iterations {}".format(iterations))
    logger.debug("Generated graphs {}".format(len(generated_graphs)))
    generated_graphs = sorted(generated_graphs, key=lambda x: x[1], reverse=True)
    return generated_graphs


def apply_grounding(g, grounding):
    """
    Given a grounding obtained from WikiData apply it to the graph.
    Note: that the variable names returned by WikiData are important as they encode some grounding features.

    :param g: a single ungrounded graph
    :param grounding: a dictionary representing the grounding of relations and variables
    :return: a grounded graph
    >>> apply_grounding({'edgeSet':[{}]}, {'r0d':'P31v'}) == {'edgeSet': [{'type': 'direct', 'kbID': 'P31v', }], 'entities': []}
    True
    >>> apply_grounding({'edgeSet':[{}]}, {'r0v':'P31v'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v'}], 'entities': []}
    True
    >>> apply_grounding({'edgeSet':[{"hopUp":None}]}, {'r0v':'P31v', 'hop0v':'P131v'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'hopUp':'P131v'}], 'entities': []}
    True
    >>> apply_grounding({'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'hopUp':'P131v'}], 'tokens': []}, {}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'hopUp':'P131v'}], 'entities': [], 'tokens': []}
    True
    >>> apply_grounding({'edgeSet':[{}, {}]}, {'r1d':'P39v', 'r0v':'P31v', 'e20': 'Q18'}) == {'edgeSet': [{'type': 'v-structure', 'kbID': 'P31v', 'rightkbID': 'Q18'}, {'type': 'direct', 'kbID': 'P39v'}], 'entities': []}
    True
    >>> apply_grounding({'edgeSet':[]}, {}) == {'entities': [], 'edgeSet': []}
    True
    """
    grounded = graph.copy_graph(g)
    for i, edge in enumerate(grounded.get('edgeSet', [])):
        if "e2" + str(i) in grounding:
            edge['rightkbID'] = grounding["e2" + str(i)]
        if "hop{}v".format(i) in grounding:
            if 'hopUp' in edge:
                edge['hopUp'] = grounding["hop{}v".format(i)]
            else:
                edge['hopDown'] = grounding["hop{}v".format(i)]
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


def add_canonical_labels_to_entities(g):
    """
    Label all the entities in the given graph that participate in relations with their canonical names.

    :param g: a graph as a dictionary with an 'edgeSet'
    :return: the original graph with added labels.
    """
    for edge in g.get('edgeSet', []):
        entitykbID = edge.get('rightkbID')
        if entitykbID and ('canonical_right' not in edge or edge['canonical_right'] is None):
            entity_label = wdaccess.label_entity(entitykbID)
            if entity_label:
                edge['canonical_right'] = entity_label
    return g


def post_process_answers_given_graph(model_answers_labels, g):
    """
    Post process some of the retrieved answers to match freebase canonical labels

    :param model_answers_labels: list of list of answers
    :param g: graph as a dictionary
    :return: list of list of answers
    >>> post_process_answers_given_graph([['eng', 'english']], {'edgeSet':[{'kbID': 'P37v', 'rightkbID':'Q843'}]})
    [['eng', 'english', 'eng language', 'english language'], ['pakistani english', 'pakistani english language']]
    >>> post_process_answers_given_graph([['Q76']], {'edgeSet':[{'kbID': 'P31v', 'rightkbID':'Q3'}]})
    [['Q76']]
    """
    # Language -> add english variants
    relevant_edge = [e for e in g.get('edgeSet', []) if e.get("kbID", "")[:-1] == "P37"]
    if len(relevant_edge) > 0:
        for answer_set in model_answers_labels:
            if all('language' not in a.lower() for a in answer_set):
                answer_set.extend([a + " language" for a in answer_set])
            if 'arabic' in answer_set:
                answer_set.append("modern standard arabic")
            if ('english' in answer_set or 'arabic' in answer_set) and relevant_edge[0].get('rightkbID'):
                demonym = wdaccess.query_wikidata(wdaccess.demonym_query(relevant_edge[0].get('rightkbID')), starts_with="")
                if demonym:
                    demonym = demonym[0]['labelright'].lower()
                    if 'english' in answer_set:
                        model_answers_labels.append([demonym + " english", demonym + " english language"])
                    elif 'arabic' in answer_set:
                        model_answers_labels.append([demonym + " arabic", demonym + " arabic language"])
    # Character role -> split name
    relevant_edge = [e for e in g.get('edgeSet', []) if e.get("kbID", "")[:-1] in {"P175", "P453", "P161"}]
    if len(relevant_edge) > 0:
        for answer_set in model_answers_labels:
            answer_set.extend([a.split()[0].strip() for a in answer_set if len(a.split()) > 0])
    return model_answers_labels


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
