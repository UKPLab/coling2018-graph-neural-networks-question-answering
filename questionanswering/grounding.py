import staged_generation

def ground_with_gold(g, question_obj):
    silver_graphs = []
    states = []
    queque = []

    queque.extend(staged_generation.restrict(g))

    # select at least one relation to one entity without type
    # search for filling relations
        # if any of them results in recall more than 0.0 take them
            # for each of them continue with further restrict
                # select at least one relation to one entity without type and search for filling relations
        # else take the structure
            # Atempt expand
                # Search for filling relations

    # while len(queque) > 0:

    # possible_groundings = query_wikidata(graph_to_query(g))
    # for possible_grounding in possible_groundings:
    #     grounded_graph = apply_grounding(g, possible_grounding)
    #     results = query_wikidata(graph_to_query(grounded_graph, return_var_values=True))
    #     answers = [r['e1'] for r in results]
    #     answers = [e.lower() for a in answers for e in entity_map.get(a, [a])]
    #     grounded_graph['answers'] = answers
    #     gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(question_obj)]
    #     f1 = retrieval_prec_rec_f1(gold_answers, answers)[2] if len(answers) > 0 else 0.0
    #     grounded_graph['f1'] = f1
    #     silver_graphs.append(grounded_graph)
    return silver_graphs


def to_graph(tokens, entities):
    pass

