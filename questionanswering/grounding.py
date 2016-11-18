import staged_generation

def ground_with_gold(g, question_obj):
    silver_graphs = []
    pool = [] # pool of possible parses

    pool.append(g)
    while len(pool) > 0:
        g = pool.pop()
        uncertain_graphs = staged_generation.add_entity_and_relation(g)
        chosen_graphs = evaluate(uncertain_graphs)
        while len(chosen_graphs) == 0:
            uncertain_graphs = expand(uncertain_graphs)
            chosen_graphs = evaluate(uncertain_graphs)
        pool.extend(chosen_graphs)




    pool.extend(staged_generation.restrict(g))



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

def stage_next(g, entities_left):
    g = staged_generation.add_entity_and_relations(g, entities_left)
    possible_groundings = query_wikidata(g)
    results = [(p,) + evaluate(possible_grounding, gold_answers) for p in possible_groundings]
    positive_resulst = results > 0.0
    if positive_resulst:
        for result in positive_resulst:
            stage_next()
    else:
        for result in results:
            expanded_g = expand(result)
            possible_groundings = query_wikidata(g)
            expanded_results = [(p,) + evaluate(possible_grounding, gold_answers) for p in possible_groundings]









def to_graph(tokens, entities):
    pass

