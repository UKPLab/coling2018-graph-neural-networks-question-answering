import pytest

from questionanswering.construction import sentence
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering.grounding import staged_generation, graph_queries

from entitylinking import core
from test_sparql_queries import test_graphs_grounded

test_graphs_with_groundings = [
    SemanticGraph([Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='P674', rightentityid='Q3899725'),
                   Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid='Q571')]),
    SemanticGraph([Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid='Q3899725'),
                   Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid='Q571')]),
    SemanticGraph([Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='class', rightentityid='Q6256')]),
    SemanticGraph([Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='class', rightentityid='Q6256'),
                   Edge(leftentityid='Q866345', rightentityid=graph_queries.QUESTION_VAR)]),
    SemanticGraph(edges=[Edge(qualifierentityid=graph_queries.QUESTION_VAR, rightentityid='Q5620660')]),
    SemanticGraph(edges=[Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='class', rightentityid='Q5'),
                         Edge(qualifierentityid=graph_queries.QUESTION_VAR, rightentityid='Q5620660')]),
    SemanticGraph([Edge(rightentityid=graph_queries.QUESTION_VAR, relationid='P161', qualifierentityid='Q5620660'),
                   Edge(leftentityid='Q1079', rightentityid=graph_queries.QUESTION_VAR)]),
]


test_graphs_without_groundings = [
    SemanticGraph([Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='P1376', rightentityid='Q183'),
                   Edge(leftentityid=graph_queries.QUESTION_VAR, relationid='class', rightentityid='Q37226')]),
]

test_sentences_perfect_fscore = [
    ("who was the winner of the nobel peace prize in 2009?", ['Q76']),
    ("when were the texas rangers started?", ['1972']),
    ("who wrote the song hotel california?", ['Q704399']),
    ("who is the chancellor of Germany?", ['Q567']),
    ("what actors are from Fountain Valley?", ['Q229426', 'Q458984', 'Q4936852', 'Q4662412', 'Q720754']),
]

test_sentences_positive_fscore = [
    ("what is the zip code of chicago?", ['60655', '60606', '60601']),
    ("In which school did Obama's wife study?", ['Q7996715', 'Q21578']),
    ("what time zone am i in cleveland ohio?", ['Q941023']),
]

entitylinker = core.MLLinker(
    path_to_model='../../entity-linking/trainedmodels/FeatureModel_6.torchweights',
    num_candidates=1,
    confidence=0.1)


def test_find_groundings():
    groundings = []
    for test_graph in test_graphs_with_groundings:
        results = graph_queries.get_graph_groundings(test_graph)
        groundings.append(results)
        assert len(results) > 0
    assert groundings[0] == [{'r1v': 'P31v'}]
    assert groundings[1] == [{'r0v': 'P674v', 'r1v': 'P31v'}]
    assert groundings[3] == [{'r1v': 'P17v'}]
    assert {'r1v': 'P453q'} in groundings[5]
    assert groundings[6] == [{'r1v': 'P161v'}]

    for test_graph in test_graphs_without_groundings:
        results = graph_queries.get_graph_groundings(test_graph)
        assert len(results) == 0

    for test_graph in test_graphs_grounded:
        results = graph_queries.get_graph_groundings(test_graph)
        assert len(results) == 1 and results[0] == {}


def test_ground_with_gold():
    for test_sent, answers in test_sentences_perfect_fscore:
        sent = entitylinker.link_entities_in_raw_input(test_sent)
        sent = sentence.Sentence(input_text=sent.input_text, tagged=sent.tagged, entities=sent.entities)

        graphs = staged_generation.generate_with_gold(sent.graphs[0], answers)
        graphs = sorted(graphs, key=lambda x: x[1][2], reverse=True)
        assert graphs[0][1][2] == 1.0

    for test_sent, answers in test_sentences_positive_fscore:
        sent = entitylinker.link_entities_in_raw_input(test_sent)
        sent = sentence.Sentence(input_text=sent.input_text, tagged=sent.tagged, entities=sent.entities)

        graphs = staged_generation.generate_with_gold(sent.graphs[0], answers)
        graphs = sorted(graphs, key=lambda x: x[1][2], reverse=True)
        assert graphs[0][1][2] > 0.5


if __name__ == '__main__':
    pytest.main(['-v', __file__])
