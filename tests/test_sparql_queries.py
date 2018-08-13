import pytest

from questionanswering import grounding
from questionanswering.grounding import stages
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering.grounding import graph_queries

test_graphs_with_groundings = [
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q571", qualifierentityid="MAX")
    ]),
    SemanticGraph(edges=[
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, leftentityid="Q127367"),
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="MAX"),
    ]),
    SemanticGraph(edges=[
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, qualifierentityid="Q37876")
    ]),
    SemanticGraph(edges=[
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, leftentityid="Q329816")
    ], tokens=['when', 'were']),
    SemanticGraph(edges=[
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, leftentityid="Q458")
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q458")
    ]),
    SemanticGraph(edges=[
        Edge(rightentityid="?e1", leftentityid="Q76"),
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, leftentityid="?e1"),
    ]),
]

test_graphs_without_groundings = [
    SemanticGraph(edges=[
        Edge(leftentityid='Q76', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid='P131')
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q3899725', qualifierentityid=grounding.graph_queries.QUESTION_VAR, qualifierrelationid="P813")
    ]),
]

test_graphs_grounded = [
    SemanticGraph(edges=[
        Edge(rightentityid=grounding.graph_queries.QUESTION_VAR, leftentityid="Q76", relationid="P21")
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q19686", relationid="P206"),
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid="Q515", relationid="class")
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q35637', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P1346", qualifierentityid='2009')
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q329816', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P571")
    ], tokens=["when", "did", "start"]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q1297', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P281")
    ], tokens=["what", "zip", "code"]),
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid='Q19686', relationid="P206")
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid='Q19686', relationid="P206"),
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, relationid='P571', qualifierentityid="MIN"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid='Q19686', relationid="P206"),
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid='Q515', relationid="class"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q37571', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P1066"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q84', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P131"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q177', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P527"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q35637', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P1346", qualifierentityid='2009'),
        Edge(leftentityid=grounding.graph_queries.QUESTION_VAR, rightentityid='Q5', relationid="class"),
    ]),
    SemanticGraph(edges=[
        Edge(leftentityid='Q155', rightentityid=grounding.graph_queries.QUESTION_VAR, relationid="P35", qualifierentityid="MAX"),
    ]),
]


def test_graph_to_query():
    for test_graph in test_graphs_with_groundings:
        sparql = graph_queries.graph_to_query(test_graph)
        assert " ?r0v " in sparql

    for test_graph in test_graphs_grounded:
        sparql = graph_queries.graph_to_query(test_graph)
        assert " ?r0v " not in sparql and "SELECT DISTINCT ?qvar WHERE" in sparql


def test_query_graph_groundings():
    groundings = []
    for test_graph in test_graphs_with_groundings:
        results = graph_queries.get_graph_groundings(test_graph)
        groundings.append(results)
        assert len(results) > 1
    assert {'r0v': 'P800v'} in groundings[0]
    assert {'r0v': 'P453q'} in groundings[2]
    assert {'r0v': 'P571v'} in groundings[3]
    assert {'r0v': 'P36v'} in groundings[4]
    assert {'r0v': 'P361v'} in groundings[5]

    for test_graph in test_graphs_without_groundings:
        results = graph_queries.get_graph_groundings(test_graph)
        assert len(results) == 0

    for test_graph in test_graphs_grounded:
        results = graph_queries.get_graph_groundings(test_graph)
        assert len(results) == 1 and results[0] == {}


def test_query_graph_denotations():
    denotations = []
    for test_graph in test_graphs_grounded:
        result = graph_queries.get_graph_denotations(test_graph)
        denotations.append(result)
        assert len(result) > 0
    assert denotations[2] == ['Q76']
    assert denotations[3] == ['1972']
    assert denotations[4] == ['60655', '60601', '60827', '60601', '60827']
    assert denotations[6] == ['Q84']
    assert len(denotations[7]) == 3
    assert all(a in denotations[9] for a in {"Q21", "Q145"})
    assert 'Q36465' in denotations[10]
    assert len(denotations[12]) == 1

    for test_graph in test_graphs_without_groundings:
        result = graph_queries.get_graph_denotations(test_graph)
        assert len(result) == 0


def test_query_graph_topics():
    topics = []
    for test_graph in test_graphs_grounded:
        test_graph = stages.with_denotation_class_edge(test_graph)
        topic = graph_queries.get_graph_groundings(test_graph)
        topics.append(topic)
        assert len(topic) > 0
    assert 'Q82955' in {r['topic'] for r in topics[2]}
    assert {r['topic'] for r in topics[2]} == {r['topic'] for r in topics[11]}
    assert len({r['topic'] for r in topics[8]} & {'Q1622272', 'Q1028181'}) == 2
    assert topics[4] == [{'r1v': 'P31c', 'topic': 'Q37447'}]


if __name__ == '__main__':
    pytest.main(['-v', __file__])
