import pytest

import json

from questionanswering.construction.sentence import Sentence, SentenceEncoder, sentence_object_hook
from questionanswering.construction.graph import SemanticGraph, Edge, EdgeList


def test_encode():
    g = SemanticGraph([
        Edge(leftentityid="?qvar", rightentityid="Q76")
    ], free_entities=[
        {'linkings':[("Q37876", "Natalie Portman")], 'tokens':["Portman"], 'type':'PERSON'},
        {'linkings': [('2012', '2012')], 'type': 'YEAR', 'tokens': ['2012']}]
    )
    json_str = json.dumps(g, cls=SentenceEncoder, sort_keys=True)
    assert '"leftentityid": "?qvar", "qualifierentityid": null, "qualifierrelationid": null, "relationid": null, "rightentityid": "Q76"' in json_str
    assert '"free_entities": [{"linkings": [["Q37876", "Natalie Portman"]], "tokens": ["Portman"], ' in json_str

    s = Sentence(entities=[{"type": "NN", "linkings": [("Q5", "human")], 'token_ids': [0]}])
    json_str = json.dumps(s, cls=SentenceEncoder, sort_keys=True)
    assert '"entities": [{"linkings": [["Q5", "human"]], "token_ids": [0], "type": "NN"}],' in json_str
    assert ', [0.0, 0.0, 0.0]]]' in json_str


def test_decode():
    g = SemanticGraph([
        Edge(leftentityid="?qvar", rightentityid="Q76")
    ], free_entities=[
        {'linkings': [("Q37876", "Natalie Portman")], 'tokens':["Portman"], 'type':'PERSON'},
        {'linkings': [('2012', '2012')], 'type': 'YEAR', 'tokens': ['2012']}]
    )
    json_str = json.dumps(g, cls=SentenceEncoder, sort_keys=True)
    g_decoded = json.loads(json_str, object_hook=sentence_object_hook)
    assert len(g_decoded.edges) > 0
    assert isinstance(g_decoded.edges, EdgeList)
    assert g_decoded.edges[0].relationid is None

    s = Sentence(entities=[{"type": "NN", "linkings": [("Q5", "human")], 'token_ids': [0]}])
    json_str = json.dumps(s, cls=SentenceEncoder, sort_keys=True)
    s_decoded = json.loads(json_str, object_hook=sentence_object_hook)
    assert len(s_decoded.graphs) == 1
    assert s_decoded.graphs[0].scores[2] == 0.0


if __name__ == '__main__':
    pytest.main(['-v', __file__])
