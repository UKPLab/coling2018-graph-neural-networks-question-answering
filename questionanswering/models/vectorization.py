import numpy as np

from typing import List

from collections import defaultdict

from questionanswering import _utils
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering.construction.sentence import Sentence
from questionanswering.grounding import graph_queries, stages
from wikidata import scheme

ENTITY_TOKEN = "<e>"
SPECIAL_TOKENS = {
    "MAX": "<max>",
    "MIN": "<min>",
    "YEAR": "<year>"
}
SENT_TOKENS = ["<s>", "<f>"]

MAX_LABEL_TOKEN_LEN = 20
MAX_EDGES = 7
MAX_EDGES_PER_ENTITY = 4
MAX_NEGATIVE_GRAPHS = 100

WORD_2_IDX = None


def encode_for_model(selected_questions, model_type, word2idx=None):
    assert word2idx or WORD_2_IDX
    if not word2idx:
        word2idx = WORD_2_IDX
    samples = {
        "OneEdgeModel": lambda: (encode_batch_questions(selected_questions, word2idx)[..., 0, :],
                                 encode_batch_graphs(selected_questions, word2idx)[..., 0, 0, :]),
        "STAGGModel": lambda: (encode_batch_questions(selected_questions, word2idx),
                               encode_batch_graphs(selected_questions, word2idx)[..., 0, :, :],
                               encode_structural_features(selected_questions)),
        "PooledEdgesModel": lambda: (encode_batch_questions(selected_questions, word2idx)[..., 1, :],
                                     encode_batch_graphs(selected_questions, word2idx)[..., 1, :]),
        "GNNModel": lambda: (encode_batch_questions(selected_questions, word2idx)[..., 1, :],
                             *encode_batch_graph_structure(selected_questions, word2idx))
    }[model_type]()
    return samples


def extend_embeddings_with_special_tokens(embeddings, word2idx):
    for el in SPECIAL_TOKENS.values():
        word2idx[el] = len(word2idx)
    for el in SENT_TOKENS:
        word2idx[el] = len(word2idx)
    word2idx[ENTITY_TOKEN] = len(word2idx)
    std = np.std(embeddings)
    mean = np.mean(embeddings)
    embeddings = np.concatenate((embeddings,
                                 std*np.random.randn(len(word2idx) - embeddings.shape[0], embeddings.shape[1]))+mean,
                                axis=0)
    return embeddings, word2idx


def encode_batch_graphs(questions: List[Sentence], vocab):
    max_negative_graphs = min(max(len(s.graphs) for s in questions), MAX_NEGATIVE_GRAPHS)
    out = np.zeros((len(questions), max_negative_graphs, MAX_EDGES, 2, MAX_LABEL_TOKEN_LEN), dtype=np.int32)
    for i, s in enumerate(questions):  # Iterate over lists of graphs for questions
        entity2label = {k: l for e in s.entities for k, l in e['linkings']}
        entity2type = {k: e['type'] for e in s.entities for k, l in e['linkings']}
        for gi, g in enumerate(s.graphs[:max_negative_graphs]):  # Iterate over graph alternatives for a question
                main_edges = [e for e in g.graph.edges
                              if graph_queries.QUESTION_VAR in e.nodes()
                              and e.relationid not in graph_queries.sparql_class_relation] \
                             + [e for e in g.graph.edges if e.relationid in graph_queries.sparql_class_relation]
                for ei, e in enumerate(main_edges[:MAX_EDGES]):
                    word_ids = [vocab[w.lower()]
                                for w in _get_edge_str_representation(e, entity2label, entity2type,
                                                                      replace_entities=True,
                                                                      mark_boundaries=True)][:MAX_LABEL_TOKEN_LEN]
                    out[i, gi, ei, 0, :len(word_ids)] = word_ids
                    word_ids = [vocab[w.lower()]
                                for w in _get_edge_str_representation(e, entity2label, entity2type,
                                                                      replace_entities=False,
                                                                      mark_boundaries=True)][:MAX_LABEL_TOKEN_LEN]
                    out[i, gi, ei, 1, :len(word_ids)] = word_ids
    return out


def encode_batch_questions(questions: List[Sentence], vocab):
    out = np.zeros((len(questions), 2,  MAX_LABEL_TOKEN_LEN), dtype=np.int32)
    for i, s in enumerate(questions):  # Iterate over lists of graphs for questions
        word_ids = [vocab[w.lower()] for w in _get_sentence_tokens(s, replace_entities=True, mark_boundaries=True)][:MAX_LABEL_TOKEN_LEN]
        out[i, 0, :len(word_ids)] = word_ids
        word_ids = [vocab[w.lower()] for w in _get_sentence_tokens(s, replace_entities=False, mark_boundaries=True)][:MAX_LABEL_TOKEN_LEN]
        out[i, 1, :len(word_ids)] = word_ids
    return out


def encode_structural_features(questions: List[Sentence]):
    max_negative_graphs = min(max(len(s.graphs) for s in questions), MAX_NEGATIVE_GRAPHS)
    out = np.zeros((len(questions), max_negative_graphs, 7), dtype=np.int32)
    for i, s in enumerate(questions):  # Iterate over lists of graphs for questions
        tokens = {t.lower() for t in s.tokens}
        for j, g in enumerate(s.graphs[:max_negative_graphs]):  # Iterate over graph alternatives for a question
            g = g.graph
            out[i, j, 0] = len(g.edges)  # Num edges
            out[i, j, 1] = len(g.denotations)  # Num answers
            out[i, j, 2] = any(k in tokens for k in stages.argmax_markers)  # AggregationKeyword
            out[i, j, 3] = any(k in tokens for k in stages.argmin_markers)  # AggregationKeyword
            out[i, j, 4] = any("MAX" in e.nodes() for e in g.edges)  # Argmax marker
            out[i, j, 5] = any("MIN" in e.nodes() for e in g.edges)  # Argmin marker
            out[i, j, 6] = any(e.temporal for e in g.edges)  # Temporal marker
    return out


def _get_sentence_tokens(s: Sentence, replace_entities=True, mark_boundaries=False):
    """
    Get the list of sentence tokens.

    :param s: a sentence object
    :param replace_entities: optionally replace the entities with a placeholder.
    :param mark_boundaries: optionally add sentence start and end markers.
    :return: a list of tokens
    >>> _get_sentence_tokens(Sentence(tagged=[{'originalText': k, 'pos': 'O', 'ner': 'O'} for k in "Who killed Lora Palmer ?".split()], entities=[{"type": "NNP", 'linkings': [], 'token_ids': [2,3]}]))
    ['Who', 'killed', '<e>', '?']
    >>> _get_sentence_tokens(Sentence(tagged=[{'originalText': k, 'pos': 'O', 'ner': 'O'} for k in "Who killed Lora Palmer ?".split()], entities=[{"type": "NN", 'linkings': [], 'token_ids': [0]}]))
    ['Who', 'killed', 'Lora', 'Palmer', '?']
    >>> _get_sentence_tokens(Sentence(tagged=[{'originalText': k, 'pos': 'O', 'ner': 'O'} for k in "where are the nfl redskins from ?".split()], entities=[{'linkings': [['Q212654', None]], 'token_ids': [4], 'type': 'NNP'}, {'linkings': [['Q1215884', None]], 'token_ids': [3], 'type': 'NNP'}, {'linkings': [['Q618123', 'geographical object']], 'token_ids': [0], 'type': 'NN'}]))
    ['where', 'are', 'the', '<e>', '<e>', 'from', '?']
    >>> _get_sentence_tokens(Sentence(tagged=[{'originalText': k, 'pos': 'O', 'ner': 'O'} for k in "who won the prise 2009 ?".split()], entities=[{'linkings': [['2009', '2009']], 'token_ids': [4], 'type': 'YEAR'}]))
    ['who', 'won', 'the', 'prise', '<year>', '?']
    """
    sentence_tokens = s.tokens
    if replace_entities:
        updated_tokens = []
        right = 0
        entities = sorted(s.entities, key=lambda x: x['token_ids'][0] if x['token_ids'] else 0)
        for e in entities:
            if e['token_ids'] and e['type'] != "NN":
                left = min(e['token_ids'])
                entity_token = SPECIAL_TOKENS['YEAR'] if e['type'] == "YEAR" else ENTITY_TOKEN
                updated_tokens += sentence_tokens[right:left] + [entity_token]
                right = max(e['token_ids']) + 1
        if right < len(sentence_tokens):
            updated_tokens += sentence_tokens[right:]
        sentence_tokens = updated_tokens
    if mark_boundaries:
        sentence_tokens = SENT_TOKENS[0:1] + sentence_tokens + SENT_TOKENS[1:2]
    return sentence_tokens


def _get_edge_str_representation(edge: Edge, entity2label, entity2type,
                                 replace_entities=True,
                                 mark_boundaries=False,
                                 no_entity=False):
    """
    >>> _get_edge_str_representation(Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid="Q5", relationid="P175"), {"Q5": "human"}, {"Q5": "NN"})
    ['performer', 'human']
    >>> _get_edge_str_representation(Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid="MAX", relationid="P585"), {}, {})
    ['point', 'in', 'time', '<max>']
    >>> _get_edge_str_representation(Edge(leftentityid=graph_queries.QUESTION_VAR, rightentityid="MAX", relationid="P36", qualifierrelationid="P585"), {}, {}, no_entity=True, mark_boundaries=True)
    ['<s>', 'capital', 'point', 'in', 'time', '<f>']
    """
    property_label = [""]
    p_meta = scheme.property2label.get(edge.relationid)
    if p_meta:
        property_label = _utils.split_pattern.split(p_meta['label'])
    p_meta = scheme.property2label.get(edge.qualifierrelationid)
    if p_meta:
        property_label += _utils.split_pattern.split(p_meta['label'])
    entity_kbids = [n for n in edge.nodes() if n and n != graph_queries.QUESTION_VAR]
    if any(entity_kbids) and not no_entity:
        entity_kbid = entity_kbids[0]
        property_label += _entity_kbid2token(entity_kbid, entity2label, entity2type, replace_entities, mark_boundaries=False)
    if mark_boundaries:
        property_label = SENT_TOKENS[0:1] + property_label + SENT_TOKENS[1:2]
    return property_label


def _entity_kbid2token(entity_kbid, entity2label, entity2type, replace_entities, mark_boundaries=False, resolve_m=True):
    if entity_kbid in {"MIN", "MAX"}:
        tokens = [SPECIAL_TOKENS[entity_kbid]]
    elif entity2type.get(entity_kbid) == "YEAR":
        tokens = [SPECIAL_TOKENS["YEAR"]]
    else:
        if entity_kbid.startswith("?"):
            if not resolve_m:
                return []
            entity_kbid = entity_kbid[3:]
        entity_label = entity2label[entity_kbid]
        if (entity2type.get(entity_kbid) == "NN" or not replace_entities) and entity_label:
            tokens = _utils.split_pattern.split(entity_label)
        else:
            tokens = [ENTITY_TOKEN]
    if mark_boundaries:
        tokens = SENT_TOKENS[0:1] + tokens + SENT_TOKENS[1:2]
    return tokens


def encode_batch_graph_structure(questions: List[Sentence], vocab):
    max_negative_graphs = min(max(len(s.graphs) for s in questions), MAX_NEGATIVE_GRAPHS)

    out_nodes = np.zeros((len(questions), max_negative_graphs, MAX_EDGES, MAX_LABEL_TOKEN_LEN//2), dtype=np.int32)
    out_edges = np.zeros((len(questions), max_negative_graphs, MAX_EDGES, MAX_LABEL_TOKEN_LEN//2), dtype=np.int32)

    out_A_nodes = np.zeros((len(questions), max_negative_graphs, MAX_EDGES, MAX_EDGES_PER_ENTITY), dtype=np.uint8)
    out_A_edges = np.zeros((len(questions), max_negative_graphs, MAX_EDGES, MAX_EDGES_PER_ENTITY), dtype=np.uint8)

    for i, s in enumerate(questions):  # Iterate over lists of graphs for questions
        entity2label = {k: l for e in s.entities for k, l in e['linkings']}
        entity2type = {k: e['type'] for e in s.entities for k, l in e['linkings']}

        for gi, g in enumerate(s.graphs[:max_negative_graphs]):  # Iterate over graph alternatives for a question
            edges = [e for e in g.graph.edges
                     if e.relationid not in graph_queries.sparql_class_relation] \
                         + [e for e in g.graph.edges if e.relationid in graph_queries.sparql_class_relation]

            # edges = edges[:(MAX_EDGES - 2)]
            nodes = {n for e in edges for n in e.nodes() if n} - {graph_queries.QUESTION_VAR}
            nodes = list(nodes)[:(MAX_EDGES - 2)]  # The first row in the matrix is 0 padding and the second is the Qvar
            node2id = {n: ni for ni, n in enumerate(nodes, start=2)}

            for n, ni in node2id.items():
                entity_tokens = _entity_kbid2token(n, entity2label, entity2type,
                                                   replace_entities=False,
                                                   mark_boundaries=False, resolve_m=False)
                if entity_tokens:
                    word_ids = [vocab[w.lower()] for w in entity_tokens][:MAX_LABEL_TOKEN_LEN//2]
                    out_nodes[i, gi, ni, :len(word_ids)] = word_ids
                else:
                    out_nodes[i, gi, ni, 0] = vocab[ENTITY_TOKEN.lower()]
            node2id[graph_queries.QUESTION_VAR] = 1
            out_nodes[i, gi, 1] = 1

            temp_edges = defaultdict(list)
            temp_nodes = defaultdict(list)
            for ei, e in enumerate(edges, start=1):
                property_tokens = _get_edge_str_representation(e, entity2label, entity2type,
                                                               mark_boundaries=False,
                                                               no_entity=True)
                if property_tokens:
                    word_ids = [vocab[w.lower()] for w in property_tokens][:MAX_LABEL_TOKEN_LEN//2]
                    out_edges[i, gi, ei, :len(word_ids)] = word_ids

                if e.leftentityid in node2id:
                    if e.rightentityid in node2id:
                        temp_edges[node2id[e.leftentityid]].append(ei)
                        temp_nodes[node2id[e.leftentityid]].append(node2id[e.rightentityid])
                        temp_edges[node2id[e.rightentityid]].append(ei + MAX_EDGES)
                        temp_nodes[node2id[e.rightentityid]].append(node2id[e.leftentityid])
                    if e.qualifierentityid in node2id:
                        temp_edges[node2id[e.leftentityid]].append(ei)
                        temp_nodes[node2id[e.leftentityid]].append(node2id[e.qualifierentityid])
                        temp_edges[node2id[e.qualifierentityid]].append(ei + MAX_EDGES)
                        temp_nodes[node2id[e.qualifierentityid]].append(node2id[e.leftentityid])
                if e.rightentityid in node2id and e.qualifierentityid in node2id:
                    temp_edges[node2id[e.rightentityid]].append(ei)
                    temp_nodes[node2id[e.rightentityid]].append(node2id[e.qualifierentityid])
                    temp_edges[node2id[e.qualifierentityid]].append(ei + MAX_EDGES)
                    temp_nodes[node2id[e.qualifierentityid]].append(node2id[e.rightentityid])

            for nodeid, nodeids in temp_nodes.items():
                nodeids = nodeids[:MAX_EDGES_PER_ENTITY]
                out_A_nodes[i, gi, nodeid, :len(nodeids)] = nodeids
            for nodeid, edgeids in temp_edges.items():
                edgeids = edgeids[:MAX_EDGES_PER_ENTITY]
                out_A_edges[i, gi, nodeid, :len(edgeids)] = edgeids
    return out_nodes, out_edges, out_A_nodes, out_A_edges
