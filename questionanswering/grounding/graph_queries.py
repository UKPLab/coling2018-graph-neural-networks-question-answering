import logging
import re
import itertools


from wikidata import scheme, endpoint_access, queries

from questionanswering.construction import graph, sentence
from questionanswering.construction.graph import SemanticGraph, Edge
from questionanswering._utils import RESOURCES_FOLDER, load_blacklist

QUESTION_VAR = "?qvar"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

EXPAND_TRANSITIVE_RELATIONS = True

sparql_relation_template = """
        {{ GRAPH g:statements {{ {triples} }} }}
    """

sparql_triple_template = {
    'left': "{left} {relationid}s ?m{edgeid} . ",
    'right': "?m{edgeid} {relationid}{branch} {right} . ",
    'left-to-right': "{left} {relationid}s/{relationid}v {right} {option}. ",
    'time': "?m{edgeid} {relationid}{branch} [base:time ?n{edgeid}] . ",
    'time-filter': "FILTER (YEAR(?n{edgeid}) = {right})"
}

sparql_class_relation = {
    "class": """
        {{   {{ GRAPH g:instances {{ {left} rdf:type {right}. }}}}
            UNION
            {{
                GRAPH g:statements {{ {left} e:P106s/e:P106v ?m{edgeid}. }} 
                GRAPH g:instances {{ ?m{edgeid} rdfs:subClassOf? {right}. }}
            }}
        }}
    """,
    "iclass": """
        {{VALUES ?r{edgeid}v {{ e:P106c e:P31c }} 
        GRAPH g:simple-statements {{ {left} ?r{edgeid}v ?topic. }} }}
    """
}

sparql_character_label = """
        {GRAPH <http://wikidata.org/terms> { ?e2 rdfs:label ?labelright }
        GRAPH <http://wikidata.org/statements> { ?e1 ?p0 ?m0 . ?m0 e:P453q ?e2 .  }
        FILTER CONTAINS(?labelright, %entitylabels)}
"""

sparql_restriction_time_argmax = "?m ?a [base:time ?n]. FILTER (YEAR(?n) = ?yearvalue)"

sparql_filter_main_entity = """
        BIND (STR(?e1) as ?e1s)
        FILTER(STRSTARTS(?e1s, "http://www.wikidata.org/entity/Q") && !CONTAINS(?e1s, "-"))
        """

HOP_UP_RELATIONS = load_blacklist(RESOURCES_FOLDER + "property_hopup.txt")
HOP_DOWN_RELATIONS = load_blacklist(RESOURCES_FOLDER + "property_hopdown.txt")
TRANSITIVE_RELATIONS = {"P131", "P361"}

LONG_LEG_RELATIONS = HOP_UP_RELATIONS | HOP_DOWN_RELATIONS | TRANSITIVE_RELATIONS

TEMPORAL_RELATIONS_Q = {"P585q", "P580q", "P582q", "P577q", "P571q"}
# TEMPORAL_RELATIONS_V = {"P580v", "P582v", "P577v", "P571v", "P569v", "P570v"}
QUALIFIER_RELATIONS = {"P1365q", "P812q", "P453q", "P175q"}
EXCEPTION_RELATIONS = QUALIFIER_RELATIONS | {"P281v"}

BLACK_LIST = {"P138", "P2348", "P530", "P279", "P180", "P669", "P197"}
CONTENT_PROPERTIES = scheme.content_properties - BLACK_LIST

FREQ_THRESHOLD = 500


def filter_relations(results, b='p', freq_threshold=0):
    """
    Takes results of a SPARQL query and filters out all rows that contain blacklisted relations.

    :param results: results of a SPARQL query
    :param b: the key of the relation value in the results dictionary
    :return: filtered results
    >>> filter_relations([{"p":"http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "e2":"http://www.wikidata.org/ontology#Item"}, {"p":"http://www.wikidata.org/entity/P1429s", "e2":"http://www.wikidata.org/entity/Q76S69dc8e7d-4666-633e-0631-05ad295c891b"}])
    []
    """
    results = [r for r in results if b not in r or
               (r[b] in EXCEPTION_RELATIONS or
                (r[b][:-1] in CONTENT_PROPERTIES and r[b][-1] not in endpoint_access.FILTER_RELATION_CLASSES))
               ]
    results = [r for r in results if b not in r or scheme.property2label[r[b][:-1]]['freq'] > freq_threshold]
    return results


def get_all_groundings(g: SemanticGraph):
    """
    Construct groudnings based on the wikidata scheme.

    :param g:
    :return:
    >>> len(get_all_groundings(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid='Q571', qualifierentityid='MAX')])))
    365
    >>> len(get_all_groundings(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid='Q571'), Edge(leftentityid=QUESTION_VAR, rightentityid='Q5')])))
    133225
    """
    variables = set()

    for i, edge in enumerate(g.edges):
        if not edge.grounded:
            variables.add(f"r{edge.edgeid:d}v")
    groundings = [[(v, r+"v") for r in scheme.frequent_properties] for v in variables]
    groundings = [dict(p) for p in itertools.product(*groundings)]
    return groundings


def get_graph_groundings(g: SemanticGraph, pass_exception=False, use_wikidata=True):
    """
    Convert the given graph to a WikiData query and retrieve the results. The results contain possible bindings
    for all free variables in the graph. If there are no free variables a single empty grounding is returned.

    :param g: graph as a dictionary
    :param pass_exception:
    :return: graph groundings encoded as a list of dictionaries
    >>> get_graph_groundings(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid='Q571', qualifierentityid='MAX')]))
    [{'r0v': 'P31v'}, {'r0v': 'P800v'}]
    >>> get_graph_groundings(SemanticGraph([Edge(leftentityid='Q35637', relationid='P1346', rightentityid=QUESTION_VAR, qualifierentityid='2009'), Edge(leftentityid=QUESTION_VAR, relationid='iclass')]))
    [{'r1v': 'P31c', 'topic': 'human'}, {'r1v': 'P106c', 'topic': 'politician'}]
    """
    ungrouded_edges = g.get_ungrounded_edges()
    if ungrouded_edges:
        if len(ungrouded_edges) == 1 and ungrouded_edges[0].relationid == "iclass":
            if "zip" in g.tokens and any(e.relationid == "P281" for e in g.edges):
                return [{'r1v': 'P31c', 'topic': 'Q37447'}]
            elif any([scheme.property2label[e.relationid]["type"] == "time"
                      for e in g.edges if e.leftentityid != QUESTION_VAR]):
                return [{'r1v': 'P31c', 'topic': "Q577"}]
        if use_wikidata:
            groundings = endpoint_access.query_wikidata(graph_to_query(g, limit=500))
        else:
            groundings = get_all_groundings(g)
        if groundings is None:  # If there was an exception
            return None if pass_exception else []
        elif len(groundings) > 0:
            # keys = {b for r in groundings for b in r if b.startswith("r")}
            for e in ungrouded_edges:
                groundings = filter_relations(groundings, b=f"r{e.edgeid:d}v", freq_threshold=FREQ_THRESHOLD)
            if sentence.get_question_type(" ".join(g.tokens)) != 'temporal':
                groundings = [r for r in groundings if all([scheme.property2label[r[f"r{e.edgeid:d}v"][:-1]]["type"] != "time"
                                                           for e in ungrouded_edges if e.leftentityid != QUESTION_VAR])]
        groundings = sorted(groundings,
                            key=lambda r: sum([scheme.property2label[r[f"r{e.edgeid:d}v"][:-1]]['freq']
                                               for e in ungrouded_edges if f"r{e.edgeid:d}v" in r]), reverse=True)
        return groundings
    else:
        if verify_grounding(g) or not use_wikidata:
            return [{}]
        else:
            return []


def verify_grounding(g: SemanticGraph):
    """
    Verify the given graph with (partial) grounding exists in Wikidata.

    :param g: graph as a dictionary
    :return: true if the graph exists, false otherwise
    >>> verify_grounding(SemanticGraph([Edge(leftentityid=QUESTION_VAR, rightentityid="Q76")]))
    True
    """
    # if len(filter_relations(g.edges, b='kbID')) < len(g.edges):
    #     return False
    if not sentence.get_question_type(" ".join(g.tokens)) == 'temporal' and \
            any([scheme.property2label.get(edge.relationid, {}).get("type") == "time"
                 for edge in g.edges if edge.leftentityid != QUESTION_VAR]):
        return False
    verified = endpoint_access.query_wikidata(graph_to_ask(g), timeout=1)
    if verified == []:
        return False
    return verified


def get_graph_denotations(g: SemanticGraph):
    """
    Convert the given graph to a WikiData query and retrieve the denotations of the graph. The results contain the
     list of the possible graph denotations

    :param g: graph as a SemanticGraph
    :return: graph denotations as a list of dictionaries
    >>> get_graph_denotations(SemanticGraph([Edge(leftentityid='Q35637', relationid='P1346', rightentityid=QUESTION_VAR, qualifierentityid='2009')]))
    ['Q76']
    >>> get_graph_denotations(SemanticGraph([Edge(leftentityid='Q37320', relationid='P131', rightentityid='?m0Q37320'), Edge(leftentityid='?m0Q37320', relationid='P421', rightentityid=QUESTION_VAR)]))
    ['Q941023', 'Q28146035']
    """
    qvar_name = QUESTION_VAR[1:]
    if "zip" in g.tokens and any(e.relationid == "P281" for e in g.edges):
        denotations = endpoint_access.query_wikidata(graph_to_query(g, limit=100))
        denotations = [r for r in denotations if any('x' not in r[b] for b in r)]  # Post process zip codes
        post_processed = []
        for r in denotations:
            codes = re.split("[-–]", r[qvar_name])
            for p in codes:
                if p:
                    if len(p) < len(codes[0]):
                        p = codes[0][:(len(codes[0])) - len(p)] + p
                    post_processed.append(p)
        return post_processed
    edges = [e for e in g.edges if e.rightentityid != "Q5"]  # filter out edges with human as argument since they often fail
    denotations = endpoint_access.query_wikidata(graph_to_query(SemanticGraph(edges=edges), limit=100))
    if denotations and all('step' in d for d in denotations):
        min_transitive_steps = min([d['step'] for d in denotations])
        denotations = [d for d in denotations if d['step'] == min_transitive_steps]
    denotations = list({d[qvar_name] for d in denotations if qvar_name in d})
    if not sentence.get_question_type(" ".join(g.tokens)) == 'temporal':
        denotations = filter_auxiliary_entities_by_id(denotations)  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    else:
        denotations = [l for _, labels in queries.get_labels_for_entities(denotations).items() for l in labels]
    return denotations


def filter_auxiliary_entities_by_id(denotations):
    """
    A safe net method that removes all auxiliary methods from the denotations.

    :param denotations: a list of entity ids
    :return: list of entitiy ids
    >>> filter_auxiliary_entities_by_id(['Q161-491', 'Q52-3651', 'Q114-3278', 'Q179385', 'Q592123', 'Q62378', 'Q617407', 'Q858775'])
    ['Q179385', 'Q592123', 'Q62378', 'Q617407', 'Q858775']
    >>> filter_auxiliary_entities_by_id([{'e1': 'Q161-491'}, {'e1': 'Q52P3651'}, {'e1': 'Q1143278'}, {'e1': 'Q179385'}, {'e1': 'Q592123'}, {'e1': 'Q62378'}])
    [{'e1': 'Q52P3651'}, {'e1': 'Q1143278'}, {'e1': 'Q179385'}, {'e1': 'Q592123'}, {'e1': 'Q62378'}]
    """
    qvar_name = QUESTION_VAR[1:]
    keep = endpoint_access.GLOBAL_RESULT_LIMIT
    if len(denotations) > 0 and type(denotations[0]) == dict:
        denotations = [r for r in denotations if any('-' not in r[b] and r[b][0] in 'pqPQ' for b in r)]
        if keep < len(denotations):
            denotations = sorted([k for k in denotations if k.get(qvar_name, ' ')[1:].isnumeric()], key=lambda k: int(k.get(qvar_name, ' ')[1:]))[:keep]
    else:
        denotations = [r for r in denotations if '-' not in r and r[0] in 'pqPQ']
        if keep < len(denotations):
            denotations = sorted([k for k in denotations if k[1:].isnumeric()], key=lambda k: int(k[1:]))[:keep]
    return denotations


def graph_to_select(g, **kwargs):
    return graph_to_query(g, ask=False, **kwargs)


def graph_to_ask(g, **kwargs):
    return graph_to_query(g, ask=True,  **kwargs)


def edge_to_sparql(edge: graph.Edge, expand_transitive=EXPAND_TRANSITIVE_RELATIONS):
    """
    Convert a graph edge to a piece of a SPARQL query.

    :param edge: input Edge
    :return: SPARQL piece as a string
    >>> edge_to_sparql(graph.Edge("Q76", None , QUESTION_VAR)).strip()
    '{ GRAPH g:statements { e:Q76 ?r0s ?m0 . ?m0 ?r0v ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge("Q76", None , None, 'P453', QUESTION_VAR)).strip()
    '{ GRAPH g:statements { e:Q76 ?r0s ?m0 . ?m0 e:P453q ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge(QUESTION_VAR,  None , None, 'P453', "Q76")).strip()
    '{ GRAPH g:statements { ?qvar ?r0s ?m0 . ?m0 e:P453q e:Q76 .  } }'
    >>> edge_to_sparql(graph.Edge("Q76", "P36" , QUESTION_VAR)).strip()
    '{ GRAPH g:statements { e:Q76 e:P36s/e:P36v ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge("?e1", "P36" , QUESTION_VAR)).strip()
    '{ GRAPH g:statements { ?e1 e:P36s/e:P36v ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge("?e1", "P131" , QUESTION_VAR)).strip()
    "{ GRAPH g:statements { ?e1 e:P131s/e:P131v ?qvar option (transitive,t_no_cycles, t_min (1), t_max(5), t_step ('step_no') as ?step).  } }"
    >>> edge_to_sparql(graph.Edge(None, None , "Q37876", None, QUESTION_VAR)).strip()
    '{ GRAPH g:statements { ?m0 ?r0v e:Q37876 . ?m0 ?r0q ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge(None, None, "Q37876", "P175", QUESTION_VAR)).strip()
    '{ GRAPH g:statements { ?m0 ?r0v e:Q37876 . ?m0 e:P175q ?qvar .  } }'
    >>> edge_to_sparql(graph.Edge(None, "P161", QUESTION_VAR, None, "Q37876")).strip()
    '{ GRAPH g:statements { ?m0 e:P161v ?qvar . ?m0 ?r0q e:Q37876 .  } }'
    >>> edge_to_sparql(graph.Edge("Q678", None, QUESTION_VAR, None, "2009")).strip()
    '{ GRAPH g:statements { e:Q678 ?r0s ?m0 . ?m0 ?r0v ?qvar . ?m0 ?r0q [base:time ?n0] . FILTER (YEAR(?n0) = 2009) } }'
    >>> edge_to_sparql(graph.Edge("Q678", "P89", QUESTION_VAR, "P453", "Q896")).strip()
    '{ GRAPH g:statements { e:Q678 e:P89s ?m0 . ?m0 e:P89v ?qvar . ?m0 e:P453q e:Q896 .  } }'
    >>> edge_to_sparql(graph.Edge(QUESTION_VAR, None, "Q678", None, "2009")).strip()
    '{ GRAPH g:statements { ?qvar ?r0s ?m0 . ?m0 ?r0v e:Q678 . ?m0 ?r0q [base:time ?n0] . FILTER (YEAR(?n0) = 2009) } }'
    >>> edge_to_sparql(graph.Edge(QUESTION_VAR, None, "2009")).strip()
    '{ GRAPH g:statements { ?qvar ?r0s ?m0 . ?m0 ?r0v [base:time ?n0] . FILTER (YEAR(?n0) = 2009) } }'
    >>> edge_to_sparql(graph.Edge("Q678", None, None, None, "MAX")).strip()
    '{ GRAPH g:statements { e:Q678 ?r0s ?m0 . ?m0 ?r0q [base:time ?n0] .  } }'
    >>> edge_to_sparql(graph.Edge("Q678", None, "MAX")).strip()
    '{ GRAPH g:statements { e:Q678 ?r0s ?m0 . ?m0 ?r0v [base:time ?n0] .  } }'
    >>> edge_to_sparql(graph.Edge(QUESTION_VAR, "class", "Q5",)).strip()
    '{   { GRAPH g:instances { ?qvar rdf:type e:Q5. }}\\n            UNION\\n            {\\n                GRAPH g:statements { ?qvar e:P106s/e:P106v ?m0. } \\n                GRAPH g:instances { ?m0 rdfs:subClassOf? e:Q5. }\\n            }\\n        }'
    >>> edge_to_sparql(graph.Edge(QUESTION_VAR, "iclass")).strip()
    '{VALUES ?r0v { e:P106c e:P31c } \\n        GRAPH g:simple-statements { ?qvar ?r0v ?topic. } }'
    """
    relationid = f"e:{edge.relationid}" if edge.relationid is not None else f"?r{edge.edgeid:d}"

    values = {
        'edgeid': edge.edgeid,
        'left': f"e:{edge.leftentityid}" if edge.leftentityid and edge.leftentityid.startswith("Q") else edge.leftentityid,
        'relationid': relationid,
        'right': f"e:{edge.rightentityid}" if edge.rightentityid and edge.rightentityid.startswith("Q") else edge.rightentityid,
        'option': ""
    }
    if edge.relationid in sparql_class_relation:
        return sparql_class_relation[edge.relationid].format(**values)

    triples = []
    if edge.simple:
        if edge.relationid in TRANSITIVE_RELATIONS and expand_transitive:
            values['option'] = queries.sparql_transitive_option
        triples.append(sparql_triple_template['left-to-right'].format(**values))
    else:
        if edge.leftentityid is not None:
            triples.append(sparql_triple_template['left'].format(**values))
        if edge.rightentityid is not None:
            template = sparql_triple_template['right']
            if values['right'].isdigit():
                template = sparql_triple_template['time'] + sparql_triple_template['time-filter']
            elif values['right'] in {"MAX", "MIN"}:
                template = sparql_triple_template['time']
            triples.append(template.format(**{**values, "branch": 'v'}))
        if edge.qualifierentityid is not None:
            relationid = f"e:{edge.qualifierrelationid}" if edge.qualifierrelationid is not None else f"?r{edge.edgeid:d}"
            right = f"e:{edge.qualifierentityid}" if edge.qualifierentityid.startswith("Q") else edge.qualifierentityid
            template = sparql_triple_template['right']
            if right.isdigit():
                template = sparql_triple_template['time'] + sparql_triple_template['time-filter']
            elif right in {"MAX", "MIN"}:
                template = sparql_triple_template['time']
            triples.append(template.format(**{**values,
                                              "branch": 'q',
                                              "relationid": relationid,
                                              "right": right}))

    return sparql_relation_template.format(triples="".join(triples))


def graph_to_query(g: SemanticGraph, ask=False, limit=endpoint_access.GLOBAL_RESULT_LIMIT):
    """
    Convert graph to a SPARQL query.

    :param ask: if the a simple existence of the graph should be checked instead of returning variable values.
    :param g: a graph as a dictionary with non-empty edgeSet
    :param return_var_values: if True the denotations for free variables will be returned
    :param limit: limit on the result list size
    :return: a SPARQL query as a string
    >>> print(graph_to_query(SemanticGraph(edges=[graph.Edge(0, "Q76", None , QUESTION_VAR)]) ))

    """
    variables = set()
    order_by = []
    edges = []

    for i, edge in enumerate(g.edges):
        edges.append(edge_to_sparql(edge, expand_transitive=not ask))
        if not edge.grounded:
            variables.add(f"?r{edge.edgeid:d}v")
        if edge.qualifierentityid in {'MAX', 'MIN'}:
            order_by.append(f"{'DESC' if edge.qualifierentityid=='MAX' else 'ASC'}(?n{edge.edgeid:d})")
        if edge.relationid == 'iclass':
            variables.add("?topic")
        if edge.relationid == 'iclass':
            variables.add("?topic")
        if edge.simple and edge.relationid in TRANSITIVE_RELATIONS and not ask\
                and QUESTION_VAR not in edge.nodes():
            variables.add("?step")

    query = queries.sparql_prefix + (
        queries.sparql_select if not ask else queries.sparql_ask)
    if any(edge.relationid == 'class' for edge in g.edges):
        query = queries.sparql_inference_clause + query
    order_by_pattern = ""
    if len(variables - {'?step'}) == 0 and not ask:
        variables.add(QUESTION_VAR)
        if order_by:
            order_by_pattern = queries.sparql_close_order.format(" ".join(order_by))
            limit = 1
    else:
        variables = variables - {'?step'}
    query = query.format(queryvariables=" ".join(variables))
    query += "{{ {} }}".format('\n'.join(edges))
    if not ask:
        query += order_by_pattern + queries.sparql_close.format(limit)

    return query


def character_query(label, film_id, limit=3):
    """
    Depricated!
    A method to look up a WikiData film character by a label.

    :param label: label of the entity as str
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    >>> query_wikidata(character_query("Bella", "Q160071"), starts_with=None) == \
    [{'labelright': 'Bella Swan', 'e2': 'http://www.wikidata.org/entity/Q223757', 'label': 'Bella Swan'}]
    True
    >>> query_wikidata(character_query("Anakin", "Q42051"), starts_with=None) == \
    [{'labelright': 'Anakin Skywalker', 'e2': 'http://www.wikidata.org/entity/Q51752', 'label': 'Anakin Skywalker'}, {'labelright': 'Anakin Skywalker', 'e2': 'http://www.wikidata.org/entity/Q51752', 'label': 'Anakin Skywalker'}]
    True
    """
    query = queries.sparql_prefix
    variables = []
    query += queries.sparql_select
    query += "{"
    sparql_entity_label_inst = sparql_character_label + queries.sparql_get_main_entity_label
    sparql_entity_label_inst = sparql_entity_label_inst.replace("?e1", "e:{}".format(film_id))
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%entitylabels", "\"{}\"".format(label, label))
    variables.append("?e2")
    variables.append("?labelright")
    variables.append("?label")
    query += sparql_entity_label_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += queries.sparql_close.format(limit)
    logger.debug("Querying for entity with variables: {}".format(variables))
    return query


def label_query_results(query_results):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> sorted(sorted(label_query_results(['Q76', 'Q235234', 'r68123123-12dd222']))[0])  # doctest: +ELLIPSIS
    ['barack h. obama', 'barack hussein obama', ...]
    >>> label_query_results(['10000', '10499', '11004', '05',  ""])
    [['10000'], ['10499'], ['11004'], ['05']]
    """
    answers_to_label = {a for a in query_results if not a.isnumeric() and len(a) > 0}
    rest_answers = [[a] for a in query_results if a.isnumeric()]
    answers = [[l.lower() for l in labels] for _, labels in queries.get_labels_for_entities(answers_to_label).items()]
    answers = normalize_answer_strings(answers)
    return answers + rest_answers


def normalize_answer_strings(answers):
    """
    Add normalized alternative labels.

    :param answers: list of lists of string answers
    :return: list of lists of string answers
    >>> normalize_answer_strings([['twilight saga: breaking dawn - part 2'], ['the twilight saga: new moon', 'twilight saga: new moon']])
    [['twilight saga: breaking dawn - part 2', 'twilight saga', 'breaking dawn - part 2', 'twilight saga: breaking dawn', 'part 2', 'breaking dawn'], ['the twilight saga: new moon', 'twilight saga: new moon', 'twilight saga', 'the twilight saga', 'new moon']]
    >>> normalize_answer_strings([['2010 world series', 'world series 2010'], ['2012 world series', 'world series 2012'], ['world series 2014', '2014 world series']])
    [{'2010 world series', 'world series 2010'}, {'2012 world series', 'world series 2012'}]
    >>> normalize_answer_strings([['liste gegenwärtig amtierender staatsoberhäupter nach amtszeiten', 'list of heads of state by diplomatic precedence']])
    []
    >>> normalize_answer_strings([["eberhard-karls-gymnasium"]])
    [{'eberhard-karls-gymnasium', 'eberhard karls gymnasium'}]
    >>> normalize_answer_strings([["brown hair"]])
    [{'brown hair', 'brown'}]
    >>> normalize_answer_strings([["ngurah rai airport"]])
    [{'ngurah rai airport', 'ngurah rai international airport'}]
    """
    answers = [{a.replace("–", "-").lower() for a in answer_set} for answer_set in answers]
    new_answers = []
    for answer_set in answers:
        new_answer_list = set(answer_set)
        for a in answer_set:
            if ":" in a:
                new_answer_list |= {w.strip() for w in a.split(":")}
            if "," in a:
                new_answer_list |= {w.strip() for w in a.split(",")}
            if " - " in a:
                new_answer_list |= {w.strip() for w in a.split(" - ")}
            elif "-" in a:
                new_answer_list.add(a.replace("-", " "))
            if "standard time" in a:
                new_answer_list.add(a.replace("standard time", "time zone"))
            if "airport" in a:
                new_answer_list.add(a.replace("airport", "international airport"))
            if any(re.findall("\\b{}\\b".format(t), a) for t in {'hair', 'color'}):
                new_answer_list.add(" ".join([t for t in a.split() if t not in {'hair', 'color'}]))
        if not any(re.search("\\b(2014|2015|2016|2017|list of)\\b", a) for a in answer_set):
            new_answers.append(new_answer_list)
    return new_answers


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
