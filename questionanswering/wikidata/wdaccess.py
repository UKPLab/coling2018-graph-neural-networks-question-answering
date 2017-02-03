import logging
import re
from collections import defaultdict

from SPARQLWrapper import SPARQLWrapper, JSON

from construction import graph
from utils import load_blacklist, load_property_labels, load_entity_map, RESOURCES_FOLDER

WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"

wdaccess_p = {
    'wikidata_url': "http://knowledgebase:8890/sparql",
    'timeout': 20,
    'global_result_limit': 2000,
    'logger': logging.getLogger(__name__),
    'restrict.hop': False
}

logger = wdaccess_p['logger']
logger.setLevel(logging.ERROR)


def sparql_init():
    global sparql
    sparql = SPARQLWrapper(wdaccess_p.get('wikidata_url', "http://knowledgebase:8890/sparql"))
    sparql.setReturnFormat(JSON)
    sparql.setMethod("GET")
    sparql.setTimeout(wdaccess_p.get('timeout', 40))

sparql = None
sparql_init()
GLOBAL_RESULT_LIMIT = wdaccess_p['global_result_limit']


sparql_prefix = """
        PREFIX e:<http://www.wikidata.org/entity/>
        PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
        PREFIX base:<http://www.wikidata.org/ontology#>
        """
sparql_select = """
        SELECT DISTINCT %queryvariables% WHERE
        """

sparql_ask = """
        ASK WHERE
        """

sparql_relation = {
    "direct": """
        {GRAPH <http://wikidata.org/statements> { ?e1 ?p ?m . ?m ?rd ?e2 . %restriction% }}
    """,
    "reverse": """
        {GRAPH <http://wikidata.org/statements> { ?e2 ?p ?m . ?m ?rr ?e1 . %restriction% }}
    """,
    "v-structure": """
        {VALUES ?rv { e:P161v e:P453q e:P175q}  GRAPH <http://wikidata.org/statements> { ?m ?p ?e2 . ?m ?rv ?e1 . %restriction% }}
    """,
    "time": """
        {GRAPH <http://wikidata.org/statements> { ?e1 ?o0 [ ?a [base:time ?n]]. }}
    """
}

sparql_relation_complex = """
    {
    """ + sparql_relation['direct'] + """
    UNION
    """ + sparql_relation['reverse'] + """
    }
    """

sparql_entity_label = """
        { {VALUES ?labelright { %entitylabels }
          VALUES ?labelpredicate {rdfs:label skos:altLabel}
          GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate ?labelright  }}
          UNION
          {  VALUES ?labelright { %entitylabels }
          GRAPH <http://wikidata.org/statements> { ?e2 e:P1549s [ e:P1549v ?labelright ] } }
        } FILTER NOT EXISTS {
            VALUES ?topic {e:Q4167410 e:Q21286738 e:Q11266439 e:Q13406463 e:Q4167836}
            GRAPH <http://wikidata.org/instances> {?e2 rdf:type ?topic}}
        """

sparql_label_entity = """
        {
        VALUES ?e2 { %entityids }
        VALUES ?labelpredicate {rdfs:label skos:altLabel}
        GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate ?label }
        }
        """

sparql_character_label = """
        {GRAPH <http://wikidata.org/terms> { ?e2 rdfs:label ?labelright }
        GRAPH <http://wikidata.org/statements> { ?e1 ?p0 ?m0 . ?m0 e:P453q ?e2 .  }
        FILTER CONTAINS(?labelright, %entitylabels)}
"""

sparql_year_entity = """
        {
        VALUES ?e2 { %entityids }
        GRAPH <http://wikidata.org/statements> { ?e2 base:time ?et. BIND (YEAR(?et) AS ?label) }
        }
        """

sparql_canoncial_label_entity = """
        {
        GRAPH <http://wikidata.org/terms> { ?e2 rdfs:label ?label }
        FILTER ( lang(?label) = "en" )
        }
        """

sparql_get_demonym = """
        {
        GRAPH <http://wikidata.org/statements> { ?e2 e:P1549s [ e:P1549v ?labelright ] }
        FILTER ( lang(?labelright) = "en" )
        }
        """

entity_map = load_entity_map(RESOURCES_FOLDER + "manual_entity_map.tsv")
property_blacklist = load_blacklist(RESOURCES_FOLDER + "property_blacklist.txt")
entity_blacklist = load_blacklist(RESOURCES_FOLDER + "entity_blacklist.txt")
property_whitelist = load_blacklist(RESOURCES_FOLDER + "property_whitelist.txt")
property2label = load_property_labels(RESOURCES_FOLDER + "properties_with_labels.txt")


sparql_restriction_time_argmax = "?m ?a [base:time ?n]. FILTER (YEAR(?n) = ?yearvalue)"

sparql_relation_filter = 'FILTER NOT EXISTS { GRAPH <http://wikidata.org/properties> {%relationvar% rdf:type base:Property}}'

sparql_close_order = " ORDER BY {}"
sparql_close = " LIMIT {}"

# TODO: Additional?: given name
HOP_UP_RELATIONS = load_blacklist(RESOURCES_FOLDER + "property_hopup.txt") # {"P131", "P31", "P279", "P17", "P361", "P1445", "P179"} # + P674 Depricated
HOP_DOWN_RELATIONS = load_blacklist(RESOURCES_FOLDER + "property_hopdown.txt") # {"P131", "P31", "P279", "P17", "P361", "P1445", "P179"} # + P674 Depricated
TEMPORAL_RELATIONS_Q = {"P585q", "P580q", "P582q", "P577q", "P571q"}
TEMPORAL_RELATIONS_V = {"P580v", "P582v", "P577v", "P571v", "P569v", "P570v"}
TEMPORAL_RELATIONS = TEMPORAL_RELATIONS_Q | TEMPORAL_RELATIONS_V

sparql_entity_abstract = "?e3 ?hops [ ?hopv ?e2]."
sparql_entity_specify = " ?e2 ?hops [ ?hopv ?e3]. "
sparql_hopup_values = ""
sparql_hopdown_values = ""
sparql_temporal_values_q = "VALUES ?a {" + " ".join(["e:{}".format(r) for r in TEMPORAL_RELATIONS_Q]) + "}"
sparql_temporal_values_v = "VALUES ?a {" + " ".join(["e:{}".format(r) for r in TEMPORAL_RELATIONS_V]) + "}"

FILTER_ENDINGS = "r"


def update_sparql_clauses():
    global sparql_hopup_values
    global sparql_hopdown_values
    if wdaccess_p.get('restrict.hop'):
        sparql_hopup_values = "VALUES (?hops ?hopv) {" + " ".join(["(e:{}s e:{}v)".format(r, r) for r in HOP_UP_RELATIONS]) + "}"
        sparql_hopdown_values = "VALUES (?hops ?hopv) {" + " ".join(["(e:{}s e:{}v)".format(r, r) for r in HOP_DOWN_RELATIONS]) + "}"


def query_graph_groundings(g, use_cache=False, with_denotations=False, pass_exception=False):
    """
    Convert the given graph to a WikiData query and retrieve the results. The results contain possible bindings
    for all free variables in the graph. If there are no free variables a single empty grounding is returned.

    :param g: graph as a dictionary
    :param use_cache
    :return: graph groundings encoded as a list of dictionaries
    >>> len(query_graph_groundings({'edgeSet': [{'right': ['book'], 'rightkbID': 'Q571', 'type':'direct', 'argmax':'time'}], 'entities': []}))
    3
    >>> len(query_graph_groundings({'edgeSet': [{'rightkbID': 'Q127367', 'type':'reverse'}, {'type':'time'}], 'entities': []}))
    23
    >>> len(query_graph_groundings({'edgeSet': [{'rightkbID': 'Q127367', 'type':'reverse'}, {'type':'time', 'argmax':'time'}], 'entities': []}))
    23
    >>> len(query_graph_groundings({'edgeSet': [{'rightkbID': 'Q37876', 'type':'v-structure'}], 'entities': []}))
    2
    """
    if get_free_variables(g):
        groundings = query_wikidata(graph_to_query(g, limit=GLOBAL_RESULT_LIMIT*(10 if with_denotations else 1), return_var_values=with_denotations), use_cache=use_cache)
        if groundings is None:  # If there was an exception
            return None if pass_exception else []
        groundings = [r for r in groundings if not any(r[b][:-1] in property_blacklist or r[b][-1] in FILTER_ENDINGS for b in r)]
        question_text = " ".join(g.get('tokens', []))
        if not question_text.startswith("when") and not question_text.startswith("what year"):
            groundings = [r for r in groundings if not any(r[b] in TEMPORAL_RELATIONS for b in r)]
        return groundings
    return [{}]


def query_graph_denotations(g):
    """
    Convert the given graph to a WikiData query and retrieve the denotations of the graph. The results contain the
     list of the possible graph denotations

    :param g: graph as a dictionary
    :return: graph denotations as a list of dictionaries
    >>> query_graph_denotations({'edgeSet': [{'right': ['Percy', 'Jackson'], 'type': 'reverse', 'rightkbID': 'Q6256', 'kbID':"P813q"}]})
    []
    >>> query_graph_denotations({'edgeSet': [{'type': 'reverse', 'rightkbID': 'Q35637', 'kbID':"P1346v", 'num':['2009']}]})
    [{'e1': 'Q76'}]
    >>> query_graph_denotations({'edgeSet': [{'type': 'reverse', 'rightkbID': 'Q329816', 'kbID':"P571v"}], 'tokens':["when", "did","start"]})
    [{'e1': 'VTfb0eeb812ca69194eaaa87efa0c6d51d'}]
    >>> query_graph_denotations({'edgeSet': [{'rightkbID': 'Q1297', 'kbID':'P281v', 'type':'reverse'}], 'tokens':["what", "zip", "code"]})
    [{'e1': '60601'}, {'e1': '60827'}, {'e1': '60601'}, {'e1': '60827'}]
    >>> label_query_results(query_graph_denotations({'filter':'importance', 'edgeSet': [{'kbID': 'P206v', 'rightkbID': 'Q19686', 'type': 'direct'}]}))
    [['london, united kingdom', 'london, uk', 'london', 'london, england'], ['square mile', 'city and county of the city of london', 'the city', 'city of london'], ['oxford']]
    """
    if "zip" in g.get('tokens', []) and any(e.get('kbID') == "P281v" for e in g.get('edgeSet',[])):
        denotations = query_wikidata(graph_to_query(g, return_var_values=True), starts_with="")
        denotations = [r for r in denotations if any('x' not in r[b] for b in r)]  # Post process zip codes
        post_processed = []
        for r in denotations:
            for b in r:
                codes = re.split("[-–]", r[b])
                for p in codes:
                    if p:
                        if len(p) < len(codes[0]):
                            p = codes[0][:(len(codes[0])) - len(p)] + p
                        post_processed.append({b: p})
        return post_processed
    denotations = query_wikidata(graph_to_query(g, return_var_values=True))
    question_text = " ".join(g.get('tokens', []))
    if not question_text.startswith("when") and not question_text.startswith("what year"):
        denotations = [r for r in denotations if any('-' not in r[b] and r[b][0] in 'pqPQ' for b in r)]  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    if 'filter' in g and g['filter'] == 'importance':
        denotations = filter_denotation_by_importance(denotations)
    return denotations


def filter_denotation_by_importance(denotations, keep=3):
    """
    Keep only top most important entities judging by their ids.

    :param denotations: a list of entity ids
    :param keep: how many entities to keep
    :return: list of entitiy ids
    >>> filter_denotation_by_importance(['Q161491', 'Q523651', 'Q1143278', 'Q179385', 'Q592123', 'Q62378', 'Q617407', 'Q858775'])
    ['Q62378', 'Q161491', 'Q179385']
    >>> filter_denotation_by_importance(['Q161491', 'Q523651', 'Q1143278', 'Q179385', 'Q592123', 'Q62h378', 'Q617407', 'Q858775'])
    ['Q161491', 'Q179385', 'Q523651']
    >>> filter_denotation_by_importance([{'e1': 'Q161491'}, {'e1': 'Q523651'}, {'e1': 'Q1143278'}, {'e1': 'Q179385'}, {'e1': 'Q592123'}, {'e1': 'Q62378'}])
    [{'e1': 'Q62378'}, {'e1': 'Q161491'}, {'e1': 'Q179385'}]
    >>> filter_denotation_by_importance([{'e1': 'Q161-491'}, {'e1': 'Q52P3651'}, {'e1': 'Q1143278'}, {'e1': 'Q179385'}, {'e1': 'Q592123'}, {'e1': 'Q62378'}])
    [{'e1': 'Q62378'}, {'e1': 'Q179385'}, {'e1': 'Q592123'}]
    """
    if len(denotations) > 0 and type(denotations[0]) == dict:
        denotations = [r for r in denotations if any('-' not in r[b] and r[b][0] in 'pqPQ' for b in r)]
        denotations = sorted([k for k in denotations if k.get('e1', ' ')[1:].isnumeric()], key=lambda k: int(k.get('e1', ' ')[1:]))[:keep]
    else:
        denotations = [r for r in denotations if '-' not in r and r[0] in 'pqPQ']
        denotations = sorted([k for k in denotations if k[1:].isnumeric()], key=lambda k: int(k[1:]))[:keep]
    return denotations


def graph_to_select(g, **kwargs):
    return graph_to_query(g, ask=False, **kwargs)


def graph_to_ask(g, **kwargs):
    return graph_to_query(g, ask=True,  **kwargs)


def graph_to_query(g, ask=False, return_var_values=False, limit=GLOBAL_RESULT_LIMIT):
    """
    Convert graph to a sparql query.

    :param ask: if the a simple existence of the graph should be checked instead of returning variable values.
    :param g: a graph as a dictionary with non-empty edgeSet
    :param return_var_values: if True the denotations for free variables will be returned
    :param limit: limit on the result list size
    :return: a sparql query
    >>> g = {'edgeSet': [{'kbID': 'P35v', 'type': 'reverse', 'rightkbID': 'Q155', 'right': [5], 'argmax':'time'}], 'entities': []}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = True)))
    1
    >>> g = {'edgeSet': [{'kbID': 'P35v', 'type': 'reverse', 'rightkbID': 'Q155', 'right': [5]}], 'entities': []}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = True)))
    6
    >>> g = {'edgeSet': [{'right': ["Missouri"]}], 'entities': [[4]], 'tokens': ['who', 'are', 'the', 'current', 'senator', 'from', 'missouri', '?']}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = False)))
    152
    """
    variables = []
    order_by = []
    query = "{"
    if graph.graph_has_temporal(g):
        if any(edge.get('type') == 'time' for edge in g.get('edgeSet', [])):
            query += sparql_temporal_values_v
        else:
            query += sparql_temporal_values_q
    for i, edge in enumerate(g.get('edgeSet', [])):
        local_variables = []
        if 'type' in edge:
            sparql_relation_inst = sparql_relation[edge['type']]
        else:
            sparql_relation_inst = sparql_relation_complex

        if 'kbID' in edge:
            if edge.get('type') == 'v-structure':
                sparql_relation_inst = sparql_relation_inst.replace("VALUES ?rv { e:P161v e:P453q e:P175q}", "")

            # This is a very special case but there is no other place to put it
            if edge['kbID'] == "P131v":
                sparql_relation_inst = sparql_relation_inst.replace("?p ?m . ?m ?rr", "(e:P131s/e:P131v)+")
            else:
                sparql_relation_inst = re.sub(r"\?r[drv]", "e:" + edge['kbID'], sparql_relation_inst)
        elif edge.get('type') not in {'time'}:
            sparql_relation_inst = sparql_relation_inst.replace("?r", "?r" + str(i))
            local_variables.extend(["?r{}{}".format(i, t[0]) for t in ['direct', 'reverse']] if 'type' not in edge else ["?r{}{}".format(i, edge['type'][0])])
            # for v in local_variables:
            #     sparql_relation_inst += sparql_relation_filter.replace("%relationvar%", v)

        if 'hopUp' in edge or 'hopDown' in edge:
            hop = 'hopDown' if 'hopDown' in edge else 'hopUp'
            sparql_hop = sparql_entity_specify if 'hopDown' in edge else sparql_entity_abstract
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "?e3")
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", sparql_hop + " %restriction%")
            if edge[hop]:
                sparql_relation_inst = sparql_relation_inst.replace("?hopv",  "e:" + edge[hop])
                sparql_relation_inst = sparql_relation_inst.replace("?hops",  "e:" + edge[hop][:-1] + "s")
            else:
                sparql_relation_inst = sparql_hopup_values + sparql_relation_inst
                local_variables.append("?hop{}v".format(i))

        if any(arg_type in edge for arg_type in ['argmax', 'argmin']):
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", sparql_restriction_time_argmax)
            sparql_relation_inst = sparql_relation_inst.replace("FILTER (YEAR(?n) = ?yearvalue)", "")
            # sparql_relation_inst = sparql_relation_inst.replace("?n", "?n" + str(i))
            # sparql_relation_inst = sparql_relation_inst.replace("?a", "?a" + str(i))
            if return_var_values:
                order_by.append("{}({})".format("DESC" if 'argmax' in edge else "ASC", "?n"))
                limit = 1
        elif 'num' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", sparql_restriction_time_argmax)
            sparql_relation_inst = sparql_relation_inst.replace("?yearvalue",  " ".join(edge['num']))
        else:
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", "")

        sparql_relation_inst = sparql_relation_inst.replace("?p", "?p" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?m", "?m" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?e3", "?e3" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?hop", "?hop" + str(i))

        if 'rightkbID' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "e:" + edge['rightkbID'])
        elif 'right' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "?e2" + str(i))
            right_label = " ".join(edge['right'])
            sparql_entity_label_inst = sparql_entity_label.replace("VALUES ?labelright { %entitylabels }", "")
            sparql_entity_label_inst = sparql_entity_label_inst.replace("?e2", "?e2" + str(i))
            sparql_entity_label_inst = sparql_entity_label_inst.replace("?labelright", "\"{}\"@en".format(right_label))
            local_variables.append("?e2" + str(i))
            query += sparql_entity_label_inst
        sparql_relation_inst = sparql_relation_inst.replace("_:m", "_:m" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("_:s", "_:s" + str(i))

        query += sparql_relation_inst
        variables.extend(local_variables)
        # if not local_variables or return_var_values:
        #     local_variables.append('?e1')
        # query = query.replace("%queryvariables%", " ".join(local_variables))

    query = sparql_prefix + (sparql_select if not ask else sparql_ask) + query

    if return_var_values and not ask:
        variables.append("?e1")
        # query += "BIND (xsd:integer(SUBSTR(STR(?e1), 33)) AS ?eid)"
        # order_by.append("?eid")
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    if order_by and not ask:
        order_by_pattern = sparql_close_order.format(" ".join(order_by))
        query += order_by_pattern
    if not ask:
        query += sparql_close.format(limit)

    logger.debug("Querying with variables: {}".format(variables))
    return query


def get_free_variables(g, include_relations=True, include_entities=True):
    """
    Construct a list of free (not linked) variables in the graph.

    :param g: the graph as a dictionary with an 'edgeSet'
    :param include_relations: if include variables that denote relations
    :param include_entities: if include variables that denote entities
    :return:
    """
    free_variables = []
    for i, edge in enumerate(g.get('edgeSet', [])):
        if edge.get('type') not in {'time'}:
            if include_relations and 'kbID' not in edge:
                free_variables.extend(["?r{}{}".format(i, t[0]) for t in ['direct', 'reverse']] if 'type' not in edge else ["?r{}{}".format(i, edge['type'][0])])
            if include_entities and 'rightkbID' not in edge:
                free_variables.append("?e2" + str(i))
    return free_variables


def entity_query(label, limit=100):
    """
    A method to look up a WikiData entity by a label.

    :param label: label of the entity as str
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    sparql_entity_label_inst = sparql_entity_label.replace("VALUES ?labelright { %entitylabels }", "")
    sparql_entity_label_inst = sparql_entity_label_inst.replace("?e2", "?e2" + str(0))
    sparql_entity_label_inst = sparql_entity_label_inst.replace("?labelright", "\"{}\"@en".format(label, label))
    variables.append("?e2" + str(0))
    query += sparql_entity_label_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit)
    logger.debug("Querying for entity with variables: {}".format(variables))
    return query


def character_query(label, film_id, limit=3):
    """
    A method to look up a WikiData film character by a label.

    :param label: label of the entity as str
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    >>> query_wikidata(character_query("Bella", "Q160071"))
    [{'e20': 'Q223757'}]
    >>> query_wikidata(character_query("Anakin", "Q42051"))
    [{'e20': 'Q51752'}]
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    sparql_entity_label_inst = sparql_character_label.replace("?e2", "?e2" + str(0))
    sparql_entity_label_inst = sparql_entity_label_inst.replace("?e1", "e:{}".format(film_id))
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%entitylabels", "\"{}\"".format(label, label))
    variables.append("?e2" + str(0))
    query += sparql_entity_label_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit)
    logger.debug("Querying for entity with variables: {}".format(variables))
    return query


def multi_entity_query(labels, limit=100):
    """
    A method to look up a WikiData entities given a set of labels

    :param labels: entity labels as a list of str
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    sparql_entity_label_inst = sparql_entity_label.replace("?e2", "?e2" + str(0))
    labels = ["\"{}\"@en \"{}\"@de".format(l, l) for l in labels]
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%entitylabels", " ".join(labels))
    variables.append("?e2" + str(0))
    variables.append("?labelright")
    query += sparql_entity_label_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit)
    logger.debug("Querying for entity with variables: {}".format(variables))
    return query


def label_query(entity, limit=10):
    """
    Construct a WikiData query to retrieve entity labels for the given entity id.

    :param entity: entity kbID
    :param limit: limit on the result list size
    :return: a WikiData query
    >>> query_wikidata(label_query("Q36"), starts_with=None)
    [{'label0': 'Poland'}, {'label0': 'Polen'}, {'label0': 'Republic of Poland'}, {'label0': 'Polska'}, {'label0': 'PL'}, {'label0': 'pl'}, {'label0': 'POL'}]
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    sparql_label_entity_inst = sparql_label_entity.replace("VALUES ?e2 { %entityids }", "")
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?e2", "e:" + entity)
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?label", "?label" + str(0))
    variables.append("?label" + str(0))
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit)
    logger.debug("Querying for label with variables: {}".format(variables))
    return query


def multientity_label_query(entities, limit=10):
    """
    Construct a WikiData query to retrieve entity labels for the given list of entity ids.

    :param entities: entity kbIDs
    :param limit: limit on the result list size (multiplied with the size of the entity list)
    :return: a WikiData query
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    if all(e[0] not in 'qQ' or '-' in e for e in entities):
        sparql_label_entity_inst = sparql_year_entity
    else:
        entities = [e for e in entities if '-' not in e and e[0] in 'pqPQ']
        sparql_label_entity_inst = sparql_label_entity
    sparql_label_entity_inst = sparql_label_entity_inst.replace("%entityids", " ".join(["e:" + entity for entity in entities]))
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?label", "?label" + str(0))
    variables.append("?e2")
    variables.append("?label" + str(0))
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit*len(entities))
    logger.debug("Querying for label with variables: {}".format(variables))
    return query


def main_label_query(entity):
    """
    Construct a WikiData query to retrieve the main entity label for the given entity id.

    :param entity: entity kbID
    :return: a WikiData query
    >>> query_wikidata(main_label_query("Q36"), starts_with=None)
    [{'label0': 'Poland'}]
    """
    query = sparql_prefix
    query += sparql_select
    query += "{"
    sparql_label_entity_inst = sparql_canoncial_label_entity.replace("VALUES ?e2 { %entityids }", "")
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?e2", "e:" + entity)
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?label", "?label" + str(0))
    sparql_label_entity_inst = sparql_label_entity_inst.replace("skos:altLabel", "")
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", "?label" + str(0))
    query += sparql_close.format(1)
    return query


def demonym_query(entity):
    """
    Construct a WikiData query to retrieve the main entity label for the given entity id.

    :param entity: entity kbID
    :return: a WikiData query
    >>> query_wikidata(demonym_query("Q183"), starts_with="")
    [{'labelright': 'German'}]
    """
    query = sparql_prefix
    query += sparql_select
    query += "{"
    sparql_label_entity_inst = sparql_get_demonym
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?e2", "e:" + entity)
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", "?labelright")
    query += sparql_close.format(1)
    return query


query_cache = {}


def clear_cache():
    global query_cache
    query_cache = {}


def query_wikidata(query, starts_with=WIKIDATA_ENTITY_PREFIX, use_cache=False):
    """
    Execute the following query against WikiData
    :param query: SPARQL query to execute
    :param starts_with: if supplied, then each result should have the give prefix. The prefix is stripped
    :param use_cache:
    :return: a list of dictionaries that represent the queried bindings
    """
    if use_cache and query in query_cache:
        return query_cache[query]
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except Exception as inst:
        logger.debug(inst)
        return []
    if "results" in results and len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        logger.debug("Results bindings: {}".format(results[0].keys()))
        if starts_with:
            results = [r for r in results if all(r[b]['value'].startswith(starts_with) for b in r)]
        results = [{b: (r[b]['value'].replace(starts_with, "") if starts_with else r[b]['value']) for b in r} for r in results]
        if use_cache:
            query_cache[query] = results
        return results
    elif "boolean" in results:
        return results['boolean']
    else:
        logger.debug(results)
        return []


def map_query_results(query_results, question_variable='e1'):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :param question_variable: the variable to extract
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> map_query_results([{'e1':'Q76'}, {'e1':'Q235234'}])
    [['barack obama'], ['q235234']]
    """
    answers = [r[question_variable] for r in query_results]
    answers = [a for a in answers if '-' not in a and a[0] in 'pqPQ']  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    answers = [[e.lower() for e in entity_map.get(a, [a])] for a in answers]
    # TODO: what to do about further inconsistencies
    # answers = [[e.lower() for e in entity_map.get(a, [l.get('label0') for l in query_wikidata(label_query(a), starts_with="", use_cache=True)])] for a in answers]
    return answers


def label_entity(entity):
    """
    Retrieve the main label of the given entity. None is returned if no label could be found.

    :param entity: entity KB ID
    :return: entity label as a string
    >>> label_entity("Q12143")
    'time zone'
    """
    results = query_wikidata(main_label_query(entity), starts_with="", use_cache=True)
    if results and 'label0' in results[0]:
        return results[0]['label0']
    return None


def label_many_entities_with_alt_labels(entities):
    """
    Label the given set of variables with all labels available for them in the knowledge base.

    :param entities: a list of entity ids.
    :return: a dictionary mapping entity id to a list of labels
    >>> dict(label_many_entities_with_alt_labels(["Q76", "Q188984", "Q194339"])) == \
    {'Q188984': {'NY Rangers', 'Blue-Shirts', 'Blue Shirts', 'Broadway Blueshirts', 'New York Rangers', 'NYR'}, 'Q194339': {'Bahamian dollar', 'Bahama-Dollar', 'B$', 'Bahamas-Dollar'}, 'Q76': {'Barack Obama', 'Barack Hussein Obama II', 'Barack H. Obama', 'Barack Obama II', 'Barack Hussein Obama, Jr.', 'Barack Hussein Obama', 'Obama'}}
    True
    >>> dict(label_many_entities_with_alt_labels(["VTfb0eeb812ca69194eaaa87efa0c6d51d"]))
    {'VTfb0eeb812ca69194eaaa87efa0c6d51d': {'1972'}}
    """
    results = query_wikidata(multientity_label_query(entities), starts_with="", use_cache=False)
    if len(results) > 0:
        retrieved_labels = defaultdict(set)
        for result in results:
            entity_id = result.get("e2", "").replace(WIKIDATA_ENTITY_PREFIX, "")
            retrieved_labels[entity_id].add(result.get("label0", ""))
        return retrieved_labels
    return {}


def label_query_results(query_results, question_variable='e1'):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :param question_variable: the variable to extract
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> sorted(sorted(label_query_results([{'e1':'Q76'}, {'e1':'Q235234'}, {'e1':'r68123123-12dd222'}]))[0])
    ['barack h. obama', 'barack hussein obama', 'barack hussein obama ii', 'barack hussein obama, jr.', 'barack obama', 'barack obama ii', 'obama']
    >>> label_query_results([{'e1': '10000'}, {'e1': '10499'}, {'e1': '11004'}, {'e1': '05'}, {'e1': ""}])
    ['10000', '10499', '11004', '05']
    """
    answers = [r[question_variable] for r in query_results]
    # answers = [a for a in answers if '-' not in a and a[0] in 'pqPQ']  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    answers_to_label = [a for a in answers if not a.isnumeric() and len(a) > 0]
    rest_answers = [[a] for a in answers if a.isnumeric()]
    answers = [[l.lower() for l in labels] for _, labels in label_many_entities_with_alt_labels(answers_to_label).items()]
    answers = normalize_answer_strings(answers)
    # answers = [[l.get('label0').lower() for l in query_wikidata(label_query(a), starts_with="", use_cache=True)] for a in answers]
    return answers + rest_answers


def normalize_answer_strings(answers):
    """
    Add normalized alternative labels.

    :param answers: list of lists of string answers
    :return: list of lists of string answers
    >>> normalize_answer_strings([['twilight saga: breaking dawn - part 2'], ['the twilight saga: new moon', 'twilight saga: new moon']])
    [['twilight saga: breaking dawn - part 2', 'twilight saga', 'breaking dawn - part 2', 'twilight saga: breaking dawn', 'part 2', 'breaking dawn', 'part 2', 'twilight saga', 'breaking dawn'], ['the twilight saga: new moon', 'twilight saga: new moon', 'the twilight saga', 'new moon', 'twilight saga', 'new moon']]
    >>> normalize_answer_strings([['2010 world series', 'world series 2010'], ['2012 world series', 'world series 2012'], ['world series 2014', '2014 world series']])
    [['2010 world series', 'world series 2010'], ['2012 world series', 'world series 2012']]
    >>> normalize_answer_strings([['liste gegenwärtig amtierender staatsoberhäupter nach amtszeiten', 'list of heads of state by diplomatic precedence']])
    []
    """
    answers = [[a.replace("–", "-").lower() for a in answer_set] for answer_set in answers]
    new_answers = []
    for answer_set in answers:
        for a in answer_set:
            if ":" in a:
                answer_set.extend([w.strip() for w in a.split(":")])
            if "," in a:
                answer_set.extend([w.strip() for w in a.split(",")])
            if " - " in a:
                answer_set.extend([w.strip() for w in a.split(" - ")])
            if "standard time" in a:
                answer_set.append(a.replace("standard time", "time zone"))
        if not any(re.search("\\b(2014|2015|2016|2017|list of)\\b", a) for a in answer_set):
            new_answers.append(answer_set)
    return new_answers


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

