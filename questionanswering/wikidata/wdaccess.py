import logging
import re
from collections import defaultdict

import nltk
from SPARQLWrapper import SPARQLWrapper, JSON
from construction import graph

WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"

wdaccess_p = {
    'wikidata_url': "http://knowledgebase:8890/sparql",
    'timeout': 40,
    'global_result_limit': 1000,
    'logger': logging.getLogger(__name__),
    'restrict.hopup': False
}

logger = wdaccess_p['logger']
logger.setLevel(logging.ERROR)

sparql = SPARQLWrapper(wdaccess_p.get('wikidata_url', "http://knowledgebase:8890/sparql"))
sparql.setReturnFormat(JSON)
sparql.setMethod("GET")
sparql.setTimeout(wdaccess_p.get('timeout', 40))
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
        {GRAPH <http://wikidata.org/statements> { ?m ?p ?e2 . ?m ?rv ?e1 . %restriction% }}
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
        { VALUES ?labelpredicate {rdfs:label skos:altLabel}
        GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate "%labelright%"@en  }
        } FILTER NOT EXISTS {
            VALUES ?topic {e:Q4167410 e:Q21286738 e:Q11266439 e:Q13406463}
            GRAPH <http://wikidata.org/instances> {?e2 rdf:type ?topic}}
        """

sparql_label_entity = """
        {
        VALUES ?e2 { %entityids }
        VALUES ?labelpredicate {rdfs:label skos:altLabel}
        GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate ?label }
        FILTER ( lang(?label) = "en" )
        }
        """

sparql_canoncial_label_entity = """
        {
        GRAPH <http://wikidata.org/terms> { ?e2 rdfs:label ?label }
        FILTER ( lang(?label) = "en" )
        }
        """


sparql_restriction_time_argmax = "?m ?a [base:time ?n]."

sparql_relation_filter = 'FILTER NOT EXISTS { GRAPH <http://wikidata.org/properties> {%relationvar% rdf:type base:Property}}'

sparql_close_order = " ORDER BY {}"
sparql_close = " LIMIT {}"

# TODO: Additional?: given name
HOP_UP_RELATIONS = {"P131", "P31", "P279", "P17", "P361", "P1445", "P179"} # + P674 Depricated
TEMPORAL_RELATIONS = {"P585q", "P580q", "P582q", "P577q", "P571q", "P580v", "P582v", "P577v", "P571v", "P569v", "P570v"}

sparql_entity_abstract = "[ ?hopups [ ?hopupv ?e2]]"
#Can we also have something like [ [?e2 ?hopups ] ?hopupv ]
sparql_hopup_values = ""
sparql_temporal_values = "VALUES ?a {" + " ".join(["e:{}".format(r) for r in TEMPORAL_RELATIONS]) + "}"

FILTER_ENDINGS = "r"


def update_sparql_clauses():
    global sparql_hopup_values
    if wdaccess_p.get('restrict.hopup'):
        sparql_hopup_values = "VALUES (?hopups ?hopupv) {" + " ".join(["(e:{}s e:{}v)".format(r, r) for r in HOP_UP_RELATIONS]) + "}"


def query_graph_groundings(g, use_cache=False, with_denotations=False, pass_exception=False):
    """
    Convert the given graph to a WikiData query and retrieve the results. The results contain possible bindings
    for all free variables in the graph. If there are no free variables a single empty grounding is returned.

    :param g: graph as a dictionary
    :param use_cache
    :return: graph groundings encoded as a list of dictionaries
    >>> len(query_graph_groundings({'edgeSet': [{'right': ['book'], 'rightkbID': 'Q571', 'type':'direct', 'argmax':'time'}], 'entities': []}))
    4
    >>> len(query_graph_groundings({'edgeSet': [{'rightkbID': 'Q127367', 'type':'reverse'}, {'type':'time'}], 'entities': []}))
    27
    >>> len(query_graph_groundings({'edgeSet': [{'rightkbID': 'Q127367', 'type':'reverse'}, {'type':'time', 'argmax':'time'}], 'entities': []}))
    27
    """
    if get_free_variables(g):
        groundings = query_wikidata(graph_to_query(g, limit=GLOBAL_RESULT_LIMIT*(10 if with_denotations else 1), return_var_values=with_denotations), use_cache=use_cache)
        if groundings is None:  # If there was an exception
            return None if pass_exception else []
        groundings = [r for r in groundings if not any(r[b][:-1] in property_blacklist or r[b] in TEMPORAL_RELATIONS or r[b][-1] in FILTER_ENDINGS for b in r)]
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
    """
    denotations = query_wikidata(graph_to_query(g, return_var_values=True))
    denotations = [r for r in denotations if any('-' not in r[b] and r[b][0] in 'pqPQ' for b in r)]  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
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
        query += sparql_temporal_values
    for i, edge in enumerate(g.get('edgeSet', [])):
        local_variables = []
        if 'type' in edge:
            sparql_relation_inst = sparql_relation[edge['type']]
        else:
            sparql_relation_inst = sparql_relation_complex

        if 'kbID' in edge:
            sparql_relation_inst = re.sub(r"\?r[drv]", "e:" + edge['kbID'], sparql_relation_inst)
        elif edge.get('type') not in {'time'}:
            sparql_relation_inst = sparql_relation_inst.replace("?r", "?r" + str(i))
            local_variables.extend(["?r{}{}".format(i, t[0]) for t in ['direct', 'reverse']] if 'type' not in edge else ["?r{}{}".format(i, edge['type'][0])])
            # for v in local_variables:
            #     sparql_relation_inst += sparql_relation_filter.replace("%relationvar%", v)

        if 'hopUp' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", sparql_entity_abstract)
            if edge['hopUp']:
                sparql_relation_inst = sparql_relation_inst.replace("?hopupv",  "e:" + edge['hopUp'])
                sparql_relation_inst = sparql_relation_inst.replace("?hopups",  "e:" + edge['hopUp'][:-1] + "s")
            else:
                sparql_relation_inst = sparql_hopup_values + sparql_relation_inst
                local_variables.append("?hopup{}v".format(i))

        if any(arg_type in edge for arg_type in ['argmax', 'argmin']):
            sparql_relation_inst = sparql_relation_inst
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", sparql_restriction_time_argmax)
            # sparql_relation_inst = sparql_relation_inst.replace("?n", "?n" + str(i))
            # sparql_relation_inst = sparql_relation_inst.replace("?a", "?a" + str(i))
            if return_var_values:
                order_by.append("{}({})".format("DESC" if 'argmax' in edge else "ASC", "?n" + str(i)))
                limit = 1
        else:
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", "")

        sparql_relation_inst = sparql_relation_inst.replace("?p", "?p" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?m", "?m" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?hopup", "?hopup" + str(i))

        if 'rightkbID' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "e:" + edge['rightkbID'])
        elif 'right' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "?e2" + str(i))
            right_label = " ".join(edge['right'])
            sparql_entity_label_inst = sparql_entity_label.replace("?e2", "?e2" + str(i))
            sparql_entity_label_inst = sparql_entity_label_inst.replace("%labelright%", right_label)
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
        query += "BIND (xsd:integer(SUBSTR(STR(?e1), 33)) AS ?eid)"
        order_by.append("?eid")
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
    sparql_entity_label_inst = sparql_entity_label.replace("?e2", "?e2" + str(0))
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%labelright%", label)
    variables.append("?e2" + str(0))
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
    [{'label0': 'Poland'}, {'label0': 'Republic of Poland'}, {'label0': 'Polska'}, {'label0': 'PL'}, {'label0': 'pl'}, {'label0': 'POL'}]
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
    sparql_label_entity_inst = sparql_label_entity.replace("%entityids", " ".join(["e:" + entity for entity in entities]))
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
    sparql_label_entity_inst = sparql_label_entity.replace("VALUES ?e2 { %entityids }", "")
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?e2", "e:" + entity)
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?label", "?label" + str(0))
    sparql_label_entity_inst = sparql_label_entity_inst.replace("skos:altLabel", "")
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", "?label" + str(0))
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


def load_blacklist(path_to_list):
    try:
        with open(path_to_list) as f:
            return_list = {l.strip() for l in f.readlines()}
        return return_list
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
        return set()


def load_property_labels(path_to_property_labels):
    try:
        with open(path_to_property_labels) as infile:
            return_map = {l.split("\t")[0]: l.split("\t")[1].strip().lower() for l in infile.readlines()}
        return return_map
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
        return {}


def load_entity_map(path_to_map):
    """
    Load the map of entity labels from a file.

    :param path_to_map: location of the map file
    :return: entity map as an nltk.Index
    """
    try:
        with open(path_to_map) as f:
            return_map = [l.strip().split("\t") for l in f.readlines()]
        return nltk.Index({(t[1], t[0]) for t in return_map})
    except Exception as ex:
        logger.error("No entity map found. {}".format(ex))
        return {"Q76": ["Barack Obama"]}


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
    {"Q188984": ["New York Rangers"], "Q76": ["Barack Obama", "Barack Hussein Obama II", "Obama", "Barack Hussein Obama", "Barack Obama II"], "Q194339": ["Bahamian dollar"]}
    True
    """
    results = query_wikidata(multientity_label_query(entities), starts_with="", use_cache=False)
    if len(results) > 0:
        retrieved_labels = defaultdict(list)
        for result in results:
            entity_id = result.get("e2", "").replace(WIKIDATA_ENTITY_PREFIX, "")
            retrieved_labels[entity_id].append(result.get("label0", ""))
        return retrieved_labels
    return {}


def label_query_results(query_results, question_variable='e1'):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :param question_variable: the variable to extract
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> sorted(label_query_results([{'e1':'Q76'}, {'e1':'Q235234'}, {'e1':'r68123123-12dd222'}]))
    [['barack obama', 'barack hussein obama ii', 'obama', 'barack hussein obama', 'barack obama ii'], ['james i of scotland', 'james i, king of scots']]
    """
    answers = [r[question_variable] for r in query_results]
    answers = [a for a in answers if '-' not in a and a[0] in 'pqPQ']  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    answers = [[l.lower() for l in labels] for _, labels in label_many_entities_with_alt_labels(answers).items()]
    # answers = [[l.get('label0').lower() for l in query_wikidata(label_query(a), starts_with="", use_cache=True)] for a in answers]
    return answers

RESOURCES_FOLDER = "../resources/"
entity_map = load_entity_map(RESOURCES_FOLDER + "entity_map.tsv")
property_blacklist = load_blacklist(RESOURCES_FOLDER + "property_blacklist.txt")
property_whitelist = load_blacklist(RESOURCES_FOLDER + "property_whitelist.txt")
property2label = load_property_labels(RESOURCES_FOLDER + "properties-with-labels.txt")

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

