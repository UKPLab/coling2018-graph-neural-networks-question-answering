import logging
import re

import nltk
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

sparql = SPARQLWrapper("http://knowledgebase:8890/sparql")
sparql.setReturnFormat(JSON)
sparql.setMethod("GET")
sparql.setTimeout(40)
GLOBAL_RESULT_LIMIT = 500


sparql_prefix = """
        PREFIX e:<http://www.wikidata.org/entity/>
        PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
        PREFIX base:<http://www.wikidata.org/ontology#>
        """
sparql_select = """
        SELECT DISTINCT %queryvariables% WHERE
        """

sparql_relation = {
    "direct": "{GRAPH <http://wikidata.org/statements> { ?e1 ?p ?m . ?m ?rd ?e2 . %restriction% }}",

    "reverse": "{GRAPH <http://wikidata.org/statements> { ?e2 ?p ?m . ?m ?rr ?e1 . %restriction% }}",

    "v-structure": "{GRAPH <http://wikidata.org/statements> { ?m ?p ?e2 . ?m ?rv ?e1 . %restriction% }}",
}

sparql_relation_complex = """
        {
        {GRAPH <http://wikidata.org/statements> { ?e1 ?p ?m . ?m ?rd ?e2 . }}
        UNION
        {GRAPH <http://wikidata.org/statements> { ?e2 ?p ?m . ?m ?rr ?e1 . }}
        }
        """

# sparql_relation = {
#     "direct": "{GRAPH <http://wikidata.org/statements> { ?e1 ?rd ?m . ?m ?p ?e2 . %restriction% }}",
#
#     "reverse": "{GRAPH <http://wikidata.org/statements> { ?e2 ?rr ?m . ?m ?p ?e1 . %restriction% }}",
#
#     "v-structure": "{GRAPH <http://wikidata.org/statements> { ?m ?rv ?e2 . ?m ?p ?e1 . %restriction% }}",
# }
#
# sparql_relation_complex = """
#         {
#         {GRAPH <http://wikidata.org/statements> { ?e1 ?rd ?m . ?m ?p ?e2 . }}
#         UNION
#         {GRAPH <http://wikidata.org/statements> { ?e2 ?rr ?m . ?m ?p ?e1 . }}
#         }
#         """
# sparql_relation_complex = """
#         {
#         {GRAPH <http://wikidata.org/statements> { ?e1 ?p ?m . ?m ?rd ?e2 . }}
#         UNION
#         {GRAPH <http://wikidata.org/statements> { ?e2 ?p ?m . ?m ?rr ?e1 . }}
#         UNION
#         {GRAPH <http://wikidata.org/statements> { ?m ?p ?e2. ?m ?rv ?e1. }}
#         }
#         """

sparql_entity_label = """
        { VALUES ?labelpredicate {rdfs:label skos:altLabel}
        GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate "%labelright%"@en  }
        } FILTER NOT EXISTS {
            VALUES ?topic {e:Q4167410 e:Q21286738 e:Q11266439}
            GRAPH <http://wikidata.org/instances> {?e2 rdf:type ?topic}}
        """

sparql_label_entity = """
        { VALUES ?labelpredicate {rdfs:label skos:altLabel}
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


sparql_relation_time_argmax = "?m ?a [base:time ?n]."

sparql_close_order = " ORDER BY {} LIMIT 1"
sparql_close = " LIMIT {}"

# TODO: Additional?: given name
HOP_UP_RELATIONS = ["P131", "P31", "P279", "P17", "P361"]

sparql_entity_abstract = "[ ?hopups [ ?hopupv ?e2]]"
sparql_hopup_values = "VALUES (?hopups ?hopupv) {" + " ".join(["(e:{}s e:{}v)".format(r, r) for r in HOP_UP_RELATIONS]) + "}"


def query_graph_groundings(g):
    """
    Convert the given graph to a WikiData query and retrieve the results. The results contain possible bindings
    for all free variables in the graph. If there are no free variables a single empty grounding is returned.

    :param g: graph as a dictionary
    :return: graph groundings encoded as a list of dictionaries
    """
    if get_free_variables(g):
        return query_wikidata(graph_to_query(g))
    return [{}]


def query_graph_denotations(g):
    """
    Convert the given graph to a WikiData query and retrieve the denotations of the graph. The results contain the
     list of the possible graph denotations

    :param g: graph as a dictionary
    :return: graph denotations as a list of dictionaries
    """
    return query_wikidata(graph_to_query(g, return_var_values=True))


def graph_to_query(g, return_var_values=False, limit=GLOBAL_RESULT_LIMIT):
    """
    Convert graph to a sparql query.
    :param g: a graph as a dictionary with non-empty edgeSet
    :param return_var_values: if True the denotations for free variables will be returned
    :param limit: limit on the result list size
    :return: a sparql query
    >>> g = {'edgeSet': [{'left': [0], 'kbID': 'P35v', 'type': 'reverse', 'rightkbID': 'Q155', 'right': [5], 'argmax':'time'}], 'entities': []}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = True)))
    1
    >>> g = {'edgeSet': [{'left': [0], 'kbID': 'P35v', 'type': 'reverse', 'rightkbID': 'Q155', 'right': [5]}], 'entities': []}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = True)))
    5
    >>> g = {'edgeSet': [{'left': [0], 'right': ["Missouri"]}], 'entities': [[4]], 'tokens': ['who', 'are', 'the', 'current', 'senator', 'from', 'missouri', '?']}
    >>> len(query_wikidata(graph_to_query(g, return_var_values = False)))
    114
    """
    query = sparql_prefix
    variables = []
    order_by = []
    query += sparql_select
    query += "{"
    for i, edge in enumerate(g.get('edgeSet', [])):
        if 'type' in edge:
            sparql_relation_inst = sparql_relation[edge['type']]
        else:
            sparql_relation_inst = sparql_relation_complex

        if 'kbID' in edge:
            sparql_relation_inst = re.sub(r"\?r[drv]", "e:" + edge['kbID'], sparql_relation_inst)
        else:
            sparql_relation_inst = sparql_relation_inst.replace("?r", "?r" + str(i))
            variables.extend(["?r{}{}".format(i, t[0]) for t in sparql_relation] if 'type' not in edge else ["?r{}{}".format(i, edge['type'][0])])

        if 'hopUp' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", sparql_entity_abstract)
            if edge['hopUp']:
                sparql_relation_inst = sparql_relation_inst.replace("?hopupv",  "e:" + edge['hopUp'])
                sparql_relation_inst = sparql_relation_inst.replace("?hopups",  "e:" + edge['hopUp'][:-1] + "s")
            else:
                sparql_relation_inst = sparql_hopup_values + sparql_relation_inst
                variables.append("?hopup{}v".format(i))

        if any(arg_type in edge for arg_type in ['argmax', 'argmin']):
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", sparql_relation_time_argmax)
            sparql_relation_inst = sparql_relation_inst.replace("?n", "?n" + str(i))
            sparql_relation_inst = sparql_relation_inst.replace("?a", "?a" + str(i))
            order_by.append("{}({})".format("DESC" if 'argmax' in edge else "ASC", "?n" + str(i)))
        else:
            sparql_relation_inst = sparql_relation_inst.replace("%restriction%", "")

        sparql_relation_inst = sparql_relation_inst.replace("?p", "?p" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?m", "?m" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("?hopup", "?hopup" + str(i))

        if 'rightkbID' in edge:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "e:" + edge['rightkbID'])
        else:
            sparql_relation_inst = sparql_relation_inst.replace("?e2", "?e2" + str(i))
            right_label = " ".join(edge['right'])
            sparql_entity_label_inst = sparql_entity_label.replace("?e2", "?e2" + str(i))
            sparql_entity_label_inst = sparql_entity_label_inst.replace("%labelright%", right_label)
            variables.append("?e2" + str(i))
            query += sparql_entity_label_inst
        sparql_relation_inst = sparql_relation_inst.replace("_:m", "_:m" + str(i))
        sparql_relation_inst = sparql_relation_inst.replace("_:s", "_:s" + str(i))

        query += sparql_relation_inst

    if return_var_values:
        variables.append("?e1")
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    if order_by:
        order_by_pattern = sparql_close_order.format(" ".join(order_by))
        query += order_by_pattern
    else:
        query += sparql_close.format(limit)

    logger.debug("Querying with variables: {}".format(variables))
    return query


def get_free_variables(g, include_relations=True, include_entities=True):
    """
    Construct a list of free (not linked) variables in the graph.

    :param g: the graph as a dictionary with an 'edgeSet'
    :param include_relations: if include variables that denote relations
    :param include_entities: if include variables that denote entities
    :param include_question_variable: if include the question variable that is always present in the question graphs
    :return:
    """
    free_variables = []
    for i, edge in enumerate(g.get('edgeSet', [])):
        if include_relations and 'kbID' not in edge:
            free_variables.extend(["?r{}{}".format(i, t[0]) for t in sparql_relation] if 'type' not in edge else ["?r{}{}".format(i, edge['type'][0])])
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
    [{'label0': 'Poland'}, {'label0': 'Republic of Poland'}, {'label0': 'POL'}]
    """
    query = sparql_prefix
    variables = []
    query += sparql_select
    query += "{"
    sparql_label_entity_inst = sparql_label_entity.replace("?e2", "e:" + entity)
    sparql_label_entity_inst = sparql_label_entity_inst.replace("?label", "?label" + str(0))
    variables.append("?label" + str(0))
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", " ".join(variables))
    query += sparql_close.format(limit)
    logger.debug("Querying for label with variables: {}".format(variables))
    return query


query_cache = {}


def query_wikidata(query, starts_with="http://www.wikidata.org/entity/", use_cache=False):
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
    if len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        logger.debug("Results bindings: {}".format(results[0].keys()))
        if starts_with:
            results = [r for r in results if all(r[b]['value'].startswith(starts_with) for b in r)]
        results = [{b: (r[b]['value'].replace(starts_with, "") if starts_with else r[b]['value']) for b in r} for r in results]
        results = [r for r in results if not any(r[b][:-1] in property_blacklist for b in r)]
        if use_cache:
            query_cache[query] = results
        return results
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
        return {}


def map_query_results(query_results, question_variable='e1'):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :param question_variable: the variable to extract
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> map_query_results([{'e1':'Q76'}, {'e1':'Q235234'}])
    ['barack obama', 'q235234']
    """
    answers = [r[question_variable] for r in query_results]
    answers = [a for a in answers if '-' not in a]
    answers = [[e.lower() for e in entity_map.get(a, [a])] for a in answers]
    # TODO: what to do about further inconsistencies
    # answers = [[e.lower() for e in entity_map.get(a, [l.get('label0') for l in query_wikidata(label_query(a), starts_with="", use_cache=True)])] for a in answers]
    return answers


def label_entity(entity):
    return query_wikidata(label_query(entity), starts_with="", use_cache=True)


def label_query_results(query_results, question_variable='e1'):
    """
    Extract the variable values from the query results and map them to canonical WebQuestions strings.

    :param query_results: list of dictionaries returned by the sparql endpoint
    :param question_variable: the variable to extract
    :return: list of answers as entity labels or an original id if no canonical label was found.
    >>> label_query_results([{'e1':'Q76'}, {'e1':'Q235234'}])
    [['barack obama', 'barack hussein obama ii', 'obama', 'barack hussein obama', 'barack obama ii', 'barry obama'], ['james i of scotland', 'james i, king of scots']]
    """
    answers = [r[question_variable] for r in query_results]
    answers = [a for a in answers if '-' not in a]  # Filter out WikiData auxiliary variables, e.g. Q24523h-87gf8y48
    answers = [[l.get('label0').lower() for l in query_wikidata(label_query(a), starts_with="", use_cache=True)] for a in answers]
    return answers

RESOURCES_FOLDER = "../resources/"
entity_map = load_entity_map(RESOURCES_FOLDER + "entity_map.tsv")
property_blacklist = load_blacklist(RESOURCES_FOLDER + "property_blacklist.txt")
property_whitelist = load_blacklist(RESOURCES_FOLDER + "property_whitelist.txt")
property2label = load_property_labels(RESOURCES_FOLDER + "properties-with-labels.txt")

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

