import collections
import itertools
from collections import namedtuple
from copy import copy
import re
from typing import List, Dict

from questionanswering import base_objects

WithScore = namedtuple("WithScore", ['graph', 'scores'])


class Edge:
    def __init__(self,
                 leftentityid=None,
                 relationid=None,
                 rightentityid=None,
                 qualifierrelationid=None,
                 qualifierentityid=None):
        self.edgeid = 0
        self.leftentityid = leftentityid
        self.relationid = relationid
        self.rightentityid = rightentityid
        self.qualifierrelationid = qualifierrelationid
        self.qualifierentityid = qualifierentityid
        if self.relationid != 'iclass':
            assert len({self.leftentityid, self.rightentityid, self.qualifierentityid}) == 3

    @property
    def type(self):
        return "ternary" if self.qualifierentityid and (self.relationid or self.rightentityid) else "binary"

    @property
    def grounded(self):
        return (self.relationid is not None and self.relationid != "iclass") or self.qualifierrelationid is not None

    @property
    def temporal(self):
        return self.qualifierentityid in {"MIN", "MAX"} or \
               (self.qualifierentityid is not None and self.qualifierentityid.isdigit()) or \
               (self.rightentityid is not None and self.rightentityid.isdigit())

    @property
    def simple(self):
        return self.leftentityid and self.rightentityid is not None and self.grounded \
               and not(self.qualifierentityid or self.qualifierrelationid) \
               and not self.rightentityid.isdigit()

    def nodes(self):
        return self.leftentityid, self.rightentityid, self.qualifierentityid

    def invert(self):
        """
        Switch the right and left nodes changing the edge direction. Doesn't affect ternary edges.
        """
        if self.type == 'binary':
            self.leftentityid, self.rightentityid = self.rightentityid, self.leftentityid

    def __str__(self):
        return f"{self.__class__.__name__}({self.edgeid})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.edgeid}, " \
               f"{self.leftentityid}-{self.relationid}->{self.rightentityid}" \
               f"{f'~{self.qualifierrelationid}->{self.qualifierentityid}' if self.qualifierentityid else ''})"


DUMMY_EDGE = Edge(leftentityid="foo", rightentityid="bar")


class EdgeList(collections.MutableSequence):

    def __init__(self):
        """
        A list implementation that makes sure that edge ids are not overlapping.


        >>> g = SemanticGraph(edges=[Edge(2, rightentityid="Q1", leftentityid="Q2")])
        >>> g.edges.append(Edge(2, rightentityid="Q1", leftentityid="Q2"))
        >>> g
        SemanticGraph([Edge(2, Q2-None->Q1), Edge(3, Q2-None->Q1)])
        """
        self._list: List[Edge] = list()

    def __setitem__(self, index, value):
        self._set_edge_id(value)
        self._list[index] = value

    def _set_edge_id(self, edge):
        edge_ids = {e.edgeid for e in self._list if e.edgeid is not None}
        if edge.edgeid is None:
            edge.edgeid = 0
        while edge.edgeid in edge_ids:
            edge.edgeid += 1

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __delitem__(self, i):
        del self._list[i]

    def insert(self, index, value):
        self._set_edge_id(value)
        self._list.insert(index, value)

    def __str__(self):
        return self._list.__str__()

    def __repr__(self):
        return self._list.__repr__()


class SemanticGraph:
    def __init__(self,
                 edges: List[Edge]=None,
                 tokens: List[str]=None,
                 free_entities: List[Dict]=None
                 ):
        """
        A semantic graph object.
        >>> SemanticGraph(edges=[Edge(rightentityid="Q1", leftentityid="Q2"), Edge(0,rightentityid="Q1", leftentityid="Q2"), Edge(rightentityid="Q1", leftentityid="Q2")])
        SemanticGraph([Edge(0, Q2-None->Q1), Edge(1, Q2-None->Q1), Edge(2, Q2-None->Q1)])
        >>> SemanticGraph(edges=[Edge(2, rightentityid="Q1", leftentityid="Q2"), Edge(2, rightentityid="Q1", leftentityid="Q2"), Edge(2, rightentityid="Q1", leftentityid="Q2")])
        SemanticGraph([Edge(2, Q2-None->Q1), Edge(3, Q2-None->Q1), Edge(4, Q2-None->Q1)])
        """
        self.edges = EdgeList()
        self.tokens = tokens if tokens else []
        self.free_entities = free_entities if free_entities else []
        if edges:
            self.edges.extend(edges)

        self.denotations = []
        self.denotation_classes = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.tokens[:5]}, {self.edges}, {self.free_entities})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.edges}, {len(self.free_entities)})"

    def __copy__(self):
        return SemanticGraph(edges=[copy(e) for e in self.edges], tokens=self.tokens, free_entities=copy(self.free_entities))

    def get_ungrounded_edges(self):
        return [edge for edge in self.edges if not edge.grounded]


def graph_format_update(g):
    """
    Moves modifiers into separate edges.

    :param g:
    :return:
    >>> graph_format_update({"edgeSet":[{'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'argmax':'time'}]})
    {'edgeSet': [{'type': 'time', 'kbID': 'P585v', 'argmax': 'time'}, {'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct'}], 'entities': []}
    >>> graph_format_update({"edgeSet":[{'type': 'time', 'argmax':'time'}]})
    {'edgeSet': [{'type': 'time', 'argmax': 'time', 'kbID': 'P585v'}], 'entities': []}
    >>> graph_format_update({"edgeSet":[{'type': 'iclass', 'kbID': 'P31v', 'canonical_right': ['MTV Movie award', 'award', 'MTV annual movie award']}]})
    {'edgeSet': [{'type': 'iclass', 'kbID': 'P31v', 'canonical_right': 'award'}, {'type': 'iclass', 'kbID': 'P31v', 'canonical_right': 'MTV Movie award'}, {'type': 'iclass', 'kbID': 'P31v', 'canonical_right': 'MTV annual movie award'}], 'entities': []}
    """
    if if_graph_adheres(g, allowed_extensions={'multi_rel', 'qualifier_rel', 'v-structure'}):
        return g
    g = copy_graph(g, with_iclass=True)
    new_edgeset = []
    for edge in g.get('edgeSet', []):
        if edge.get("type") == 'time':
            edge['kbID'] = "P585v"
        elif "argmax" in edge:
            del edge['argmax']
            new_edgeset.append({"type": 'time', "kbID": "P585v", "argmax": 'time'})
        elif "argmin" in edge:
            del edge['argmin']
            new_edgeset.append({"type": 'time', "kbID": "P585v", "argmin": 'time'})
        elif "num" in edge:
            new_edgeset.append({"type": 'time', "kbID": "P585v", 'right': edge['num']})
            del edge['num']
        if edge.get("type") == 'iclass':
            for iclass in sorted(edge.get("canonical_right", []), key=len):
                new_edgeset.append({'type': 'iclass', 'kbID': edge.get("kbID"), 'canonical_right': iclass})
        else:
            new_edgeset.append(edge)
    g['edgeSet'] = new_edgeset
    return g


def graph_has_temporal(g):
    """
    Test if there are temporal relation in the graph.

    :param g: graph as a dictionary
    :return: True if graph has temporal relations, False otherwise
    """
    return any(any(edge.get(p) == 'time' for p in {'argmax', 'argmin', 'type'}) or 'num' in edge for edge in g.get('edgeSet', []))


def if_graph_adheres(g, allowed_extensions=set()):
    """
    Test if the given graphs only uses the allowed extensions.

    :param g: graphs a dictionary with an edgeSet
    :param allowed_extensions: a set of allowed extensions
    :return: True if graph uses only allowed extensions, false otherwise
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}]}, allowed_extensions=set())
    True
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'v-structure'}]}, allowed_extensions=set())
    False
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','argmin': 'time'}]}, allowed_extensions=set())
    False
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'time'}]}, allowed_extensions={'temporal'})
    True
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P17v','right': ['Iceland']}, {'kbID':'P31v'}]}, allowed_extensions=set())
    False
    >>> if_graph_adheres({'edgeSet': [{'kbID': 'P512q', 'rightkbID': 'Q8027', 'type': 'reverse'}], 'filter': 'importance'}, allowed_extensions=['multi_rel', 'temporal', 'qualifier_rel', 'v-structure'])
    False
    """
    allowed_extensions = set(allowed_extensions)
    if 'v-structure' not in allowed_extensions and 'v-structure' in {e.get('type', "direct") for e in g.get('edgeSet', [])}:
        return False
    if 'temporal' not in allowed_extensions and graph_has_temporal(g):
        return False
    if 'hopUp' not in allowed_extensions and any('hopUp' in e for e in g.get('edgeSet', [])):
        return False
    if 'hopDown' not in allowed_extensions and any('hopDown' in e for e in g.get('edgeSet', [])):
        return False
    if 'qualifier_rel' not in allowed_extensions and any(e.get('kbID', "").endswith('q') for e in g.get('edgeSet', [])):
        return False
    if 'multi_rel' not in allowed_extensions and len(g.get('edgeSet', [])) > 1:
        return False
    if 'filter' not in allowed_extensions and 'filter' in g:
        return False
    if 'iclass' not in allowed_extensions and any(edge.get("type") == 'iclass' for edge in g.get('edgeSet', [])):
        return False
    return True


def get_property_str_representation(edge, property2label,
                                    use_placeholder=False,
                                    mind_direction=True,
                                    include_modifiers=True,
                                    include_all_hop_labels=True
                                    ):
    """
    Construct a string representation of a label using the property to label mapping.

    :param edge: edge to translate
    :param property2label: property id to label mapping
    :param use_placeholder: if an named entity should be included or just a placeholder, doesn't affect classes
    :param mind_direction: if the direction of the edge should be accounted for
    :return: a string representation of an edge
    >>> get_property_str_representation({'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}, {'P17': "country"})
    'country Iceland'
    >>> get_property_str_representation({'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}, {'P17': "country"}, use_placeholder=True)
    'country <e>'
    >>> get_property_str_representation({'kbID': 'P17v','right': ['Iceland'],'rightkbID': 'Q189','type': 'direct'}, {'P17': {"label":"country", "altlabel": ["land", "province"]}}, use_placeholder=True)
    'country <e>'
    >>> get_property_str_representation({'kbID': 'P17487v','right': ['Iceland'],'rightkbID': 'Q1859','type': 'direct'}, {'P17': "country"}, use_placeholder=True)
    '_UNKNOWN <e>'
    >>> get_property_str_representation({'argmax':'time','type': 'time'}, {'P17': "country"}, use_placeholder=True)
    '<a> _UNKNOWN'
    >>> get_property_str_representation({'type': 'time', 'kbID': 'P585v', 'argmax': 'time'}, {'P585': "point in time"}, use_placeholder=True)
    '<a> point in time'
    >>> get_property_str_representation({'type': 'time', 'kbID': 'P585v', 'right': ['2012']}, {'P585': "point in time"}, use_placeholder=True)
    'point in time <n>'
    >>> get_property_str_representation({'kbID': 'P31v',   'right': ['Australia'], \
   'rightkbID': 'Q408', 'type': 'reverse'}, {'P31': "instance of"}, use_placeholder=True)
    '<e> instance of'
    >>> get_property_str_representation({'kbID': 'P31v',   'right': ['Australia'], \
   'rightkbID': 'Q408', 'type': 'reverse'}, {'P31': "instance of"}, use_placeholder=True, mind_direction=False)
    '<r> instance of <e>'
    >>> get_property_str_representation({'kbID': 'P31v',  'right': ['language'], \
   'rightkbID': 'Q408', 'type': 'class'}, {'P31': "instance of"}, use_placeholder=True, mind_direction=False)
    'language'
    >>> get_property_str_representation({'kbID': 'P31v',  'canonical_right': 'language communication method', \
   'rightkbID': 'Q408', 'type': 'iclass'}, {'P31': "instance of"}, use_placeholder=True, mind_direction=False)
    'language communication method'
    >>> get_property_str_representation({'type': 'iclass', 'kbID': 'P31v', 'canonical_right': 'currency'}, {'P31': "instance of"}, use_placeholder=True, mind_direction=False)
    'currency'
    >>> get_property_str_representation({'type': 'iclass', 'kbID': 'P31v', 'canonical_right': ['MTV Movie award', 'award', 'MTV annual movie award']}, {'P31': "instance of"}, use_placeholder=True, mind_direction=False)
    'award'
    >>> get_property_str_representation({'argmin': 'time', 'kbID': 'P17s',  'right': ['IC'], \
    'rightkbID': 'Q189', 'canonical_right': 'Iceland', 'type': 'direct'}, {'P17': "country"}, use_placeholder=False)
    '<i> country Iceland'
    >>> get_property_str_representation({'hopUp': 'P131v', 'kbID': 'P69s',  'right': ['Missouri'], \
    'rightkbID': 'Q189', 'type': 'direct'}, {'P69': "educated at", 'P131': 'located in'}, use_placeholder=True)
    'educated at <x> located in <e>'
    >>> get_property_str_representation({'hopUp': 'P17v', 'kbID': 'P140v',  'right': ['Russia'], \
    'rightkbID': 'Q159', 'type': 'reverse'}, {'P17': "country", 'P140': 'religion'}, use_placeholder=True)
    '<x> country <e> religion'
    >>> get_property_str_representation({'hopUp': None, 'kbID': 'P140v',  'right': ['Russia'], \
    'rightkbID': 'Q159', 'type': 'reverse'}, {'P17': "country", 'P140': 'religion'}, use_placeholder=True)
    '<e> religion'
    >>> get_property_str_representation({'kbID': 'P453q', 'right': ['Natalie', 'Portman'], 'type': 'reverse', 'hopUp': 'P161v', 'rightkbID': 'Q37876', 'canonical_right': 'Natalie Portman'},  {'P453': "character role", "P161": "cast member" }, use_placeholder=True)
    '<x> cast member <e> character role'
    >>> get_property_str_representation({'kbID': 'P453q', 'right': ['Natalie', 'Portman'], 'type': 'reverse', 'hopUp': 'P161v', 'rightkbID': 'Q37876', 'canonical_right': 'Natalie Portman'},  {'P453': "character role", "P161": "cast member" }, use_placeholder=True, include_all_hop_labels=False)
    '<e> character role'
    >>> get_property_str_representation({'canonical_right': 'Indiana', 'hopUp': 'P1001v', 'kbID': 'P39v', 'type': 'direct'}, {'P39': 'position held', 'P1001':'applies to territorial jurisdiction'}, use_placeholder=True)
    'position held <x> applies to territorial jurisdiction <e>'
    >>> get_property_str_representation({'canonical_right': 'Facebook','hopDown': 'P17v','kbID': 'P150v','type': 'reverse'}, {'P17':'country', 'P150':'contains administrative territorial entity'}, use_placeholder=True)
    '<e> country <x> contains administrative territorial entity'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct'}, {'P361': 'part of'}, use_placeholder=True)
    'part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Meg Griffin', 'kbID': 'P161v', 'type': 'v-structure'}, {'P161': "cast member"}, use_placeholder=True)
    '<v> cast member <e>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'argmax':'time'}, {'P361': 'part of'}, use_placeholder=True)
    '<a> part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'argmax':'time'}, {'P361': 'part of'}, use_placeholder=True, include_modifiers=False)
    'part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'filter':'importance'}, {'P361': 'part of'}, use_placeholder=True)
    'part of <e> part of <x>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'direct', 'filter':'importance'}, {'P361': 'part of'}, use_placeholder=False)
    'part of Washington Redskins part of <x>'
    >>> get_property_str_representation({'kbID': 'P69s',  'right': ['Missouri'], \
    'rightkbID': 'Q189', 'type': 'direct', 'num': '2012'}, {'P69': "educated at", 'P131': 'located in'}, use_placeholder=True)
    'educated at <e> <n>'
    >>> get_property_str_representation({'canonical_right': 'Washington Redskins', 'hopDown': 'P361v', 'kbID': 'P361v', 'type': 'reverse'}, {'P361': 'part of'}, use_placeholder=True, mind_direction=False, include_all_hop_labels=False)
    '<r> part of <e>'
    >>> get_property_str_representation({'label': 'netflix genres', 'type': 'direct'},{'P17': "country"}, use_placeholder=True)
    'netflix genres <e>'
    """
    e_type = edge.get('type', 'direct')
    if "label" in edge:
        property_label = edge["label"]
    else:
        p_meta = property2label.get(edge.get('kbID', " ")[:-1], {})
        property_label = base_objects.unknown_el
        if type(p_meta) == dict and len(p_meta) != 0:
            property_label = p_meta.get("label", "") # + " " + " ".join(p_meta.get("altlabel", "")[:5])
            property_label = property_label.strip()
        elif type(p_meta) == str:
            property_label = p_meta

    e_arg, num = "", ""
    if e_type == 'v-structure':
        property_label = "<v> " + property_label
    elif e_type == 'reverse' and not mind_direction:
        property_label = "<r> " + property_label
    if include_modifiers:
        e_arg = '<a> ' if 'argmax' in edge else '<i> ' if 'argmin' in edge else ""
        num = "<n>" if "num" in edge else ""
        # if 'filter' in edge and edge['filter'] == 'importance':
        #     e_arg += "<f> "
    hopUp_label, hopDown_label = '', ''
    if include_all_hop_labels and 'hopUp' in edge and edge['hopUp']:
        hopUp_label = "<x> {} ".format(property2label.get(edge['hopUp'][:-1],
                                                          base_objects.unknown_el))
    elif include_all_hop_labels and 'hopDown' in edge and edge['hopDown']:
        hopDown_label = " {} <x>".format(property2label.get(edge['hopDown'][:-1],
                                                            base_objects.unknown_el))
    if e_type.endswith("class"):
        property_label = ""
    if use_placeholder and not e_type.endswith("class"):
        if e_type == "time":
            entity_name = "<n>" if 'right' in edge else ""
        else:
            entity_name = "<e>"
    else:
        if "canonical_right" in edge:
            entity_name = " , ".join(sorted(edge["canonical_right"], key=len)[:1]) if type(edge["canonical_right"]) is list else edge["canonical_right"]
        else:
            entity_name = " ".join(edge.get('right', []))
    base_pattern = "{2}{3}{1}{4} {0} {5}" if mind_direction and e_type in {'reverse'} else "{2}{0} {3}{1}{4} {5}"
    str_representation = base_pattern.format(property_label, entity_name, e_arg, hopUp_label, hopDown_label, num)
    return str_representation.strip()


def replace_entities_in_instance(sentence_tokens, graph_set):
    """
    Replace entities in a sentences given a list of graphs.

    :param sentence_tokens: list of tokens
    :param graph_set: list of graphs
    :return: list of tokens with replaced entities
    >>> replace_entities_in_instance(['This', 'can', 'include', 'erotic', 'and', 'nude', 'modeling', ',', 'pornography', \
    ',', 'escorting', ',', 'and', 'in', 'some', 'cases', 'prostitution', '.'], [{'entities': [(['prostitution'], 'NNP'), (['escorting'], 'NNP')], 'edgeSet': [{'type': 'direct', 'kbID': 'P264v'}]}])
    ['This', 'can', 'include', 'erotic', 'and', 'nude', 'modeling', ',', 'pornography', ',', '<e>', ',', 'and', 'in', 'some', 'cases', '<e>', '.']
    >>> replace_entities_in_instance(['what', 'is', 'the', 'president', 'of', 'brazil', '?'],\
     [{'edgeSet': [{'canonical_right': 'Brazil', 'kbID': 'P35v','right': ['Brazil'],'rightkbID': 'Q155','type': 'reverse'}, \
     {'canonical_right': 'president','hopUp': 'P279v','kbID': 'P1308v','right': ['president'],'rightkbID': 'Q30461','type': 'reverse'},\
     {'canonical_right': ['sociologist','diplomat','economist','politician','trade unionist','writer'],'kbID': 'P106v','type': 'iclass'},\
     {'canonical_right': ['human'], 'kbID': 'P31v', 'type': 'iclass'}], 'entities': []}])
    ['what', 'is', 'the', 'president', 'of', '<e>', '?']
    >>> replace_entities_in_instance(['what', 'is', 'the', 'president', 'of', 'brazil', '?'],\
     [{'edgeSet': [{'kbID': 'P35v', 'type': 'reverse'}, \
     {'hopUp': 'P279v','kbID': 'P1308v','type': 'reverse'}]}])
    ['what', 'is', 'the', 'president', 'of', 'brazil', '?']
    """
    graph_entities = {}
    for g in graph_set:
        entities = [entity if type(entity) == dict else {"type": entity[1], "tokens": entity[0]} for entity in
                    g.get('entities', [])]
        # First add entities that are not features in edges yet
        graph_entities.update({" ".join(entity.get("tokens", [])): entity for entity in entities})
        # Add entities from the edges
        for e in g.get("edgeSet", []):
            if e.get("type") != "iclass" and e.get('right'):
                e_type = "NNP"
                e_relations = {e.get("kbID"), e.get("hopUp"), e.get("hopDown")}
                if e.get("type") in {"class", "iclass", "time"} or e_relations & {"P31v", "P106v", "P279v"}:
                    e_type = "NN"
                graph_entities[" ".join(e.get('right', []))] = {"tokens": e.get('right', []), "type": e_type}
    sentence_tokens = replace_entities(sentence_tokens, graph_entities.values())
    return sentence_tokens


def replace_entities(sentence_tokens, entities):
    """
    Replaces an entity participating in the first relation in the graph with <e>

    :param sentence_tokens: 
    :param entities: 
    :return: the list of tokens
    >>> replace_entities(['where', 'are', 'the', 'nfl', 'redskins', 'from', '?'], [{'linkings': [],'tokens': ['Nfl', 'Redskins'],'type': 'NNP'}])
    ['where', 'are', 'the', '<e>', 'from', '?']
    >>> replace_entities(['what', 'was', 'vasco', 'nunez', 'de', 'balboa', 'original', 'purpose', 'of', 'his', 'journey', '?'], \
    [{'linkings': [],'tokens': ['Vasco', 'Nunez', 'De', 'Balboa'],'type': 'NNP'}, {'linkings': [],'tokens': ['journey'],'type': 'NN'}])
    ['what', 'was', '<e>', 'original', 'purpose', 'of', 'his', 'journey', '?']
    >>> replace_entities("What movies did Natalie Portman and Johnny Cash played in ?".split(), \
    [{'linkings': [],'tokens': ['Natalie', 'Portman'],'type': 'NNP'}, {'linkings': [],'tokens': ['Johnny', 'Cash'],'type': 'NNP'}])
    ['What', 'movies', 'did', '<e>', 'and', '<e>', 'played', 'in', '?']
    >>> replace_entities("what is the upper house of the house of representatives ?".split(), [{'linkings': [],'tokens': ['House', 'Of', "Representatives"],'type': 'NNP'}])
    ['what', 'is', 'the', 'upper', 'house', 'of', 'the', '<e>', '?']
    >>> replace_entities("Where is Mount McKinley ?".split(), [{'linkings': [],'tokens': ["Mount","McKinley"],'type': 'NNP'}])
    ['Where', 'is', '<e>', '?']
    >>> replace_entities(['what', 'are', 'the', 'names', 'of', 'michael', 'jackson', 'movies', '?'], \
    [{'linkings': [],'tokens': ['Michael', 'Jackson'],'type': 'NNP'}, {'linkings': [],'tokens': ['michael', 'jackson', 'movies'],'type': 'NNP'}])
    ['what', 'are', 'the', 'names', 'of', '<e>', 'movies', '?']
    >>> replace_entities(['what', 'are', 'the', 'names', 'of', 'michael', 'jackson'], \
    [{'linkings': [],'tokens': ['Michael', 'Jackson'],'type': 'NNP'}, {'linkings': [],'tokens': ['names'],'type': 'NNP'}])
    ['what', 'are', 'the', '<e>', 'of', '<e>']
    """
    entities = [e for e in entities if e.get("type") != "NN"]
    entities = [[t.lower() for t in e.get("tokens", [])] for e in entities]
    new_tokens = sentence_tokens
    for entity in entities:
        new_tokens = replace_entity(new_tokens, entity)
    return new_tokens


def replace_entity(sentence_tokens, entity):
    new_tokens = []
    entity_pos = 0
    for i, t in enumerate(sentence_tokens):
        if entity_pos == len(entity) or t.lower() != entity[entity_pos]:
            if entity_pos > 0:
                if entity_pos == len(entity):
                    new_tokens.append("<e>")
                else:
                    new_tokens.extend(entity[:entity_pos])
                entity_pos = 0
            new_tokens.append(t)
        else:
            entity_pos += 1
    if entity_pos > 0:
        if entity_pos == len(entity):
            new_tokens.append("<e>")
        else:
            new_tokens.extend(entity[:entity_pos])
    return new_tokens


def normalize_tokens(g):
    """
    Normalize a tokens of the graph by setting it to lower case and removing any numbers.

    :param g: graph to normalize
    :return: graph with normalized tokens
    >>> normalize_tokens({'tokens':["Upper", "Case"]})
    {'tokens': ['upper', 'case']}
    >>> normalize_tokens({'tokens':["He", "started", "in", "1995"]})
    {'tokens': ['he', 'started', 'in', '0']}
    """
    tokens = g.get('tokens', [])
    g['tokens'] = [re.sub(r"\d+", "0", t.lower()) for t in tokens]
    return g


def get_graph_first_edge(g):
    """
    Get the first edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :return: an edge as a dictionary
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}], 'entities': []}) == {'right':[4,5,6]}
    True
    >>> get_graph_first_edge({})
    {}
    >>> get_graph_first_edge({'edgeSet':[]})
    {}
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}, {'right':[8]}], 'entities': []}) == {'right':[4,5,6]}
    True
    """
    return g["edgeSet"][0] if 'edgeSet' in g and g["edgeSet"] else {}


def get_graph_last_edge(g, filter_out_types=set()):
    """
    Get the last edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :param filter_out_types: a set fo edge types to filter out, if not empty the last edge of the specified type is returned
    :return: an edge as a dictionary
    >>> get_graph_last_edge({'edgeSet': [{'right':[4,5,6]}, {'right':[8], 'type': 'iclass'}], 'entities': []}) ==\
    {'right': [8], 'type': 'iclass'}
    True
    >>> get_graph_last_edge({'edgeSet': [{'right':[4,5,6], 'type': 'direct'}, {'right':[8], 'type': 'iclass'}], 'entities': []}, filter_out_types={'iclass'})
    {'right': [4, 5, 6], 'type': 'direct'}
    >>> get_graph_last_edge({'edgeSet': [], 'entities': []}, filter_out_types={'iclass', 'v-structure', 'reverse', 'class'})
    {}
    >>> get_graph_last_edge({'edgeSet': [{'right':[4,5,6], 'type': 'reverse'}], 'entities': []}, filter_out_types={'iclass', 'v-structure', 'reverse', 'class'})
    {}
    >>> get_graph_last_edge({'edgeSet': [{'right':[4,5,6]}], 'entities': []}, filter_out_types={'iclass', 'v-structure', 'reverse', 'class'})
    {'right': [4, 5, 6]}
    """
    if 'edgeSet' not in g or len(g["edgeSet"]) == 0:
        return {}
    if len(filter_out_types) == 0:
        return g["edgeSet"][-1] if 'edgeSet' in g and g["edgeSet"] else {}
    else:
        for i in range(len(g["edgeSet"]) - 1, -1, -1):
            edge = g["edgeSet"][i]
            if edge.get("type") not in filter_out_types:
                return edge
        return {}


def construct_graphs(tokens, entities):
    """
    Deprecated
    """
    entity_powerset = itertools.chain.from_iterable(itertools.combinations(entities, n) for n in range(1, len(entities)+1))
    graphs = []
    for entity_set in entity_powerset:
        g = {'edgeSet': [], 'tokens': tokens}
        for entity in entity_set:
            g['edgeSet'].append({'right':entity})
        graphs.append(g)
    return graphs


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
