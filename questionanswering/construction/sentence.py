import json
from copy import copy

from questionanswering.construction.graph import SemanticGraph, WithScore, Edge, DUMMY_EDGE, EdgeList

QUESTION_TYPES = {"location", "temporal", "object", "person", "other"}


class Sentence:
    def __init__(self,
                 input_text=None,
                 tagged=None,
                 entities=None):
        """
        A sentence object.

        :param input_text: raw input text as a string
        :param tagged: a list of dict objects, one per token, with the output of the POS and NER taggers, see utils
                      for more info
        :param entities: a list of tuples, where each tuple is an entity link (first position is the KB id and
                         the second position is the label)
        """
        self.input_text = input_text if input_text else ""
        self.tagged = tagged if tagged else []
        self.tokens = [t['originalText'] for t in self.tagged]
        self.entities = [{k: e[k] for k in {'type', 'linkings', 'token_ids'}} for e in entities] if entities else []
        self.entities += [{'type': 'YEAR', 'linkings': [(t['originalText'], t['originalText'])], 'token_ids': [t['index']-1]}
                          for t in self.tagged if t['pos'] == 'CD' and t['ner'] == 'DATE']
        if get_question_type(self.input_text) == "person":
            self.entities.append({'type': 'NN', 'linkings': [("Q5", 'human')], 'token_ids': [0]})
        if get_question_type(self.input_text) == "location":
            self.entities.append({'type': 'NN', 'linkings': [("Q618123", 'geographical object')], 'token_ids': [0]})
        self.graphs = [WithScore(SemanticGraph(free_entities=self.entities, tokens=self.tokens), (0.0, 0.0, 0.0))]


class SentenceEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Sentence) or \
            isinstance(o, SemanticGraph) or \
                isinstance(o, Edge):
            return o.__dict__
        elif isinstance(o, EdgeList):
            return o._list
        return super(SentenceEncoder, self).default(o)


def sentence_object_hook(obj):
    if all(k in obj for k in Sentence().__dict__):
        s = Sentence()
        s.__dict__.update(obj)
        s.graphs = [WithScore(*l) for l in s.graphs]
        return s
    if all(k in obj for k in SemanticGraph().__dict__):
        g = SemanticGraph()
        g.__dict__.update(obj)
        g.edges = EdgeList()
        g.edges._list = obj['edges']
        return g
    if all(k in obj for k in DUMMY_EDGE.__dict__):
        e = copy(DUMMY_EDGE)
        e.__dict__.update(obj)
        return e
    return obj


def get_question_type(question_text):
    if question_text.startswith("when") or question_text.startswith("what year"):
        return "temporal"
    elif question_text.startswith("what") or question_text.startswith("which"):
        return "object"
    elif question_text.startswith("where"):
        return "location"
    elif question_text.startswith("who"):
        return "person"
    return "other"
