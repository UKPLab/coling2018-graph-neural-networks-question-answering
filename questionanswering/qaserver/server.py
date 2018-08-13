from flask import Blueprint, request, url_for, redirect

import json
import logging
import os
from datetime import datetime

from sklearn.manifold import TSNE
import numpy as np

from .. import config_utils, _utils
from ..wikidata import queries
from grounding import staged_generation
from ..models import pytorchmodel_impl

qaserver = Blueprint("qa_server", __name__, static_folder='static')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logger.setLevel(logging.DEBUG)

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)
config = config_utils.load_config(os.path.join(module_location, "..", "..", "configs/qaserver_config.yaml"))
queries.GLOBAL_RESULT_LIMIT = 200

qa_model = getattr(pytorchmodel_impl, config['model']['class'])(parameters=config['model'], logger=logger)
qa_model.load_from_file(config['model']['model.file'])
relations = list({k for k, v in queries.property2label.items() if v.get("type") not in queries.BLACKSET_PROPERTY_OBJECT_TYPES and v.get("freq", 0) > 100} - queries.property_blacklist)
logger.debug("Number of relations in the bank: {}".format(len(relations)))
relation_labels = [queries.property2label[r].get("label") for r in relations]
relationlabel2id = {rel:i for i, rel in enumerate(relations)}

with open(module_location + "/../../data/generated/webquestions.examples.train.utterances.tagged.json") as f:
    dataset_tagged = json.load(f)

disk_logger = None
if 'logger' in config and 'file' in config['logger']:
    frmt = logging.Formatter(fmt='%(asctime)s %(message)s')
    disk_logger = logging.getLogger("disk_logger")

    disk_logger.setLevel(logging.INFO)
    if not config['logger']['file'].endswith(".log"):
        config['logger']['file'] += ".log"
    fn = config['logger']['file'].replace(".log", "_{}_{}.log".format(datetime.today().month, datetime.today().year))
    fh = logging.FileHandler(filename=fn)
    fh.setLevel(logging.INFO)
    fh.setFormatter(frmt)
    disk_logger.addHandler(fh)
    logger.debug("Logging errors to {}".format(config['logger']['file']))


@qaserver.route("/")
def hello():
    return redirect(url_for('qa_server.static', filename='index.html'))


@qaserver.route("/answer/", methods=['GET', 'POST'])
def answer_question():
    if request.method == 'POST':
        question_text = request.json.get("question")
        logger.debug("Processing a answer request")
        log = {}
        ungrounded_graph = construct_ungrounded_graph(question_text)
        log['ug'] = ungrounded_graph
        chosen_graphs = generate_grounded_graphs(ungrounded_graph)
        log['chosen_graphs'] = chosen_graphs
        logger.debug("Label")
        model_answers, model_answers_labels = evaluate_chosen_graphs(chosen_graphs)
        log['model_answers'] = model_answers
        log['model_answers_labels'] = model_answers_labels
        return json.dumps((model_answers, model_answers_labels))
    return "No answer"


@qaserver.route("/answerforqald/", methods=['GET', 'POST'])
def answer_qaldquestion():
    """
    This method is here for the QALD competition, don't change it.

    :return: a list of answers in a form of Wikidata ids
    """
    if request.method == 'POST':
        logger.setLevel(logging.DEBUG)
        logger.debug("Processing a QALD request")
        question_text = request.form.get("question").strip()
        ungrounded_graph = construct_ungrounded_graph(question_text)
        chosen_graphs = generate_grounded_graphs(ungrounded_graph)
        model_answers, _ = evaluate_chosen_graphs(chosen_graphs, label_answers=False)
        logger.debug("Processing finished, sending the answers")
        return json.dumps(model_answers)
    return "No answer"


@qaserver.route("/answerforqalddataset/", methods=['GET', 'POST'])
def answer_qalddataset():
    """
    This method is here for the QALD competition, don't change it.

    :return: a list of lists of answers in a form of Wikidata ids
    """
    if request.method == 'POST':
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Processing a QALD request")
        questions = request.json
        logger.debug("Dataset length: {}".format(len(questions)))
        dataset_answers = []
        for question in questions:
            ungrounded_graph = construct_ungrounded_graph(question)
            chosen_graphs = generate_grounded_graphs(ungrounded_graph)
            model_answers, _ = evaluate_chosen_graphs(chosen_graphs, label_answers=False)
            dataset_answers.append(model_answers)
        logger.debug("Processing finished, sending the answers, {} answer sets".format(len(dataset_answers)))
        return json.dumps(dataset_answers)
    return "No answer"


@qaserver.route("/ungroundedgraph/", methods=['GET', 'POST'])
def send_ungrounded_graph():
    if request.method == 'POST':
        question_text = request.json.get("question")
        return json.dumps(construct_ungrounded_graph(question_text))
    return "No answer"


@qaserver.route("/groundedgraphs/", methods=['GET', 'POST'])
def send_grounded_graphs():
    if request.method == 'POST':
        ungrounded_graph = request.json
        chosen_graphs = generate_grounded_graphs(ungrounded_graph)
        return json.dumps({"graphs": chosen_graphs})
    return "No answer"


@qaserver.route("/encoderweigths/", methods=['POST'])
def send_encoder_weights():
    chosen_graphs = request.json.get("gs")
    chosen_graphs = [g[0] for g in chosen_graphs]
    ungrounded_graph = request.json.get("ug")
    tokens = ungrounded_graph.get("tokens", [])
    token_weights = qa_model.question_encoding_wieghts(tokens).tolist()
    tokens = qa_model._feature_extractors[0]._preprocess_sentence_tokens(tokens, chosen_graphs)
    return json.dumps({"cnnweights": token_weights, "tokens": tokens})


@qaserver.route("/semvectors/", methods=['POST'])
def send_sentence_semantic_vectors():
    ungrounded_graph = request.json.get("ug")
    tokens = ungrounded_graph.get("tokens", [])
    vectors_2dim = get_2d_sentence_vector_and_relation_vectors(tokens)
    return json.dumps({"relations": relation_names, "vectors": vectors_2dim.tolist()})


@qaserver.route("/evaluategraphs/", methods=['POST'])
def send_answers():
    if request.method == 'POST':
        chosen_graphs = request.json
        return json.dumps(evaluate_chosen_graphs(chosen_graphs))
    return "No answer"


@qaserver.route("/reporterror/", methods=['POST'])
def log_qa_error():
    if disk_logger:
        if "question" in request.json:
            disk_logger.info(request.json)
        return "OK"
    return "No logger"


def construct_ungrounded_graph(question_text):
    try:
        tagged = dataset_tagged[int(question_text)]
    except:
        logger.debug("Tagging: {}".format(question_text))
        tagged = _utils.get_tagged_from_server(question_text, caseless=question_text.islower())
    logger.debug("Tagged: {}".format(tagged))
    logger.debug("Extract entities")
    ungrounded_graph = {'tokens': [t['word'] for t in tagged],
                        'tagged': tagged,
                        'edgeSet': []}
    # ungrounded_graph = entity_linking.link_entities_in_graph(ungrounded_graph, entity_options=5)
    return ungrounded_graph


def generate_grounded_graphs(ungrounded_graph):
    logger.debug("Generate with model: {}".format(ungrounded_graph))
    chosen_graphs = staged_generation.generate_with_model(ungrounded_graph, qa_model, beam_size=10)
    for g in chosen_graphs:
        for e in g[0].get("edgeSet",[]):
            p_meta = queries.property2label.get(e.get('kbID', " ")[:-1], {})
            e['propertyName'] = p_meta.get("label", "")
    return chosen_graphs


def get_2d_sentence_vector_and_relation_vectors(tokens):
    sentence_embedding = qa_model.vectors_for_instance((tokens, [{}]))[0]
    tsnemodel = TSNE(n_components=2, n_iter=500, perplexity=5)
    vectors_2dim = tsnemodel.fit_transform(np.concatenate((sentence_embedding, relation_vectors), axis=0))
    return vectors_2dim


def evaluate_chosen_graphs(chosen_graphs, label_answers=True):
    model_answers = []
    logger.debug("Evaluate the chosen graphs")
    if chosen_graphs:
        j = 0
        while not model_answers and j < len(chosen_graphs):
            g = chosen_graphs[j]
            model_answers = queries.get_graph_denotations(g[0]).get("e1")
            j += 1
    model_answers_labels = []
    if label_answers:
        logger.debug("Label")
        model_answers_labels = [queries.label_entity(e) for e in model_answers]
    return model_answers, model_answers_labels


def init_vector_bank():
    logger.debug("Init vector bank")
    vectors = np.zeros((len(relations), qa_model._p['edge.embedding.size']))
    for i, relation in enumerate(relations):
        temp_edge = {'kbID': relation + "v", 'type': 'direct'}
        _, edge_vector, _ = qa_model.vectors_for_instance(([], [{"edgeSet": [temp_edge]}]))
        vectors[i] = edge_vector[:, 0]
    # for j, tagged in enumerate(dataset_tagged):
    #     tokens = [t for t, _, _ in tagged]
    #     sentence_vector, _ , _ = qa_model.vectors_for_instance((tokens, []))
    #     vectors[j] = sentence_vector
    logger.debug("Init vector bank. Finished")

    return vectors

relation_vectors = init_vector_bank()
relation_names = [queries.property2label[r].get("label") for r in relations]