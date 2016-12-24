import abc
import logging


class QAModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def encode_data_instance(self, instance):
        """
        Encode a single data instance in a format acceptable by the model.
        A data instance is a list of possible graphs.

        :param instance: a list of possible graphs for a single question.
        :return: a tuple that represents the instance in the model format.
        """

    def test(self, data_with_gold, verbose=False):
        graphs, gold_answers = data_with_gold
        predicted_indices = self.apply_on_batch(graphs)
        successes = deque()
        avg_f1 = 0.0
        for i, sorted_indices in enumerate(tqdm.tqdm(predicted_indices, ascii=True, disable=(not verbose))):
            sorted_indices = deque(sorted_indices)
            if sorted_indices:
                retrieved_answers = []
                while not retrieved_answers and sorted_indices:
                    index = sorted_indices.popleft()
                    g = graphs[i][index]
                    retrieved_answers = wdaccess.query_graph_denotations(g)
                retrieved_answers = wdaccess.map_query_results(retrieved_answers)
                _, _, f1 = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers[i], retrieved_answers)
                if f1:
                    successes.append((i, f1, g))
                avg_f1 += f1
        avg_f1 /= len(gold_answers)
        print("Successful predictions: {} ({})".format(len(successes), len(successes)/len(gold_answers)))
        print("Average f1: {}".format(avg_f1))

    @abc.abstractmethod
    def apply_on_batch(self, data_batch):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_on_instance(self, instance):
        raise NotImplementedError


class TrainableQAModel(QAModel):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(TrainableQAModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def train(self, data_with_targets):
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_for_training(self, data_with_targets):
        raise NotImplementedError
