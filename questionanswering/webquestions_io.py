import re


def get_answers_from_question(question_object):
    """
    Retrieve a list of answers from a question as encoded in the WebQuestions dataset.

    :param question_object: A question encoded as a Json object
    :return: A list of answers as strings
    """
    return re.findall("\(description \"?(.*?)\"?\)", question_object.get('targetValue'))