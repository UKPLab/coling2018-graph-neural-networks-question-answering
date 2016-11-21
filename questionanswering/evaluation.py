def retrieval_precision(gold, predicted):
    """
    Compute retrieval precision on the given gold set and predicted set.
    Note that it doesn't take into account the order or repeating elements.

    :param gold: the set of gold retrieved elements
    :param predicted: the set of predicted elements
    :return: precision value
    >>> retrieval_precision({1,2,3},{2})
    1.0
    >>> retrieval_precision({2}, {1,2,3})
    0.3333333333333333
    """
    gold = set(gold)
    predicted = set(predicted)
    tp = len(gold & predicted)
    fp_tp = len(predicted)
    return tp/fp_tp


def retrieval_prec_rec_f1(gold, predicted):
    """
    Compute retrieval precision, recall and f-score. Note that it doesn't take into accounte the order
    and repeating elements.

    :param gold: the set of gold retrieved elements.
    :param predicted: the set of predicted elements.
    :return: a triple of precision, recall and f-score
    >>> retrieval_prec_rec_f1(['Star Wars', 'Black Swan', 'Thor', 'Leon'], ['Thor', 'Avengers', 'Iron Man'])
    (0.3333333333333333, 0.25, 0.28571428571428575)
    """
    prec = retrieval_precision(gold, predicted)
    rec = retrieval_precision(predicted, gold)
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

