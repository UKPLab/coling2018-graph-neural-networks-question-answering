def retrieval_precision(gold, predicted):
    """

    :param gold:
    :param predicted:
    :return:
    >>> retrieval_precision({1,2,3},{2})
    1.0
    >>> "{:2d}".format(retrieval_precision({2}, {1,2,3}))
    0.33333
    """
    gold = set(gold)
    predicted = set(predicted)
    tp = len(gold & predicted)
    fp_tp = len(predicted)
    return tp/fp_tp


def retrieval_prec_rec_f1(gold, predicted):
    prec = retrieval_precision(gold, predicted)
    rec = retrieval_precision(predicted, gold)
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

