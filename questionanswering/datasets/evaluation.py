def micro_avg_precision(guessed, correct, empty=None):
    """
    Tests:
    >>> micro_avg_precision(['A', 'A', 'B', 'C'],['A', 'C', 'C', 'C'])
    0.5
    >>> round(micro_avg_precision([0,0,0,1,1,1],[1,0,0,0,1,0], empty=0), 6)
    0.333333
    >>> round(micro_avg_precision([1,0,0,0,1,0],[0,0,0,1,1,1], empty=0), 6)
    0.5
    >>> round(micro_avg_precision([1,0,0,0,1,0],[], empty=0), 6)
    1.0
    """
    correctCount = 0
    count = 0

    idx = 0
    if len(guessed) == 0:
        return 1.0
    elif len(correct) == 0:
        return 1.0
    while idx < len(guessed):
        if guessed[idx] != empty:
            count += 1
            if guessed[idx] == correct[idx]:
                correctCount +=1
        idx +=1
    precision = 0
    if count > 0:
        precision = correctCount / count

    return precision


def prec_rec_f1(predicted_idx, gold_idx, empty_label = None):
    if len(predicted_idx) != len(gold_idx):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")
    label_y = gold_idx
    pred_labels = predicted_idx

    prec = micro_avg_precision(pred_labels, label_y, empty_label)
    rec = micro_avg_precision(label_y, pred_labels, empty_label)

    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


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
    >>> retrieval_precision({2,3,4,8}, {1,6,3})
    0.3333333333333333
    """
    gold = set(gold)
    predicted = set(predicted)
    tp = len(gold & predicted)
    fp_tp = len(predicted)
    return tp/fp_tp


def retrieval_tp_with_altlabels(gold, predicted_sets):
    """
    Compute rtrue positives on the given gold set and predicted set.
    Note that it doesn't take into account the order or repeating elements.

    :param gold: the set of gold retrieved elements
    :param predicted_sets: the set of predicted elements
    :return: number of true positives
    >>> retrieval_tp_with_altlabels({1,2,3},[[2,8,4], [1], [6,7], [12, 45]])
    2
    >>> retrieval_tp_with_altlabels({1,2,3},[[2,3,4], [8]])
    1
    """
    return sum(any(l in gold for l in label_set) for label_set in predicted_sets)


def retrieval_prec_rec_f1(gold, predicted):
    """
    Compute retrieval precision, recall and f-score. Note that it doesn't take into account the order
    and repeating elements.

    :param gold: the set of gold retrieved elements.
    :param predicted: the set of predicted elements.
    :return: a triple of precision, recall and f-score
    >>> retrieval_prec_rec_f1(['Star Wars', 'Black Swan', 'Thor', 'Leon'], ['Thor', 'Avengers', 'Iron Man'])
    (0.3333333333333333, 0.25, 0.28571428571428575)
    >>> retrieval_prec_rec_f1(['Star Wars', 'Black Swan', 'Thor', 'Leon'], [])
    (0.0, 0.0, 0.0)
    >>> retrieval_prec_rec_f1([1,2], [1,2,3])
    (0.6666666666666666, 1.0, 0.8)
    >>> retrieval_prec_rec_f1([1,2], [1])
    (1.0, 0.5, 0.6666666666666666)
    """
    prec = retrieval_precision(gold, predicted) if len(predicted) > 0 else 0.0
    rec = retrieval_precision(predicted, gold) if len(gold) > 0 else 1.0
    f1 = 0.0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


def retrieval_prec_rec_f1_with_altlabels(gold, predicted_sets):
    """
    Compute retrieval precision, recall and f-score for the case when each predicted entity has alternative labels.
    Note that it doesn't take into account the order
    and repeating elements.

    :param gold: the set of gold retrieved elements.
    :param predicted_sets: the set of predicted elements.
    :return: a triple of precision, recall and f-score
    >>> retrieval_prec_rec_f1_with_altlabels(['Star Wars', 'Black Swan', 'Thor', 'Leon'], [['thor','Thor','God of thunder'], ['Avengers', 'Defenders'], ['Iron Man']])
    (0.3333333333333333, 0.25, 0.28571428571428575)
    >>> retrieval_prec_rec_f1_with_altlabels(['Star Wars', 'Black Swan', 'Thor', 'Leon'], [])
    (1.0, 0.0, 0.0)
    >>> retrieval_prec_rec_f1_with_altlabels(['Star Wars', 'Black Swan', 'Thor', 'Leon'], [[],[]])
    (1.0, 0.0, 0.0)
    >>> retrieval_prec_rec_f1_with_altlabels(['Leon'], [['Black Swan'],['Leon']])
    (0.5, 1.0, 0.6666666666666666)
    >>> retrieval_prec_rec_f1_with_altlabels([], [['Black Swan'],['Leon']])
    (0.0, 1.0, 0.0)
    >>> retrieval_prec_rec_f1_with_altlabels(['Thor'], [['Thor'], ['Black Swan'],['Leon']])
    (0.3333333333333333, 1.0, 0.5)
    >>> retrieval_prec_rec_f1_with_altlabels(['Thor', 'Brian'], [['Brian'], ['Black Swan'],['Leon']])
    (0.3333333333333333, 0.5, 0.4)
    """
    tp = retrieval_tp_with_altlabels(gold, predicted_sets)
    prec = tp / len(predicted_sets) if len(predicted_sets) and sum(len(s) for s in predicted_sets) > 0 else 1.0
    rec = tp / len(gold) if len(gold) > 0 else 1.0
    f1 = 0.0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

