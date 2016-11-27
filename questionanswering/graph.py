import itertools
import nltk


def extract_entities_from_tagged(annotated_tokens, tags):
    """
    The method takes a list of tokens annotated with the Stanford NE annotation scheme and produces a list of entites.

    :param annotated_tokens: list of tupels where the first element is a token and the second is the annotation
    :return: list of entities each represented by the corresponding token ids

    Tests:
    >>> extract_entities_from_tagged([('what', 'O'), ('character', 'O'), ('did', 'O'), ('natalie', 'PERSON'), ('portman', 'PERSON'), ('play', 'O'), ('in', 'O'), ('star', 'O'), ('wars', 'O'), ('?', 'O')], tags={'PERSON'})
    [['natalie', 'portman']]
    >>> extract_entities_from_tagged([('Who', 'O'), ('was', 'O'), ('john', 'PERSON'), ('noble', 'PERSON')], tags={'PERSON'})
    [['john', 'noble']]
    >>> extract_entities_from_tagged([(w, 'NE' if t != 'O' else 'O') for w, t in [('Who', 'O'), ('played', 'O'), ('Aragorn', 'PERSON'), ('in', 'O'), ('the', 'ORG'), ('Hobbit', 'ORG'), ('?', 'O')]], tags={'NE'})
    [['Aragorn'], ['the', 'Hobbit']]
    """
    vertices = []
    current_vertex = []
    for i, (w, t) in enumerate(annotated_tokens):
        if t in tags:
            current_vertex.append(w)
        elif len(current_vertex) > 0:
            vertices.append(current_vertex)
            current_vertex = []
    if len(current_vertex) > 0:
        vertices.append(current_vertex)
    return vertices

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def extract_entities(tokens_ne, tokens_pos):
    """
    Extract entities from the NE tags and POS tags of a sentence. Regular nouns are lemmatized to get rid of plurals.

    :param tokens_ne: list of NE tags.
    :param tokens_pos: list of POS tags.
    :return: list of entities in the order: NE>NNP>NN
    >>> extract_entities([('who', 'O'), ('are', 'O'), ('the', 'O'), ('current', 'O'), ('senators', 'O'), ('from', 'O'), ('missouri', 'LOCATION'), ('?', 'O')], [('who', 'WP'), ('are', 'VBP'), ('the', 'DT'), ('current', 'JJ'), ('senators', 'NNS'), ('from', 'IN'), ('missouri', 'NNP'), ('?', '.')])
    [['Missouri'], ['senator']]
    >>> extract_entities([('what', 'O'), ('awards', 'O'), ('has', 'O'), ('louis', 'PERSON'), ('sachar', 'PERSON'), ('won', 'O'), ('?', 'O')], [('what', 'WDT'), ('awards', 'NNS'), ('has', 'VBZ'), ('louis', 'NNP'), ('sachar', 'NNP'), ('won', 'NNP'), ('?', '.')])
    [['Louis', 'Sachar'], ['award']]
    >>> extract_entities([('who', 'O'), ('was', 'O'), ('the', 'O'), ('president', 'O'), ('after', 'O'), ('jfk', 'O'), ('died', 'O'), ('?', 'O')], [('who', 'WP'), ('was', 'VBD'), ('the', 'DT'), ('president', 'NN'), ('after', 'IN'), ('jfk', 'NNP'), ('died', 'VBD'), ('?', '.')])
    [['Jfk'], ['president']]
    """
    nes = extract_entities_from_tagged([(w, 'NE' if t != 'O' else 'O') for w, t in tokens_ne], ['NE'])
    nns = extract_entities_from_tagged(tokens_pos, ['NN', 'NNS'])
    nnps = extract_entities_from_tagged(tokens_pos, ['NNP', 'NNPS'])
    ne_vertices = nes
    vertices = []
    for nn in nnps:
        if not any(n in v for n in nn for v in vertices + ne_vertices):
            ne_vertices.append(nn)
    for nn in nns:
        if not any(n.title() in v for n in nn for v in vertices + ne_vertices):
            nn = [lemmatizer.lemmatize(n) for n in nn]
            vertices.append(nn)
    ne_vertices = [[w.title() for w in ne] for ne in ne_vertices]
    return ne_vertices + vertices


def construct_graphs(tokens, entities):
    entity_powerset = itertools.chain.from_iterable(itertools.combinations(entities, n) for n in range(1, len(entities)+1))
    graphs = []
    for entity_set in entity_powerset:
        g = {'edgeSet': [], 'tokens': tokens}
        for entity in entity_set:
            g['edgeSet'].append({'left':[0], 'right':entity})
        graphs.append(g)
    return graphs


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())

