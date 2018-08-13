# Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering

## Entity linking with the Wikidata knowledge base

This is an accompanying repository for the our submission. It contains the code to replicate the experiments and train the models descirbed in the paper.
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
 
### Project structure:

<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>configs/</td><td>Configuration files for the experiments</td>
    </tr>
    <tr>
        <td>questionanswering/construction</td><td>Base classes for semantic graphs</td>
    </tr>
    <tr>
        <td>questionanswering/datasets</td><td>Datasets IO</td>
    </tr>
    <tr>
        <td>questionanswering/grounding</td><td>Grounding graphs in KBs</td>
    </tr>
    <tr>
        <td>questionanswering/models</td><td>Model definition and training scripts</td>
    </tr>
    <tr>
        <td>questionanswering/preprocessing</td><td>Mapping data sets to Wikidata</td>
    </tr>
    <tr>
        <td>resources/</td><td>Necessary resources</td>
    </tr>
</table>


#### Requirements:
* Python 3.6
* PyTorch 0.3.0 - [read here about installation](http://pytorch.org/)
* See `requirements.txt` for the full list of packages

### Reading the training data

1. Clone and install the wikipedia-access project with pip
2. Clone the main question-answering project
3. Download the data
4. Use the following code to read it:
```python
import json
from questionanswering.construction import sentence

with open("webqsp.examples.train.silvergraphs.json") as f:
    dataset = json.load(f,  object_hook=sentence.sentence_object_hook)
```
### License:
* Apache License Version 2.0
