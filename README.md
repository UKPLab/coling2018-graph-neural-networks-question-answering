# Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering

## Question Answering on Wikidata

This is an accompanying repository for our **COLING 2018 paper** ([pdf](http://aclweb.org/anthology/C18-1280)). 
It contains the code to provide additional information on the experiments and the models described in the paper.

We are working on improving the code to make it easy to exactly replicate the experiments and apply to new question answering data. 
We also plan to release a separate implementation of the gated graph neural networks.   

Disclaimer:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

 

Please use the following citation:

```
@InProceedings{C18-1280,
  author = 	"Sorokin, Daniil
		and Gurevych, Iryna",
  title = 	"Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering",
  booktitle = 	"Proceedings of the 27th International Conference on Computational Linguistics",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"3306--3317",
  location = 	"Santa Fe, New Mexico, USA",
  url = 	"http://aclweb.org/anthology/C18-1280"
}
```

### Paper abstract:
> The most approaches to Knowledge Base Question Answering are based on semantic parsing. In
  this paper, we address the problem of learning vector representations for complex semantic parses
  that consist of multiple entities and relations. Previous work largely focused on selecting the
  correct semantic relations for a question and disregarded the structure of the semantic parse: the
  connections between entities and the directions of the relations. We propose to use Gated Graph
  Neural Networks to encode the graph structure of the semantic parse. We show on two data sets
  that the graph networks outperform all baseline models that do not explicitly model the structure.
  The error analysis confirms that our approach can successfully process complex semantic parses.

Please, refer to the paper for more the model description and training details.
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, [personal page](https://daniilsorokin.github.io)
  * https://www.informatik.tu-darmstadt.de/ukp/ukp_home/
  * https://www.tu-darmstadt.de
 
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


### Requirements:
* Python 3.6
* PyTorch 0.3.0 - [read here about installation](http://pytorch.org/)
* See `requirements.txt` for the full list of packages
* Download and install the two internal packages that are not part of this project: 
    * `wikidata-access` ([zip](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/wikidata-access-master.zip)) for sending queries to a local Wikidata endpoint
    * `fackel` ([zip](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/fackel-master.zip)) for running the provided PyTorch models 
* A local copy of the Wikidata knowledge base in RDF format. See [here](WikidataHowTo.md) for more info on the Wikidata installation. (This step takes a lot of time!)
* A running instance of the Stanford CoreNLP server (https://stanfordnlp.github.io/CoreNLP/corenlp-server.html) for tokenisation and NE recognition. Do not forget to download English model. Test your CoreNLP server before starting the experiments. Replace the address [here](https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering/blob/06f406fd1a4a7e10902ea484e0707c8810e5e3e5/questionanswering/_utils.py#L24) with URL of your CoreNLP instance.

### Data sets:

#### Evaluation

##### WebQSP-WD
* The [WebQSP data set](https://www.microsoft.com/en-us/download/details.aspx?id=52763) is a manually corrected subset of the original [WebQuestions data set](https://nlp.stanford.edu/software/sempre/). Both data sets use the Freebase knowledge base.
* We map the WebQSP subset of WebQuestions to Wikidata and filter out questions that do not have an answer in Wikidata. See the [Readme file](WEBQSP_WD_README.md) for the data set and the [script for mapping to Wikidata](questionanswering/preprocessing/map_dataset_to_wikidata.py) for more details.
* Download the WebQSP-WD train and test partitions [here](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip). Please cite both the original and our work if you use it.

##### QALD-7 
* The original QALD-7 data set is available from the official [QALD2017 Challenge web-site](https://project-hobbit.eu/challenges/qald2017/qald2017-challenge-tasks/#task4).
* We have used the 80 open-questions from the QALD-7 train set for evaluation. 
You can this subset in the format accepted by our evaluation script [here](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/qald.examples.test.wikidata.json).
* Please site the [QALD2017 Challenge](https://project-hobbit.eu/challenges/qald2017/) and follow their data policy if you use this dataset.
 


### Running the full experiments (training and testing):

* Make sure you have a local Wikidata endpoint in place (you need it for the evaluation) and the rest of the requirements are satisfied.
* Download the training and evaluation data sets.
* Use config files from `configs/train_*` to specify which model to train with the training script.
* Use config files from `configs/*_eval_config.yaml` to specify which data set to use for the evaluation. 
* In the evaluation config file, specify the location of the Wikidata endpoint.
* `GPU_id` is always an integer that is the number of the GPU to use (The default value is 0). See PyTorch documentation for more info.  

#### Training and testing multiple models
1. Use the `train_random_models.sh` script to train a set of random models and test each of them in the WebQSP-WD data set. 
   The script takes the following parameters: `[config_file_path] [GPU_id]`
   
#### Train and test one model
1.  Run `python -m questionanswering.train_model [config_file_path] [random_seed] [GPU_id]`
2.  Run `python -m questionanswering.evaluate_on_test [model_file_path] configs/webqsp_eval_config.yaml [random_seed] [GPU_id]` to test on WebQSP-WD 

* The model output on test data is saved in `data/output/webqsp/[modeltype]/`, the aggregated macro-scores are saved into 
`data/output/webqsp/qa_experiments.csv`.

### Using the pre-trained model to reproduce the results from the paper:

1. Download the pre-trained models ([.zip](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/DS_COLING_2018_QA_models.zip)) and unpack them into `trainedmodels/` 
2. For the experiments in the paper, choose the model using the table below. 
3. Run  `python -m questionanswering.evaluate_on_test [model_file_path] configs/webqsp_eval_config.yaml [random_seed] [GPU_id]` to test on WebQSP-WD

Model file names corresponding to the reported results.
<table>
    <tr>
        <th>Model type</th><th>Model file name</th>
    </tr>   
    <tr>
        <td>STAGG</td><td>STAGGModel_2018-03-14_811799.pkl</td>
    </tr>   
    <tr>
        <td>OneEdge</td><td>OneEdgeModel_2018-03-14_194679.pkl</td>
    </tr>   
    <tr>
        <td>PooledEdges</td><td>PooledEdgesModel_2018-03-13_519272.pkl</td>
    </tr>   
    <tr>
        <td>GNN</td><td>GNNModel_2018-03-15_369757.pkl</td>
    </tr>   
    <tr>
        <td>GGNN</td><td>GNNModel_2018-03-13_113541.pkl</td>
    </tr>
</table>


### Generating training data (with weak supervision) 

[coming soon]

### License:
* Apache License Version 2.0
