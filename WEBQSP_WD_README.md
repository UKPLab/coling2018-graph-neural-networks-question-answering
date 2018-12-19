# WebQSP-WD
v1.0, 2018-08

This data set contains a version of the WebQSP data set mapped to Wikidata.
Compared to the original data set, this resource only includes question for which at least some acceptable answer exists in Wikidata.
The test partition also includes the mapping of the output of an SMART-S entity linker that was originally produced for the WebQSP 
data set (see [Yih et al. 2016](http://www.aclweb.org/anthology/P16-2033) for details).  

Please consult the corresponding [code repository](https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering) 
and the [paper](http://aclweb.org/anthology/C18-1280) to learn more about how the data set was constructed and used.

WebQSP-WD is available at following location: 
https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip

The `input` folder contains the train and test partitions with answer ids mapped to Wikidata.
The `generated` folder contains automatically generated candidate graphs for the train partition that are needed to train a new model. Please consult the [paper](http://aclweb.org/anthology/C18-1280) for mode details.

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

### Reading in the data:

All data set files are in json format. For the files in the `generated` folder, please use the following code snippet. Make sure to download the [project](https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering) first.

```python
import json
from questionanswering.construction.sentence import sentence_object_hook

training_dataset = []
with open(path_to_train) as f:
    training_dataset = json.load(f,  object_hook=sentence_object_hook)

print("Graphs for the first question: ", training_dataset[0].graphs)
```

You do not need a Wikidata endpoint and the additional internal projects to read in the files. 


### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.informatik.tu-darmstadt.de/ukp/ukp_home/
  * https://www.tu-darmstadt.de

### License

* This data set is derived from the original [WebQuestions data set](https://nlp.stanford.edu/software/sempre/) 
and its subset [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763). WebQuestions was 
released under [CC-BY 4.0](http://creativecommons.org/licenses/by/4.0/).
* Please cite our paper if you use the data in your work.