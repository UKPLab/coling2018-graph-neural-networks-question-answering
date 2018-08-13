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
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
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


#### Requirements:
* Python 3.6
* PyTorch 0.3.0 - [read here about installation](http://pytorch.org/)
* See `requirements.txt` for the full list of packages

### Running the experiments from the paper:

[Coming soon]

### Using the pre-trained model:

[Coming soon]


### License:
* Apache License Version 2.0
