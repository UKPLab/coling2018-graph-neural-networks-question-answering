# Wikidata How To
## Creating a local SPARQL endpoint with a Wikidata dump

This is an accompanying instruction for our **COLING 2018 paper** ([pdf](http://aclweb.org/anthology/C18-1280)) and other work on KB at the [UKP lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/). 
In this document, we describe step by step how to create an RDF dump of Wikidata and install a local SPARQL endpoint that can be used with our QA models.

You will need a local Wikidata SPARQL endpoint in order to:
- Run the evaluation of the QA model on the public datasets (the system needs to query Wikidata for available relations and to retrieve the final answer)
- Run the system on arbitrary input.

If you just want to try out our system on an example questions 

Disclaimer:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

 
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.informatik.tu-darmstadt.de/ukp/ukp_home/
  * https://www.tu-darmstadt.de

### Requirements:
* Minimum of 16 Gb of RAM available. (For importing the RDF dump into Virtuoso I would recomend ) 

### Overview

The installation is completed in 3 steps:
1. Create a local Wikidata RDF dump with the [Wikidata Toolkit](https://www.mediawiki.org/wiki/Wikidata_Toolkit/Client)
2. Install the opensource version of [Virtuoso](http://vos.openlinksw.com/owiki/wiki/VOS/VOSDownload) database server
3. Import the RDF dump into the Virtuoso with the [bulk_loader](http://vos.openlinksw.com/owiki/wiki/VOS/VirtBulkRDFLoader)

Each step takes a significant amount of time, so please plan ahead or un it over night.


### License:
* Apache License Version 2.0
