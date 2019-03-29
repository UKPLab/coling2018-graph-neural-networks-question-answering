# Wikidata How To
## Creating a local SPARQL endpoint with a Wikidata dump

This is an accompanying instruction for our **COLING 2018 paper** ([pdf](http://aclweb.org/anthology/C18-1280)) and other work on KB at the [UKP lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/). 
In this document, we describe step by step how to create an RDF dump of Wikidata and install a local SPARQL endpoint that can be used with our QA models.

You will need a local Wikidata SPARQL endpoint in order to:
- Run the evaluation of the QA model on the public datasets (the system needs to query Wikidata for available relations and to retrieve the final answer)
- Run the system on arbitrary input.

If you just want to try out our system on an example question, feel free to use the online [demo](http://semanticparsing.ukp.informatik.tu-darmstadt.de:5000/question-answering/).

Disclaimer:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

 
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.informatik.tu-darmstadt.de/ukp/ukp_home/
  * https://www.tu-darmstadt.de

### Requirements:
* Minimum of 16 Gb of RAM available. (For importing the RDF dump into Virtuoso I would recommend at least 64 Gb) 

### Overview

The installation is completed in 3 steps. Each step takes a significant amount of time, so please plan ahead or un it over night.

#### 1. Create a local Wikidata RDF dump
Create a local Wikidata RDF dump with the [Wikidata Toolkit](https://www.mediawiki.org/wiki/Wikidata_Toolkit/Client). You need to use the 8.0.0 version, earlier version produce errors with the current Wikidatat dumps. The latest 9.0.0 release produces a different RDF mapping and won't work with our code! 

You can use the Wikidata toolkit configuration file `configs/wikidata-rdf/rdf_dumps.ini` that will exclude unnecessary languages and information from the RDF dump.
Run the following command to create the RDF from the latest Wikidata dump (which will be automatically downloaded):

```bash
java -jar wdtk-client-0.8.0-SNAPSHOT.jar --config rdf_dumps.ini
```  

The output should be foru files in the `rdf-snap` folder:  
```
rdf-snap/wikidata-instances.nt.gz
rdf-snap/wikidata-sitelinks.nt.gz
rdf-snap/wikidata-statements.nt.gz
rdf-snap/wikidata-terms.nt.gz
```

#### 2. Install Virtuosos opensource
Install the opensource version of [Virtuoso](http://vos.openlinksw.com/owiki/wiki/VOS/VOSDownload) database server. We used the 7.2.4.2 version. If the installation was successful, you should be able to start an instance with 
```
./virtuoso-opensource-install-7.2.4.2/bin/virtuoso-t -f -c virtuoso-opensource-install-7.2.4.2/var/lib/virtuoso/db/virtuoso.ini
```

There are some additional installation tips here: https://github.com/percyliang/sempre#virtuoso-graph-database

#### 3. Import the RDF dump into the Virtuoso
Import the RDF dump into the Virtuoso with the [bulk_loader](http://vos.openlinksw.com/owiki/wiki/VOS/VirtBulkRDFLoader).

Put the `*.graph` files from the `configs/wikidata-rdf/` folder into `rdf-snap/`. These configuration files will tell Virtuoso how to call each subgraph.

Start the interactive session (if you are not using the default 1111 port, change accordingly):
```bash
./virtuoso-opensource-install-7.2.4.2/bin/isql 1111
```
Load all data from `rdf-snap/` (this doesn't import the data yet, only schedules it):
```sql
SQL> ld_dir_all('rdf-snap/', '*.gz', 'http://wikidata.org/');
```

Check that the four RDF files were scheduled to be imported:
```sql
SQL> select * from DB.DBA.load_list;
```

Start the loader (you can go home now, this will take a while):
```sql
SQL> rdf_loader_run();
```

Open the SPARQL Query Editor in the browser `http://localhost:8890/sparql` and post the following query:
```SQL
SELECT DISTINCT ?g 
WHERE {
  GRAPH ?g { ?a ?b ?c  }
}
```
This show list all available graphs in your RDF storage. Check that the four Wikidata files were imported.



### License:
* Apache License Version 2.0
