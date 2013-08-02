bach-uai13-code
===============

Code for "Hinge-loss Markov Random Fields: Convex Inference for Structured Prediction."
Stephen H. Bach, Bert Huang, Ben London, and Lise Getoor.
Uncertainty in Artificial Intelligence (UAI) 2013

Please cite this work as

	@inproceedings{bach:uai13,
		author = "Bach, Stephen H. and Huang, Bert and London, Ben and Getoor, Lise",
		title = "Hinge-loss {M}arkov Random Fields: Convex Inference for Structured Prediction",
		booktitle = "{Uncertainty in Artificial Intelligence (UAI)}",
		year = "2013"
	}

Instructions
=============

Prerequisites
-------------
This software depends on Java 6 and Maven 3 (http://maven.apache.org). Matlab and
Python (>=2.7) are also required to process the results.

To run Bayesian probabilistic matrix factorization for the preference-prediction
experiment, you must download and compile the Bayesian probabilistic tensor
factorization package (http://www.cs.cmu.edu/~lxiong/bptf/bptf.html) and place
the export_bptf directory in src/main/matlab.

PSL Library
-----------
The algorithms for these experiments are implemented in the PSL library, version 1.1
(https://github.com/linqs/psl/tree/1.1). It will be downloaded by Maven automatically.

Executing
---------
Each experiment from the paper can be run from one of the shell scripts in the root directory.

Data
====
All data is in the data directory. The four data sets are described and credited below.

Citeseer
--------
This data set contains a selection of the CiteSeer data set (http://citeseer.ist.psu.edu/).

These papers are classified into one of the following six classes:

	Agents
	AI
	DB
	IR
	ML
	HCI

The papers were selected in a way such that in the final corpus every paper cites or is
cited by at least one other paper. There are 3312 papers in the whole corpus. 

After stemming and removing stopwords we were left with a vocabulary of size 3703 unique words.
All words with document frequency less than 10 were removed.

The citeseer.content file contains descriptions of the papers in the following format:

	<paper_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the paper followed by binary
values indicating whether each word in the vocabulary is present (indicated by 1) or
absent (indicated by 0) in the paper. Finally, the last entry in the line contains the class
label of the paper.

The citeseer.cites file contains the citation graph of the corpus. Each line describes a
link in the following format:

	<ID of cited paper> <ID of citing paper>

Each line contains two paper IDs. The first entry is the ID of the paper being cited and
the second ID stands for the paper which contains the citation. The direction of the link
is from right to left. If a line is represented by "paper1 paper2" then the link is "paper2->paper1".

parse.py parses citeseer.content and citeseer.cites to create additional data files.
It is not necessary to run this again.

Please cite this data set as

	@Article{sen:aimag08,
		author       = "Sen, Prithviraj and Namata, Galileo Mark and Bilgic, Mustafa and Getoor, Lise and Gallagher, Brian and Eliassi-Rad, Tina",
		title        = "Collective Classification in Network Data",
		journal      = "AI Magazine",
		number       = "3",
		volume       = "29",
		pages        = "93--106",
		year         = "2008"
	}

Cora
----
This data set contains a selection of the Cora data set (http://www.research.whizbang.com/data).

The Cora dataset consists of Machine Learning papers.
These papers are classified into one of the following seven classes:

	Case_Based
	Genetic_Algorithms
	Neural_Networks
	Probabilistic_Methods
	Reinforcement_Learning
	Rule_Learning
	Theory

The papers were selected in a way such that in the final corpus every paper cites or is
cited by atleast one other paper. There are 2708 papers in the whole corpus. 

After stemming and removing stopwords we were left with a vocabulary of size 1433 unique words.
All words with document frequency less than 10 were removed.

The cora.content file contains descriptions of the papers in the following format:

	<paper_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the paper followed by binary
values indicating whether each word in the vocabulary is present (indicated by 1) or
absent (indicated by 0) in the paper. Finally, the last entry in the line contains the
class label of the paper.

The cora.cites file contains the citation graph of the corpus. Each line describes a link
in the following format:

	<ID of cited paper> <ID of citing paper>

Each line contains two paper IDs. The first entry is the ID of the paper being cited and
the second ID stands for the paper which contains the citation. The direction of the link
is from right to left. If a line is represented by "paper1 paper2" then the link is "paper2->paper1".

parse.py parses cora.content and cora.cites to create additional data files. It is not
necessary to run this again.

Please cite this data set as

	@Article{sen:aimag08,
		author       = "Sen, Prithviraj and Namata, Galileo Mark and Bilgic, Mustafa and Getoor, Lise and Gallagher, Brian and Eliassi-Rad, Tina",
		title        = "Collective Classification in Network Data",
		journal      = "AI Magazine",
		number       = "3",
		volume       = "29",
		pages        = "93--106",
		year         = "2008"
	}

Epinions
--------
This data set is a snowball sample from the Epinions trust data set
(http://snap.stanford.edu/data/soc-sign-epinions.html).

From the website:
"This is who-trust-whom online social network of a a general consumer review site Epinions.com.
Members of the site can decide whether to ''trust'' each other. All the trust relationships
interact and form the Web of Trust which is then combined with review ratings to determine
which reviews are shown to the user."

Please cite the original data set as

	@inproceedings{leskovec:chi10,
		author = "Leskovec, Jure and Huttenlocher, Daniel and Kleinberg, Jon},
		booktitle = "{28th ACM Conference on Human Factors in Computing Systems (CHI)}",
		pages = "1361--1370",
		title = "Signed Networks in Social Media",
		year = "2010"
	}

Jester
------
Data set of ratings of jokes (http://goldberg.berkeley.edu/jester-data/).

The original data (data/jester/ratings/jester-1.csv and data/jester/joketext/joketext.txt) are
parsed into the 50/50 test/train splits with 50% ratings observed and unobserved using
data/jester/ratings/parse.py and data/jester/joketext/computeSim.py. It is not necessary to
re-run these scripts. The users to model during training and testing are listed in
data/jester/users-te-1000.txt and data/jester/users-tr-1000.txt.

Please cite the original data set as

	@article{goldberg:ir01,
		author = "Goldberg, Ken and Roeder, Theresa and Gupta, Dhruv and Perkins, Chris",
		journal = "Information Retrieval",
		number = "2",
		pages = "133--151",
		title = "Eigentaste: A Constant Time Collaborative Filtering Algorithm",
		volume = "4",
		year = "2001"
	}

Olivetti
--------
Faces from the AT&T database.

Images are 64x64 and have been preprocessed to a vector format
(1 face per row) and pixel intensities have been normalized from 0-255 to 0-1.

Please cite this data set as

	@inproceedings{samaria:acv94,
		author = "Samaria, F. S. and Harter, A. C.",
		booktitle = "{Proceedings of 1994 IEEE Workshop on Applications of Computer Vision}",
		title = "Parameterisation of a stochastic model for human face identification",
		year = "1994"
	}

Caltech-101 Faces
-----------------
Faces from the Caltech-101 image data set (http://www.vision.caltech.edu/Image_Datasets/Caltech101/).

Images are the center 64x64 pixels of the original face images. They have been preprocessed
to a vector format (1 face per row) and pixel intensities have been normalized from
0-255 to 0-1.

Please cite this data set as

	@article{feifei:cviu07,
		author = "Fei-Fei, Li and Fergus, Rob and Perona, Pietro",
		journal = "Comput. Vis. Image Underst.",
		number = "1",
		pages = "59--70",
		title = "Learning generative visual models from few training examples: An incremental {B}ayesian approach tested on 101 object categories",
		volume = "106",
		year = "2007"
	}
