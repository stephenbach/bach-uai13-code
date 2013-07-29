package edu.umd.cs.bachuai13

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.ExperimentConfigGenerator;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.WeightLearner;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*** CONFIGURATION PARAMETERS ***/

def dataSet = args[0];
def modelType = args[1];

if (dataSet.equals("citeseer")) {
	dataPath = "./data/citeseer/"
	numCategories = 6
	wordFile = "citeseer.words"
	labelFile = "citeseer.labels"
	linkFile = "citeseer.links"
}
else if (dataSet.equals("cora")) {
	dataPath = "./data/cora/"
	numCategories = 7
	wordFile = "cora.words"
	labelFile = "cora.labels"
	linkFile = "cora.links"
}
else
	throw new IllegalArgumentException("Unrecognized data set: "
		+ dataSet + ". Options are 'citeseer' and 'cora.'");

sq = (!modelType.equals("linear") ? true : false)
usePerCatRules = true
folds = 20 // number of folds
double seedRatio = 0.5 // ratio of observed labels
Random rand = new Random(0) // used to seed observed data
trainTestRatio = 0.5 // ratio of train to test splits (random)
filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
targetSize = 3000 // target size of snowball sampler
explore = 0.001 // prob of random node in snowball sampler

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = cb.getString("dbpath", defaultPath + File.separator + "psl-" + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

/**
 * SET UP CONFIGS
 */

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator(dataSet);

/*
 * SET MODEL TYPES
 *
 * Options:
 * "quad" HL-MRF-Q
 * "linear" HL-MRF-L
 * "bool" MRF
 */
configGenerator.setModelTypes([modelType]);

/*
 * SET LEARNING ALGORITHMS
 *
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
configGenerator.setLearningMethods(["MLE", "MPLE", "MM"]);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 5.0]);

/* MM options */
configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
configGenerator.setMaxMarginSquaredSlackValues([false]);

List<ConfigBundle> configs = configGenerator.getConfigs();


/*
 * DEFINE MODEL
 */

PSLModel m = new PSLModel(this, data)

// rules
m.add predicate: "HasCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Cat", types: [ArgumentType.UniqueID]

// prior
m.add rule : ~(HasCat(A,N)), weight: 0.001, squared: sq

// per-cat rules
if (usePerCatRules) {
	for (int i = 0; i < numCategories; i++)  {
		UniqueID cat = data.getUniqueID(i+1)
		m.add rule : ( HasCat(A, cat) & Link(A,B)) >> HasCat(B, cat), weight: 1.0, squared: sq
		m.add rule : ( HasCat(A, cat) & Link(B,A)) >> HasCat(B, cat), weight: 1.0, squared: sq
	}
}
else {
	// neighbor has cat => has cat
	m.add rule : ( HasCat(A,C) & Link(A,B) & (A - B)) >> HasCat(B,C), weight: 1.0, squared: sq
	m.add rule : ( HasCat(A,C) & Link(B,A) & (A - B)) >> HasCat(B,C), weight: 1.0, squared: sq
}

// ensure that HasCat sums to 1
m.add PredicateConstraint.Functional , on : HasCat

/* get all default weights */
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());


/*** LOAD DATA ***/
Partition fullObserved =  new Partition(0)
Partition groundTruth = new Partition(1)

def inserter
inserter = data.getInserter(Link, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + linkFile)
inserter = data.getInserter(HasCat, groundTruth)
InserterUtils.loadDelimitedData(inserter, dataPath + labelFile)

trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()
testLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
ArrayList<Set<Integer>> trainingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingKeys = new ArrayList<Set<Integer>>()
def queries = new HashSet<DatabaseQuery>()


/*
 * DEFINE PRIMARY KEY QUERIES FOR FOLD SPLITTING
 */
Variable document = new Variable("Document")
Variable linkedDocument = new Variable("LinkedDoc")
keys.add(document)
keys.add(linkedDocument)
queries.add(new DatabaseQuery(Link(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(HasCat(document, A).getFormula()))

def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

for (int i = 0; i < folds; i++) {
	trainReadPartitions.add(i, new Partition(i + 2))
	testReadPartitions.add(i, new Partition(i + folds + 2))

	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))

	trainLabelPartitions.add(i, new Partition(i + 4*folds + 2))
	testLabelPartitions.add(i, new Partition(i + 5*folds + 2))

	Set<GroundTerm> [] documents = FoldUtils.generateRandomSplit(data, trainTestRatio,
			fullObserved, groundTruth, trainReadPartitions.get(i),
			testReadPartitions.get(i), trainLabelPartitions.get(i),
			testLabelPartitions.get(i), queries, keys, filterRatio)
	//	Set<GroundTerm> [] documents = FoldUtils.generateSnowballSplit(data, fullObserved, groundTruth,
	//		trainReadPartitions.get(i), testReadPartitions.get(i), trainLabelPartitions.get(i),
	//		testLabelPartitions.get(i), queries, keys, targetSize, Link, explore)

	partitionDocuments.put(trainReadPartitions.get(i), documents[0])
	partitionDocuments.put(testReadPartitions.get(i), documents[1])

	trainingSeedKeys.add(i, new HashSet<Integer>())
	testingSeedKeys.add(i, new HashSet<Integer>())
	trainingKeys.add(i, new HashSet<Integer>())
	testingKeys.add(i, new HashSet<Integer>())

	for (GroundTerm doc : partitionDocuments.get(trainReadPartitions.get(i))) {
		if (rand.nextDouble() < seedRatio)
			trainingSeedKeys.get(i).add(Integer.decode(doc.toString()))
		trainingKeys.get(i).add(Integer.decode(doc.toString()))
	}
	for (GroundTerm doc : partitionDocuments.get(testReadPartitions.get(i))) {
		if (rand.nextDouble() < seedRatio)
			testingSeedKeys.get(i).add(Integer.decode(doc.toString()))
		testingKeys.get(i).add(Integer.decode(doc.toString()))
	}

	// add all seedKeys into observed partition
	Database db = data.getDatabase(groundTruth)
	def trainInserter = data.getInserter(HasCat, trainReadPartitions.get(i))
	def testInserter = data.getInserter(HasCat, testReadPartitions.get(i))
	ResultList res = db.executeQuery(new DatabaseQuery(HasCat(X,Y).getFormula()))
	for (GroundAtom atom : Queries.getAllAtoms(db, HasCat)) {
		Integer atomKey = Integer.decode(atom.getArguments()[0].toString())
		if (trainingSeedKeys.get(i).contains(atomKey)) {
			trainInserter.insertValue(atom.getValue(), atom.getArguments())
		}

		if (testingSeedKeys.get(i).contains(atomKey)) {
			testInserter.insertValue(atom.getValue(), atom.getArguments())
		}
	}
	db.close()
	
	db = data.getDatabase(trainReadPartitions.get(i))
	ResultList list = db.executeQuery(new DatabaseQuery(HasCat(X,Y).getFormula()))
	log.debug("{} instances of HasCat in {}", list.size(), trainReadPartitions.get(i))
	db.close()
	db = data.getDatabase(testReadPartitions.get(i))
	list = db.executeQuery(new DatabaseQuery(HasCat(X,Y).getFormula()))
	log.debug("{} instances of HasCat in {}", list.size(), testReadPartitions.get(i))
	db.close()
}

Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
for (ConfigBundle method : configs)
	results.put(method, new ArrayList<DiscretePredictionStatistics>())

for (int fold = 0; fold < folds; fold++) {

	/*** POPULATE DBs ***/
	
	Database db;
	DatabasePopulator dbPop;
	Variable Category = new Variable("Category")
	Variable Document = new Variable("Document")
	Map<Variable, Set<GroundTerm>> substitutions = new HashMap<Variable, Set<GroundTerm>>()
	
	/* categories */
	Set<GroundTerm> categoryGroundings = new HashSet<GroundTerm>()
	for (int i = 0; i <= numCategories; i++)
		categoryGroundings.add(data.getUniqueID(i+1))
	substitutions.put(Category, categoryGroundings)

	/* populate HasCat */

	toClose = [Link, Cat] as Set;
	Database trainDB = data.getDatabase(trainWritePartitions.get(fold), toClose, trainReadPartitions.get(fold))
	Database testDB = data.getDatabase(testWritePartitions.get(fold), toClose, testReadPartitions.get(fold))

	dbPop = new DatabasePopulator(trainDB)
	substitutions.put(Document, partitionDocuments.get(trainReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(HasCat, Document, Category), substitutions)

	dbPop = new DatabasePopulator(testDB)
	substitutions.put(Document, partitionDocuments.get(testReadPartitions.get(fold)))
	dbPop.populate(new QueryAtom(HasCat, Document, Category), substitutions)

	toClose = [HasCat] as Set
	Database labelsDB = data.getDatabase(trainLabelPartitions.get(fold), toClose)

	def groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [HasCat] as Set)
	DataOutputter.outputPredicate("output/" + dataSet + "/groundTruth" + fold + ".node" , groundTruthDB, HasCat, ",", false, "nodeid,label")
	groundTruthDB.close()

	DataOutputter.outputPredicate("output/" + dataSet + "/groundTruth" + fold + ".directed" , testDB, Link, ",", false, null)
	
	/*** EXPERIMENT ***/
	
	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		learn(m, trainDB, labelsDB, config, log)

		System.out.println("Learned model " + config.getString("name", "") + "\n" + m.toString())

		/* Inference on test set */
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(testDB, HasCat)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
		/* For discrete MRFs, "MPE" inference will actually perform marginal inference */
		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

		/* Evaluation */
		def comparator = new DiscretePredictionComparator(testDB)
		groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [HasCat] as Set)
		comparator.setBaseline(groundTruthDB)
		comparator.setResultFilter(new MaxValueFilter(HasCat, 1))
		comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

		int totalTestExamples = testingKeys.get(fold).size() * numCategories;
		System.out.println("totalTestExamples " + totalTestExamples)
		DiscretePredictionStatistics stats = comparator.compare(HasCat, totalTestExamples)
		System.out.println("F1 score " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))

		results.get(config).add(fold, stats)

		DataOutputter.outputClassificationPredictions("output/" + dataSet + "/" + config.getString("name", "") + fold + ".csv", testDB, HasCat, ",")

		groundTruthDB.close()
	}
	trainDB.close()
}

for (ConfigBundle config : configs) {
	def methodStats = results.get(config)
	for (int fold = 0; fold < folds; fold++) {
		def stats = methodStats.get(fold)
		def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		System.out.println("Method " + config.getString("name", "") + ", fold " + fold +", acc " + stats.getAccuracy() +
				", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
				", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
				", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
	}
}



public void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("learningmethod", "")) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}



