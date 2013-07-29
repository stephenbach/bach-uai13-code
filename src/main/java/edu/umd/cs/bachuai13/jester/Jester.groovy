package edu.umd.cs.bachuai13.jester

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.bachuai13.jester.AdjCosineSimilarity;
import edu.umd.cs.bachuai13.jester.ProjectionAverage;
import edu.umd.cs.bachuai13.util.ExperimentConfigGenerator;
import edu.umd.cs.bachuai13.util.WeightLearner;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*** CONFIGURATION PARAMETERS ***/

def dataPath = "./data/jester/"

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("jester");

def defPath = System.getProperty("java.io.tmpdir") + "/jester"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)
folds = 10

def sq = cb.getBoolean("squared", true);
def simThresh = cb.getDouble("simThresh", 0.5);

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("jester");

/*
 * SET MODEL TYPES
 *
 * Options:
 * "quad" HLEF
 * "bool" MLN
 */
configGenerator.setModelTypes(["quad"]);

/*
 * SET LEARNING ALGORITHMS
 *
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
methods = ["MLE","MPLE","MM"];
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);

/* MM options */
configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
configGenerator.setMaxMarginSquaredSlackValues([false]);

/*** MODEL DEFINITION ***/

log.info("Initializing model ...");

PSLModel m = new PSLModel(this, data);

/* PREDICATES */

m.add predicate: "user", types: [ArgumentType.UniqueID];
m.add predicate: "joke", types: [ArgumentType.UniqueID];
m.add predicate: "rating", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "ratingObs", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "ratingPrior", types: [ArgumentType.UniqueID];
m.add predicate: "avgUserRatingObs", types: [ArgumentType.UniqueID];
m.add predicate: "avgJokeRatingObs", types: [ArgumentType.UniqueID];
m.add predicate: "simObsRating", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "simJokeText", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];

/* RULES */
// If J1,J2 have similar observed ratings, then U will rate them similarly
m.add rule: ( simObsRating(J1,J2) & rating(U,J1) ) >> rating(U,J2), weight: 1.0, squared: sq;

// Ratings should concentrate around observed user/joke averages
m.add rule: ( user(U) & joke(J) & avgUserRatingObs(U) ) >> rating(U,J), weight: 1.0, squared: sq;
m.add rule: ( user(U) & joke(J) & avgJokeRatingObs(J) ) >> rating(U,J), weight: 1.0, squared: sq;
m.add rule: ( user(U) & joke(J) & rating(U,J) ) >> avgUserRatingObs(U), weight: 1.0, squared: sq;
m.add rule: ( user(U) & joke(J) & rating(U,J) ) >> avgJokeRatingObs(J), weight: 1.0, squared: sq;

// Two-sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: ( user(U) & joke(J) & ratingPrior(constant) ) >> rating(U,J), weight: 1.0, squared: sq;
m.add rule: ( rating(U,J) ) >> ratingPrior(constant), weight: 1.0, squared: sq;

log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());


/*** LOAD DATA ***/

log.info("Loading data ...");


Map<ConfigBundle,ArrayList<Double>> expResults = new HashMap<String,ArrayList<Double>>();
for (ConfigBundle config : configs) {
	expResults.put(config, new ArrayList<Double>(folds));
}

for (int fold = 0; fold < folds; fold++) {

	Partition read_tr = new Partition(0 + fold * folds);
	Partition write_tr = new Partition(1 + fold * folds);
	Partition read_te = new Partition(2 + fold * folds);
	Partition write_te = new Partition(3 + fold * folds);
	Partition labels_tr = new Partition(4 + fold * folds);
	Partition labels_te = new Partition(5 + fold * folds);

	def inserter;

	// users
	inserter = data.getInserter(user, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-tr-1000.txt");
	inserter = data.getInserter(user, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-te-1000.txt");
	// jokes
	inserter = data.getInserter(joke, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/jokes.txt");
	inserter = data.getInserter(joke, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/jokes.txt");
	// joke text similarity
	inserter = data.getInserter(simJokeText, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/joketext/jokeTextSim.txt");
	inserter = data.getInserter(simJokeText, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/joketext/jokeTextSim.txt");
	// observed ratings
	inserter = data.getInserter(rating, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-tr-obs-" + fold + ".txt");
	inserter = data.getInserter(ratingObs, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-tr-obs-" + fold + ".txt");
	inserter = data.getInserter(rating, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-te-obs-" + fold + ".txt");
	inserter = data.getInserter(ratingObs, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-te-obs-" + fold + ".txt");
	// unobserved ratings (ground truth)
	inserter = data.getInserter(rating, labels_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-tr-uno-" + fold + ".txt");
	inserter = data.getInserter(rating, labels_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/jester-1-te-uno-" + fold + ".txt");
	// prior (we'll overwrite later)
	data.getInserter(ratingPrior, read_tr).insertValue(0.5, constant)
	data.getInserter(ratingPrior, read_te).insertValue(0.5, constant)
	

	/** POPULATE DB ***/

	/* We want to populate the database with all groundings 'rating' and 'ratingObs'
	 * To do so, we will query for all users and jokes in train/test, then use the
	 * database populator to compute the cross-product. 
	 */
	DatabasePopulator dbPop;
	Variable User = new Variable("User");
	Variable Joke = new Variable("Joke");
	Set<GroundTerm> users = new HashSet<GroundTerm>();
	Set<GroundTerm> jokes = new HashSet<GroundTerm>();
	Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(User, users);
	subs.put(Joke, jokes);
	ResultList results;
	def toClose;
	ProjectionAverage userAverager = new ProjectionAverage(ratingObs, 1);
	ProjectionAverage jokeAverager = new ProjectionAverage(ratingObs, 0);
	AdjCosineSimilarity userCosSim = new AdjCosineSimilarity(ratingObs, 1, avgJokeRatingObs, simThresh);
	AdjCosineSimilarity jokeCosSim = new AdjCosineSimilarity(ratingObs, 0, avgUserRatingObs, simThresh);

	/* First we populate training database.
	 * In the process, we will precompute averages ratings. 
	 */
	log.info("Computing averages ...")
	Database trainDB = data.getDatabase(read_tr);
	results = trainDB.executeQuery(Queries.getQueryForAllAtoms(user));
	for (int i = 0; i < results.size(); i++) {
		GroundTerm u = results.get(i)[0];
		users.add(u);
		double avg = userAverager.getValue(trainDB, u);
		RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(avgUserRatingObs, u);
		a.setValue(avg);
		trainDB.commit(a);
	}
	results = trainDB.executeQuery(Queries.getQueryForAllAtoms(joke));
	for (int i = 0; i < results.size(); i++) {
		GroundTerm j = results.get(i)[0];
		jokes.add(j);
		double avg = jokeAverager.getValue(trainDB, j);
		RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(avgJokeRatingObs, j);
		a.setValue(avg);
		trainDB.commit(a);
	}
	/* Compute the prior as average over all observed ratings.
	 * (This is not the most efficient way of doing this. We should be able to 
	 * compute the average overall rating when we compute user/item averages.)
	 */
	double avgAllRatingObs = 0.0;
	Set<GroundAtom> allRatingObs = Queries.getAllAtoms(trainDB, ratingObs);
	for (GroundAtom a : allRatingObs) {
		avgAllRatingObs += a.getValue();
	}
	avgAllRatingObs /= allRatingObs.size();
	log.info("  Average rating (train): {}", avgAllRatingObs);
	RandomVariableAtom priorAtom = (RandomVariableAtom) trainDB.getAtom(ratingPrior, constant);
	priorAtom.setValue(avgAllRatingObs);
	
	/* Precompute the similarities. */
	log.info("Computing training similarities ...")
	int nnzSim = 0;
	double avgsim = 0.0;
	List<GroundTerm> jokeList = new ArrayList(jokes);
	for (int i = 0; i < jokeList.size(); i++) {
		GroundTerm j1 = jokeList.get(i);
		for (int j = i+1; j < jokeList.size(); j++) {
			GroundTerm j2 = jokeList.get(j);
			double s = jokeCosSim.getValue(trainDB, j1, j2);
			if (s > 0.0) {
				/* upper half */
				RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(simObsRating, j1, j2);
				a.setValue(s);
				trainDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) trainDB.getAtom(simObsRating, j2, j1);
				a.setValue(s);
				trainDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	log.info("  Number nonzero sim (train): {}", nnzSim);
	log.info("  Average joke rating sim (train): {}", avgsim / nnzSim);
	trainDB.close();

	log.info("Populating training database ...");
	toClose = [user,joke,ratingObs,ratingPrior,simJokeText,avgUserRatingObs,avgJokeRatingObs,simObsRating] as Set;
	trainDB = data.getDatabase(write_tr, toClose, read_tr);
	dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(rating, User, Joke), subs);
	Database labelsDB = data.getDatabase(labels_tr, [rating] as Set)

	/* Clear the users, jokes so we can reuse */
	users.clear();
	jokes.clear();

	/* Get the test set users/jokes
	 * and precompute averages
	 */
	log.info("Computing averages ...")
	Database testDB = data.getDatabase(read_te)
	results = testDB.executeQuery(Queries.getQueryForAllAtoms(user));
	for (int i = 0; i < results.size(); i++) {
		GroundTerm u = results.get(i)[0];
		users.add(u);
		double avg = userAverager.getValue(testDB, u);
		RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(avgUserRatingObs, u);
		a.setValue(avg);
		testDB.commit(a);
	}
	results = testDB.executeQuery(Queries.getQueryForAllAtoms(joke));
	for (int i = 0; i < results.size(); i++) {
		GroundTerm j = results.get(i)[0];
		jokes.add(j);
		double avg = jokeAverager.getValue(testDB, j);
		RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(avgJokeRatingObs, j);
		a.setValue(avg);
		testDB.commit(a);
	}

	/* Compute the prior as average over all observed ratings. */
	avgAllRatingObs = 0.0;
	allRatingObs = Queries.getAllAtoms(testDB, ratingObs);
	for (GroundAtom a : allRatingObs) {
		avgAllRatingObs += a.getValue();
	}
	avgAllRatingObs /= allRatingObs.size();
	log.info("  Average rating (test): {}", avgAllRatingObs);
	priorAtom = (RandomVariableAtom) testDB.getAtom(ratingPrior, constant);
	priorAtom.setValue(avgAllRatingObs);

	/* Precompute the similarities. */
	log.info("Computing testing similarities ...")
	nnzSim = 0;
	avgsim = 0.0;
	for (int i = 0; i < jokeList.size(); i++) {
		GroundTerm j1 = jokeList.get(i);
		for (int j = i+1; j < jokeList.size(); j++) {
			GroundTerm j2 = jokeList.get(j);
			double s = jokeCosSim.getValue(testDB, j1, j2);
			if (s > 0.0) {
				/* upper half */
				RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(simObsRating, j1, j2);
				a.setValue(s);
				testDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) testDB.getAtom(simObsRating, j2, j1);
				a.setValue(s);
				testDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	log.info("  Number nonzero sim (train): {}", nnzSim);
	log.info("  Average joke rating sim (train): {}", avgsim / nnzSim);
	testDB.close();

	/* Populate testing database. */
	log.info("Populating testing database ...");
	toClose = [user,joke,ratingObs,ratingPrior,simJokeText,avgUserRatingObs,avgJokeRatingObs,simObsRating] as Set;
	testDB = data.getDatabase(write_te, toClose, read_te);
	dbPop = new DatabasePopulator(testDB);
	dbPop.populate(new QueryAtom(rating, User, Joke), subs);
	testDB.close();

	/*** EXPERIMENT ***/
	log.info("Starting experiment ...");
	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		def configName = config.getString("name", "");
		def method = config.getString("learningmethod", "");

		/* Weight learning */
		WeightLearner.learn(method, m, trainDB, labelsDB, initWeights, config, log)

		log.info("Learned model {}: \n {}", configName, m.toString())

		/* Inference on test set */
		Database predDB = data.getDatabase(write_te, toClose, read_te);
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(predDB, rating)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
			/* For discrete MRFs, "MPE" inference will actually perform marginal inference */
		MPEInference mpe = new MPEInference(m, predDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.info("Objective: {}", result.getTotalWeightedIncompatibility())
		predDB.close();
	
		/* Evaluation */
		predDB = data.getDatabase(write_te);
		Database groundTruthDB = data.getDatabase(labels_te, [rating] as Set)
		def comparator = new ContinuousPredictionComparator(predDB)
		comparator.setBaseline(groundTruthDB)
		def metrics = [ContinuousPredictionComparator.Metric.MSE, ContinuousPredictionComparator.Metric.MAE]
		double [] score = new double[metrics.size()]
		for (int i = 0; i < metrics.size(); i++) {
			comparator.setMetric(metrics.get(i))
			score[i] = comparator.compare(rating)
		}
		log.warn("Fold {} : {} : MSE {} : MAE {}", fold, configName, score[0], score[1]);
		expResults.get(config).add(fold, score);
		predDB.close();
		groundTruthDB.close()
	}
	trainDB.close()
}

log.warn("\n\nRESULTS\n");
for (ConfigBundle config : configs) {
	def configName = config.getString("name", "")
	def scores = expResults.get(config);
	for (int fold = 0; fold < folds; fold++) {
		def score = scores.get(fold)
		log.warn("{} \t{}\t{}\t{}", configName, fold, score[0], score[1]);
	}
}
