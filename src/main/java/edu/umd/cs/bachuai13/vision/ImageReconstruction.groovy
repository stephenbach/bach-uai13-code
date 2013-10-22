package edu.umd.cs.bachuai13.vision

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables

import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.vision.PatchStructure.Patch
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.MetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.GroundMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.IncompatibilityMetropolisRandOM
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight


Logger log = LoggerFactory.getLogger(this.class)

/* VISION EXPERIMENT SETTINGS */
// test on left half of face (bottom if false)
testLeft = true
// train on randomly sampled pixels
trainOnRandom = true
// number of training faces
numTraining = 100
// number of testing faces
numTesting = 50

def dataSet = args[0];

if (!dataSet.equals("olivetti") && !dataSet.equals("caltech")) {
	throw new IllegalArgumentException("Unrecognized data set. Options are 'olivetti' and 'caltech.'");
}

if (args.length >= 3) {
	if (args[1] == "bottom") {
		testLeft = false
		log.info("Testing on bottom of face")
	} else {
		testLeft = true
		log.info("Testing on left of face")
	}
	if (args[2] == "half") {
		trainOnRandom = false
		log.info("Training on given half of face")
	} else {
		trainOnRandom = true
		log.info("Training on randomly held-out pixels")
	}
}

def expSetup = (testLeft? "left" : "bottom") + "-" + (trainOnRandom? "rand" : "same")

/*
 * SET LEARNING ALGORITHMS
 * 
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
methods = ["MLE"];

/* MLE/MPLE options */
vpStepCounts = [100]
vpStepSizes = [5]


ConfigManager cm = ConfigManager.getManager()
ConfigBundle baseConfig = cm.getBundle("vision")

boolean sq = true

/*
 * DEFINES EXPERIMENT CONFIGURATIONS
 */
List<String> methodNames = new ArrayList<String>();
List<ConfigBundle> methodConfigs = new ArrayList<ConfigBundle>();
for (String method : methods) {
	if (method.equals("MLE") || method.equals("MPLE")) {
		for (int vpStepCount : vpStepCounts) {
			for (double vpStepSize : vpStepSizes) {
				ConfigBundle newBundle = cm.getBundle("vision");
				newBundle.addProperty("method", method);
				newBundle.addProperty(VotedPerceptron.NUM_STEPS_KEY, vpStepCount);
				newBundle.addProperty(VotedPerceptron.STEP_SIZE_KEY, vpStepSize);
				methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + vpStepCount + "-" + vpStepSize;
				methodNames.add(methodName);
				methodConfigs.add(newBundle);
			}
		}
	}
	else {
		ConfigBundle newBundle = cm.getBundle("vision");
		newBundle.addProperty("method", method);
		methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase();
		methodNames.add(methodName);
		methodConfigs.add(newBundle);
	}
}

/*
 * INITIALIZES DATASTORE AND MODEL
 */
def defaultPath = System.getProperty("java.io.tmpdir") + "/"
String dbpath = baseConfig.getString("dbpath", defaultPath + "psl-" + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), baseConfig)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */

width = 64
height = 64
branching = 2
depth = 7
def hierarchy = new PatchStructure(width, height, branching, depth, baseConfig)
hierarchy.generatePixels()


m.add predicate: "north", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "east", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "horizontalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "verticalMirror", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "neighbors", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "pixelBrightness", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "picture", types: [ArgumentType.UniqueID]

double initialWeight = 1.0

for (Patch p : hierarchy.getPatches().values()) {
	UniqueID patch = data.getUniqueID(p.uniqueID())
	Variable pic = new Variable("pictureVar")
	Term [] args = new Term[2]
	args[0] = patch
	args[1] = pic

	/** NEIGHBOR AGREEMENT **/
	// north neighbor
	if (p.hasNorth()) {
		m.add rule: (picture(pic) & north(patch,N) & pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(patch,N) & pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(patch,N) & ~pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(patch,N) & ~pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
	}

	// south neighbor
	if (p.hasSouth()) {
		m.add rule: (picture(pic) & north(N, patch) & pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(N, patch) & pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(N, patch) & ~pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & north(N, patch) & ~pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
	}

	// east neighbor
	if (p.hasEast()) {
		m.add rule: (picture(pic) & east(patch,N) & pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(patch,N) & pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(patch,N) & ~pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(patch,N) & ~pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
	}

	// west neighbor
	if (p.hasWest()) {
		m.add rule: (picture(pic) & east(N, patch) & pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(N, patch) & pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(N, patch) & ~pixelBrightness(N,pic)) >> pixelBrightness(patch,pic), weight: initialWeight, squared: sq
		m.add rule: (picture(pic) & east(N, patch) & ~pixelBrightness(N,pic)) >> ~pixelBrightness(patch,pic), weight: initialWeight, squared: sq
	}

	m.add rule: (horizontalMirror(patch,B) & pixelBrightness(patch,pic)) >> pixelBrightness(B,pic), weight: initialWeight, squared: sq
	m.add rule: (horizontalMirror(patch,B) & ~pixelBrightness(patch,pic) & picture(pic)) >> ~pixelBrightness(B,pic), weight: initialWeight, squared: sq
	m.add rule: (verticalMirror(patch,B) & pixelBrightness(patch,pic)) >> pixelBrightness(B,pic), weight: initialWeight, squared: sq
	m.add rule: (verticalMirror(patch,B) & ~pixelBrightness(patch,pic) & picture(pic)) >> ~pixelBrightness(B,pic), weight: initialWeight, squared: sq
}

log.info("Model has {} weighted kernels", m.getKernels().size());

Partition trainObs =  new Partition(0)
Partition testObs = new Partition(1)
Partition trainLabel = new Partition(2)
Partition testLabel = new Partition(3)
Partition trainWrite = new Partition(4)
Partition testWrite = new Partition(5)

/*
 * LOAD DATA
 */
dataDir = "data/vision"

// construct observed mask
boolean[] mask = new boolean[width * height]
boolean[] negMask = new boolean[width * height]
boolean[] trainMask = new boolean[width * height]
boolean[] negTrainMask = new boolean[width * height]
int c = 0
for (int x = 0; x < width; x++) {
	for (int y = 0; y < height; y++) {
		if (testLeft)
			mask[c] = x >= (width / 2)
		else
			mask[c] = y <= height / 2

		negMask[c] = !mask[c]

		trainMask[c] = mask[c]
		negTrainMask[c] = negMask[c]
		c++
	}
}

for (Partition part : [trainObs, testObs]) {
	def readDB = data.getDatabase(part)
	ImagePatchUtils.insertFromPatchMap(north, readDB, hierarchy.getNorth())
	ImagePatchUtils.insertFromPatchMap(east, readDB, hierarchy.getEast())
	ImagePatchUtils.insertFromPatchMap(horizontalMirror, readDB, hierarchy.getMirrorHorizontal())
	ImagePatchUtils.insertFromPatchMap(verticalMirror, readDB, hierarchy.getMirrorVertical())
	readDB.close()
}

ArrayList<double []> images = ImagePatchUtils.loadImages(dataDir + "/" + dataSet + "01.txt", width, height)
// create list of train images and test images
ArrayList<double []> trainImages = new ArrayList<double[]>()
ArrayList<double []> testImages = new ArrayList<double[]>()

for (int i = 0; i < images.size(); i++) {
	if (i < numTraining)
		trainImages.add(images.get(i))
	else if (i >= images.size() - numTesting)
		testImages.add(images.get(i))
}

images.clear()

Inserter picInserter = data.getInserter(picture, trainObs)
for (int i = 0; i < trainImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	picInserter.insert(id)
}
picInserter = data.getInserter(picture, testObs)
for (int i = 0; i < testImages.size(); i++) {
	UniqueID id = data.getUniqueID(i)
	picInserter.insert(id)
}

Random rand = new Random(0)

/** load images into base pixels **/
def trainReadDB = data.getDatabase(trainObs)
def trainLabelDB = data.getDatabase(trainLabel)
def trainWriteDB = data.getDatabase(trainWrite)
for (int i = 0; i < trainImages.size(); i++) {

	if (trainOnRandom) {
		c = 0
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				trainMask[c] = rand.nextBoolean()
				negTrainMask[c] = !trainMask[c]
				c++
			}
		}
	}

	UniqueID id = data.getUniqueID(i)
	ImagePatchUtils.setPixels(pixelBrightness, id, trainReadDB, hierarchy, width, height, trainImages.get(i), trainMask)
	ImagePatchUtils.setPixels(pixelBrightness, id, trainLabelDB, hierarchy, width, height, trainImages.get(i), negTrainMask)
}
trainWriteDB.close()
trainReadDB.close()
trainLabelDB.close()

def testReadDB = data.getDatabase(testObs)
def testLabelDB = data.getDatabase(testLabel)
def testWriteDB = data.getDatabase(testWrite)
for (int i = 0; i < testImages.size(); i++) {
	def id = data.getUniqueID(i)
	ImagePatchUtils.setPixels(pixelBrightness, id, testReadDB, hierarchy, width, height, testImages.get(i), mask)
	ImagePatchUtils.setPixels(pixelBrightness, id, testLabelDB, hierarchy, width, height, testImages.get(i), negMask)
}
testWriteDB.close()
testReadDB.close()
testLabelDB.close()

/** start experiments **/
def scores = new ArrayList<Double>()

for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++) {

	/** open databases **/
	def labelClose = [pixelBrightness] as Set
	def toClose = [north, east, horizontalMirror, verticalMirror, picture] as Set
	def trainDB = data.getDatabase(trainWrite, toClose, trainObs)
	def labelDB = data.getDatabase(trainLabel, labelClose)

	/*
	 * Weight learning
	 */
	/** populate pixelBrightness **/
	for (int i = 0; i < trainImages.size(); i++) {
		UniqueID id = data.getUniqueID(i)
		ImagePatchUtils.populatePixels(width, height, pixelBrightness, trainDB, id)
	}
	learn(m, trainDB, labelDB, methodConfigs.get(methodIndex), log)
	System.out.println("Learned model " + methodNames.get(methodIndex) + "\n" + m.toString())

	trainDB.close()
	labelDB.close()

	def testDB = data.getDatabase(testWrite, toClose, testObs)

	/*
	 * Inference on test set
	 */
	for (int i = 0; i < testImages.size(); i++) {
		UniqueID id = data.getUniqueID(i)
		ImagePatchUtils.populatePixels(width, height, pixelBrightness, testDB, id)
	}
	MPEInference mpe = new MPEInference(m, testDB, baseConfig)
	FullInferenceResult result = mpe.mpeInference()
	System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
	DataOutputter.outputPredicate("output/vision/"+ dataSet + "-" + expSetup + "-" + methodNames.get(methodIndex) + ".txt" , testDB, pixelBrightness, ",", true, "index,image")
	testDB.close()

	/*
	 * Evaluation
	 */
	def groundTruthDB = data.getDatabase(testLabel, labelClose)
	testDB = data.getDatabase(testWrite)
	def comparator = new ContinuousPredictionComparator(testDB)
	comparator.setBaseline(groundTruthDB)

	def metric = ContinuousPredictionComparator.Metric.MSE
	comparator.setMetric(metric)
	score = comparator.compare(pixelBrightness) * (255 * 255) // scale to full 256 grayscale range
	scores.add(methodIndex, score);

	methodName = methodNames.get(methodIndex)
	System.out.println("Method: " + methodName + ", mean squared error: " + scores.get(methodIndex))

	// close all databases
	groundTruthDB.close()
	testDB.close()
}

for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++) {
	methodName = methodNames.get(methodIndex)
	System.out.println(expSetup + ", method: " + methodName + ", mean squared error: " + scores.get(methodIndex))
}

private void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("method", "")) {
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
		case "None":
			break;
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}
