package edu.umd.cs.bachuai13.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang.builder.HashCodeBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabaseQuery;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.Atom;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.ObservedAtom;
import edu.umd.cs.psl.model.atom.QueryAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.util.database.Queries;

class FoldUtils {


	private static Logger log = LoggerFactory.getLogger("FoldUtils");
	private static Random rand = new Random(0);
	

	/**
	 * creates two splits of data by randomly sampling from partition fullData
	 * and places one split in partition train and one in partition test.
	 * @param data
	 * @param trainTestRatio
	 * @param fullData
	 * @param train
	 * @param test
	 * @param filterRatio
	 */
	public static List<Set<GroundTerm>> generateSnowballSplit(DataStore data, Partition observedData,
			Partition groundTruth, Partition train, Partition test, Partition trainLabels, Partition testLabels,
			Set<DatabaseQuery> queries,	Set<Variable> keys, int targetSize, Predicate edge, double explore) {
		log.info("Splitting data from " + observedData + " into clusters of target size " + targetSize +
				" into new partitions " + train +" and " + test);

		Database db = data.getDatabase(observedData);
		Set<GroundAtom> edges = Queries.getAllAtoms(db, edge);
		Set<GroundTerm> nodeSet = new HashSet<GroundTerm>();
		for (GroundAtom atom : edges) {
			nodeSet.add(atom.getArguments()[0]);
			nodeSet.add(atom.getArguments()[1]);
		}
		List<GroundTerm> nodes = new ArrayList<GroundTerm>(nodeSet.size());
		nodes.addAll(nodeSet);

		List<GroundTerm> trainList = new ArrayList<GroundTerm>();
		List<GroundTerm> testList = new ArrayList<GroundTerm>();

		// start sampling
		GroundTerm nextTrain = nodes.get(rand.nextInt(nodes.size()));
		nodes.remove(nextTrain);
		GroundTerm nextTest = nodes.get(rand.nextInt(nodes.size()));
		nodes.remove(nextTest);
		trainList.add(nextTrain);
		testList.add(nextTest);
		log.info("Started snowball sampling with train seed {}, test {}", nextTrain, nextTest);

		List<GroundTerm> frontierTrain = new ArrayList<GroundTerm>();
		List<GroundTerm> frontierTest = new ArrayList<GroundTerm>();
		boolean check;
		while (nodes.size() > 0 && (trainList.size() < targetSize || testList.size() < targetSize)) {
			// sample training point
			nextTrain = (rand.nextDouble() < explore) ? nodes.get(rand.nextInt(nodes.size())) :
					sampleNextNeighbor(db, edge, nextTrain, nodes, frontierTrain, rand);
			if (nextTrain == null) {
				nextTrain = nodes.get(rand.nextInt(nodes.size()));
			}
			check = nodes.remove(nextTrain);
			if (!check) {
				log.debug("Something went wrong. Attempted to add a train node {} that should have already been removed", nextTest);
			}
			trainList.add(nextTrain);

			if (!nodes.isEmpty()) {
				// sample testing point
				nextTest = (rand.nextDouble() < explore) ? nodes.get(rand.nextInt(nodes.size())) :
						sampleNextNeighbor(db, edge, nextTest, nodes, frontierTest, rand);
				if (nextTest == null) {
					nextTest = nodes.get(rand.nextInt(nodes.size()));
				}
				check = nodes.remove(nextTest);
				if (!check)
					log.debug("Something went wrong. Attempted to add a test node {} that should have already been removed", nextTest);

				testList.add(nextTest);
			}
			//			log.debug("added {} to train, added {} to test", nextTrain, nextTest)
		}
		db.close();

		Map<GroundTerm, Partition> keyMap = new HashMap<GroundTerm, Partition>(trainList.size() + testList.size());
		for (GroundTerm term : trainList) keyMap.put(term, train);
		for (GroundTerm term : testList) keyMap.put(term, test);
		
		// test for consistent splits
		HashCodeBuilder hcb = new HashCodeBuilder(1,5);
		for (GroundTerm term : trainList) hcb.append(term);
		for (GroundTerm term : testList) hcb.append(term);

		log.info("Hashcode of snowball split: {}", hcb.toHashCode());
		
		return processSplits(data, observedData, groundTruth, train, test, trainLabels, testLabels, queries, keys, keyMap);
	}

	private static GroundTerm sampleNextNeighbor(Database db, Predicate edge,
			GroundTerm node, List<GroundTerm> nodes, List<GroundTerm> frontier, Random rand) {

		Variable neighbor = new Variable("Neighbor");
		QueryAtom q = new QueryAtom(edge, Queries.convertArguments(db, edge, node, neighbor));

		ResultList results = db.executeQuery(new DatabaseQuery(q));

		for (int i = 0; i < results.size(); i++)
			frontier.add(db.getAtom(edge, node, results.get(i)[0]).getArguments()[1]);

		frontier.retainAll(nodes);
		frontier.remove(node);

		if (frontier.isEmpty())
			return null;
		int index = rand.nextInt(frontier.size());
		Iterator<GroundTerm> iter;
		for (iter = frontier.iterator(); index > 0; iter.next())
			index--;
		GroundTerm next = iter.next();
		frontier.remove(next);

		return next;
	}

	/**
	 * creates two splits of data by randomly sampling from partition fullData 
	 * and places one split in partition train and one in partition test.
	 * @param data
	 * @param trainTestRatio
	 * @param fullData
	 * @param train
	 * @param test
	 * @param filterRatio
	 */ 
	public static List<Set<GroundTerm>> generateRandomSplit(DataStore data, double trainTestRatio,
			Partition observedData, Partition groundTruth, Partition train,
			Partition test, Partition trainLabels, Partition testLabels, Set<DatabaseQuery> queries,
			Set<Variable> keys, double filterRatio) {
		log.info("Splitting data from " + observedData + " with ratio " + trainTestRatio +
				" into new partitions " + train +" and " + test);

		Database db = data.getDatabase(observedData, groundTruth);
		Set<GroundTerm> keySet = new HashSet<GroundTerm>();
		for (DatabaseQuery q : queries) {
			ResultList groundings = db.executeQuery(q);

			for (Variable key : keys) {
				int keyIndex = q.getVariableIndex(key);
				if (keyIndex == -1)
					continue;
				for (int i = 0; i < groundings.size(); i++) {
					GroundTerm [] grounding = groundings.get(i);
					keySet.add(grounding[keyIndex]);
				}
			}
		}

		List<GroundTerm> keyList = new ArrayList<GroundTerm>();
		keyList.addAll(keySet);
		Collections.sort(keyList);

		Collections.shuffle(keyList, rand);

		int split = (int) (keyList.size() * trainTestRatio);

		HashCodeBuilder hcb = new HashCodeBuilder(1,5);
		Map<GroundTerm, Partition> keyMap = new HashMap<GroundTerm, Partition>();
		for (int i = 0; i < keyList.size(); i++) {
			if (i < split)
				keyMap.put(keyList.get(i), train);
			else
				keyMap.put(keyList.get(i), test);
			hcb.append(keyList.get(i));
		}

		log.info("Found {} unique keys, split hashcode: {}", keyMap.size(), hcb.toHashCode());
		db.close();

		return processSplits(data, observedData, groundTruth, train, test, trainLabels, testLabels, queries, keys, keyMap);
	}


	private static List<Set <GroundTerm>> processSplits(DataStore data, Partition observedData,
			Partition groundTruth, Partition train, Partition test, Partition trainLabels,
			Partition testLabels, Set<DatabaseQuery> queries, Set<Variable> keys, Map<GroundTerm, Partition> keyMap) {
		List<Set<GroundTerm>> splits = new ArrayList<Set<GroundTerm>>(2);
		splits.add(0, new HashSet<GroundTerm>());
		splits.add(1, new HashSet<GroundTerm>());
		for (Map.Entry<GroundTerm, Partition> e : keyMap.entrySet()) {
			int index = -1;
			if (e.getValue() == train) index = 0;
			if (e.getValue() == test) index = 1;
			if (index >= 0)
				splits.get(index).add(e.getKey());
		}

		Database db = data.getDatabase(observedData);
		log.info("Assigned " + splits.get(0).size() + " in train partition and " + splits.get(1).size() + " in test");

		Partition [] partitions = {train, test};
		
		for (Partition p : partitions) {
			log.debug("Putting data into partition " + p);
			for (DatabaseQuery q : queries) {
				// get predicate from query
				StandardPredicate predicate = (StandardPredicate) getPredicate(q);

				Inserter insert = data.getInserter(predicate, p);

				ResultList groundings = db.executeQuery(q);
				for (int i = 0; i < groundings.size(); i++) {
					GroundTerm [] grounding = groundings.get(i);
					// check if all keys in this ground term are in this split
					boolean add = true;
					for (Variable key : keys) {
						int keyIndex = q.getVariableIndex(key);
						if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != p)
							add = false;
					}
					if (add) {
						GroundAtom groundAtom = db.getAtom(predicate,  grounding);
						insert.insertValue(groundAtom.getValue(), (Object []) groundAtom.getArguments());
						log.trace("Inserted " + groundAtom + " into " + p);
					}
				}
			}
		}

		db.close();

		db = data.getDatabase(groundTruth);

		// move labels from groundTruth into trainLabels and testLabels
		log.debug("Moving ground truth into split training and testing label partitions");
		for (DatabaseQuery q : queries) {
			StandardPredicate predicate = (StandardPredicate) getPredicate(q);
			// insert into train label partition
			Inserter insert = data.getInserter(predicate, trainLabels);

			ResultList groundings = db.executeQuery(q);
			for (int i = 0; i < groundings.size(); i++) {
				GroundTerm [] grounding = groundings.get(i);
				// check if all keys in this ground term are in this split
				boolean add = true;
				for (Variable key : keys) {
					int keyIndex = q.getVariableIndex(key);
					if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != train)
						add = false;
				}
				if (add) {
					GroundAtom groundAtom = db.getAtom(predicate,  grounding);
					insert.insertValue(groundAtom.getValue(), (Object []) groundAtom.getArguments());
					log.trace("Inserted " + groundAtom + " into " + trainLabels);
				}
			}

			insert = data.getInserter(predicate, testLabels);

			groundings = db.executeQuery(q);
			for (int i = 0; i < groundings.size(); i++) {
				GroundTerm [] grounding = groundings.get(i);
				// check if all keys in this ground term are in this split
				boolean add = true;
				for (Variable key : keys) {
					int keyIndex = q.getVariableIndex(key);
					if (keyIndex != -1 && keyMap.get(grounding[keyIndex]) != test)
						add = false;
				}
				if (add) {
					GroundAtom groundAtom = db.getAtom(predicate,  grounding);
					insert.insertValue(groundAtom.getValue(), (Object []) groundAtom.getArguments());
					log.trace("Inserted " + groundAtom + " into " + testLabels);
				}
			}
		}

		db.close();
		return splits;
	}

	/**
	 * Generates a list of sets of GroundTerm []s from all groundings of provided predicates and partitions
	 * Randomly splits uniformly among n sets
	 * @param data 
	 * @param predicates Predicates to distribute
	 * @param partitions partitions to look in
	 * @param n number of splits to make
	 * @return length n list of sets of GroundTerm arrays
	 */
	public static List<Set<GroundingWrapper>> splitGroundings(DataStore data, List<Predicate> predicates,
			List<Partition> partitions, int n) {
		List<Set<GroundingWrapper>> groundings = new ArrayList<Set<GroundingWrapper>>(n);
		for (int i = 0; i < n; i++)
			groundings.add(i, new HashSet<GroundingWrapper>());

		Set<GroundingWrapper> allGroundings = new HashSet<GroundingWrapper>();
		for (Partition part : partitions) {
			Database db = data.getDatabase(part);
			for (Predicate pred : predicates) {
				Set<GroundAtom> list = Queries.getAllAtoms(db, pred);
				for (GroundAtom atom : list)
					allGroundings.add(new GroundingWrapper(atom.getArguments()));
			}
			db.close();
		}
		
		List<GroundingWrapper> allGroundingList = new ArrayList<GroundingWrapper>(allGroundings.size());
		allGroundingList.addAll(allGroundings);
		Collections.sort(allGroundingList); 

		HashCodeBuilder hcb = new HashCodeBuilder(3, 7);
		
		for (GroundingWrapper grounding : allGroundingList) {
			int i = rand.nextInt(n);
			groundings.get(i).add(grounding);
			
			hcb.append(grounding);
			hcb.append(i);
		}
		
		log.info("Split hashcode {}", hcb.toHashCode());
		
		return groundings;
	}
	
	
	/**
	 * Copies groundings of predicate from one partition to another
	 * @param data
	 * @param from
	 * @param to
	 * @param predicate
	 * @param groundings
	 */
	public static void copy(DataStore data, Partition from, Partition to, Predicate predicate, Set<GroundingWrapper> groundings) {
		Inserter insert = data.getInserter((StandardPredicate) predicate, to);

		Set<StandardPredicate> predicates = new HashSet<StandardPredicate>();
		predicates.add((StandardPredicate) predicate);
		Database db = data.getDatabase(from, predicates);

		for (GroundingWrapper grounding : groundings) {
			//log.debug("grounding length {}, first arg {}", grounding.length, grounding[0])
			GroundAtom atom = db.getAtom(predicate, grounding.getArray());

			if (atom instanceof ObservedAtom)
				insert.insertValue(atom.getValue(), (Object []) grounding.getArray());
			else
				log.debug("Encountered non-ObservedAtom, " + atom);
		}
		db.close();
	}

	private static Predicate getPredicate(DatabaseQuery q) {
		Set<Atom> atoms = new HashSet<Atom>();
		q.getFormula().getAtoms(atoms);
		if (atoms.size() > 1)
			throw new IllegalArgumentException("Fold splitting query must be a single atom");
		Atom atom = atoms.iterator().next();
		Predicate predicate = atom.getPredicate();
		return predicate;
	}
}