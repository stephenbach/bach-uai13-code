package edu.umd.cs.bachuai13.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.util.database.Queries;

public class DataOutputter {

	public static void outputPredicate(String filename, Database db, Predicate p, String delimiter, boolean printTruth, String header) {
		Set<GroundAtom> groundings = Queries.getAllAtoms(db, p);
		
		try {
			File file = new File(filename);
			if (file.getParentFile() != null)
				file.getParentFile().mkdirs();
			FileWriter fw = new FileWriter(file);
			BufferedWriter bw = new BufferedWriter(fw);

			if (header != null)
				bw.write(header + "\n");
			
			for (GroundAtom atom : groundings) {
				for (int i = 0; i < atom.getArity(); i++) {
					bw.write(atom.getArguments()[i].toString());
					if (i < atom.getArity() - 1)
						bw.write(delimiter);
				}
				if (printTruth)
					bw.write(delimiter + atom.getValue() + "\n");
				else
					bw.write("\n");
			}

			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void outputClassificationPredictions(String filename, Database db, Predicate p, String delimiter) {
		Set<GroundAtom> groundings = Queries.getAllAtoms(db, p);

		Map<GroundTerm, Map<GroundTerm, Double>> scores = new HashMap<GroundTerm, Map<GroundTerm, Double>>();

		Set<GroundTerm> labels = new HashSet<GroundTerm>();

		for (GroundAtom grounding : groundings) {
			// assume arity is 2
			GroundTerm node = grounding.getArguments()[0];
			GroundTerm label = grounding.getArguments()[1];
			if (!scores.containsKey(node))
				scores.put(node, new HashMap<GroundTerm, Double>());
			scores.get(node).put(label, grounding.getValue());
			labels.add(label);
		}

		List<GroundTerm> labelList = new ArrayList<GroundTerm>();
		labelList.addAll(labels);


		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter bw = new BufferedWriter(fw);

			bw.write("nodeid" + delimiter);

			for (int i = 0; i < labelList.size(); i++) {
				bw.write(""+labelList.get(i));
				if (i < labelList.size() - 1)
					bw.write(delimiter);
			}
			bw.write("\n");

			for (GroundTerm node : scores.keySet()) {
				bw.write(node + delimiter);
				for (int i = 0; i < labelList.size(); i++) {
					GroundTerm label = labelList.get(i);
					double score = 0.0;
					if (scores.get(node).containsKey(label))
						score = scores.get(node).get(label);
					bw.write("" + score);
					if (i < labelList.size() - 1)
						bw.write(delimiter);
				}
				bw.write("\n");
			}
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}