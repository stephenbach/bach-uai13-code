package edu.umd.cs.bachuai13.jester;

import java.util.HashMap;
import java.util.Map;

import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabaseQuery;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.QueryAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.model.predicate.Predicate;

/**
 * Implements the adjusted cosine similarity (ACS).
 * <p>
 * TODO: description
 * 
 * @author Ben
 *
 */
class AdjCosineSimilarity implements ExternalFunction
{

	private static final ArgumentType[] argTypes = new ArgumentType[]{ ArgumentType.UniqueID, ArgumentType.UniqueID };
	
	private final Predicate p;
	private final Predicate avg;
	private final int dim;
	private final double thresh;
	
	/**
	 * 
	 * @param p Binary predicate whose truth value is to be used to compute similarity
	 * @param dim index of free variable (zero-based): must be {0, 1}
	 * @param avg predicate containing average value 
	 */
	public AdjCosineSimilarity(Predicate p, int dim, Predicate avg) {
		this.p = p;
		this.avg = avg;
		this.dim = dim;
		this.thresh = 0;
	}

	/**
	 * 
	 * @param p Binary predicate whose truth value is to be used to compute similarity
	 * @param dim Index of free variable (zero-based): must be {0, 1}
	 * @param avg Predicate containing average value 
	 * @param thresh Threshold below which the returned similarity is zero 
	 */
	public AdjCosineSimilarity(Predicate p, int dim, Predicate avg, double thresh) {
		this.p = p;
		this.avg = avg;
		this.dim = dim;
		this.thresh = thresh;
	}
	
	@Override
	public int getArity() {
		return 2;
	}

	@Override
	public ArgumentType[] getArgumentTypes() {
		return argTypes;
	}
	
	@Override
	public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
		/* Get args */
		UniqueID x1 = (UniqueID) args[0];
		UniqueID x2 = (UniqueID) args[1];
		
		ResultList results;
		QueryAtom q;
		GroundTerm[] grounding = new GroundTerm[p.getArity()];
		
		/* Query database for x1 vector */
		HashMap<GroundTerm,Double> set1 = new HashMap<GroundTerm,Double>(); 
		Variable var1 = new Variable("vec1");
		q = (dim == 0) ? new QueryAtom(p, var1, x1) : new QueryAtom(p, x1, var1);
		grounding[1-dim] = x1; // TODO: This indexing depends the predicate being binary
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var1);
			grounding[dim] = y; 
			double val = db.getAtom(p, grounding).getValue();
			set1.put(y, val);
		}

		/* Query database for x2 vector */
		HashMap<GroundTerm,Double> set2 = new HashMap<GroundTerm,Double>(); 
		Variable var2 = new Variable("vec1");
		q = (dim == 0) ? new QueryAtom(p, var2, x2) : new QueryAtom(p, x2, var2);
		grounding[1-dim] = x2;
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var2);
			grounding[dim] = y;
			double val = db.getAtom(p, grounding).getValue();
			set2.put(y, val);
		}
		
		/* Compute adjusted cosine similarity */
		double numer = 0.0;
		double denom1 = 0.0;
		double denom2 = 0.0;
		for (Map.Entry<GroundTerm, Double> e : set1.entrySet()) {
			GroundTerm y = e.getKey();
			if (set2.containsKey(y)) {
				double a = db.getAtom(avg, y).getValue();
				double v1 = e.getValue() - a;
				double v2 = set2.get(y) - a;
				numer += v1 * v2;
				denom1 += v1 * v1;
				denom2 += v2 * v2;
			}
		}
		if (denom1 == 0.0 || denom2 == 0.0)
			return 0;
		
		/* Similarity (clamped to [0,1]) */
		double sim = numer / (Math.sqrt(denom1) * Math.sqrt(denom2));
		sim = (sim + 1) / 2;
		sim = Math.max(0, Math.min(1, sim));
		/* Threshold similarity */
		if (sim < thresh)
			return 0.0;
		return sim;
	}

	public double getValue(Database db, GroundTerm... args) {
		/* Get args */
		UniqueID x1 = (UniqueID) args[0];
		UniqueID x2 = (UniqueID) args[1];
		
		ResultList results;
		QueryAtom q;
		GroundTerm[] grounding = new GroundTerm[p.getArity()];
		
		/* Query database for x1 vector */
		HashMap<GroundTerm,Double> set1 = new HashMap<GroundTerm,Double>(); 
		Variable var1 = new Variable("vec1");
		q = (dim == 0) ? new QueryAtom(p, var1, x1) : new QueryAtom(p, x1, var1);
		grounding[1-dim] = x1; // TODO: This indexing depends the predicate being binary
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var1);
			grounding[dim] = y; 
			double val = db.getAtom(p, grounding).getValue();
			set1.put(y, val);
		}

		/* Query database for x2 vector */
		HashMap<GroundTerm,Double> set2 = new HashMap<GroundTerm,Double>(); 
		Variable var2 = new Variable("vec1");
		q = (dim == 0) ? new QueryAtom(p, var2, x2) : new QueryAtom(p, x2, var2);
		grounding[1-dim] = x2;
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var2);
			grounding[dim] = y;
			double val = db.getAtom(p, grounding).getValue();
			set2.put(y, val);
		}
		
		/* Compute adjusted cosine similarity */
		double numer = 0.0;
		double denom1 = 0.0;
		double denom2 = 0.0;
		for (Map.Entry<GroundTerm, Double> e : set1.entrySet()) {
			GroundTerm y = e.getKey();
			if (set2.containsKey(y)) {
				double a = db.getAtom(avg, y).getValue();
				double v1 = e.getValue() - a;
				double v2 = set2.get(y) - a;
				numer += v1 * v2;
				denom1 += v1 * v1;
				denom2 += v2 * v2;
			}
		}
		if (denom1 == 0.0 || denom2 == 0.0)
			return 0;
		
		/* Similarity (clamped to [0,1]) */
		double sim = numer / (Math.sqrt(denom1) * Math.sqrt(denom2));
		sim = (sim + 1) / 2;
		sim = Math.max(0, Math.min(1, sim));
		/* Threshold similarity */
		if (sim < thresh)
			return 0.0;
		return sim;
	}

}

