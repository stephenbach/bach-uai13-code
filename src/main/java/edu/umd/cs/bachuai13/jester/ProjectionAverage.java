package edu.umd.cs.bachuai13.jester;

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
 * TODO: description
 * 
 * @author Ben
 *
 */
public class ProjectionAverage implements ExternalFunction {

	private static final ArgumentType[] argTypes = new ArgumentType[]{ ArgumentType.UniqueID };

	private final Predicate p;
	private final int dim;

	public ProjectionAverage(Predicate p, int dim) {
		this.p = p;
		this.dim = dim;
		if (dim < 0 || dim > 1)
			throw new IllegalArgumentException("Dimension must be either 0 or 1");
	}

	@Override
	public int getArity() {
		return 1;
	}

	@Override
	public ArgumentType[] getArgumentTypes() {
		return argTypes;
	}

	@Override
	public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
		/* Get args */
		UniqueID id = (UniqueID) args[0];

		/* Average over all values along dimension <dim> */
		Variable var = new Variable("vec");
		ResultList results;
		double avg = 0.0;

		QueryAtom q = (dim == 0) ? new QueryAtom(p, var, id) : new QueryAtom(p, id, var);
		GroundTerm[] resultArgs = new GroundTerm[2];
		resultArgs[1-dim] = id;
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var);
			resultArgs[dim] = y;
			double val = db.getAtom(p, resultArgs).getValue();
			avg += val;
		}
		if (results.size() > 0)
			avg /= results.size();

		return avg;
	}

	public double getValue(Database db, GroundTerm... args) {
		/* Get args */
		UniqueID id = (UniqueID) args[0];

		/* Average over all values along dimension <dim> */
		Variable var = new Variable("vec");
		ResultList results;
		double avg = 0.0;

		QueryAtom q = (dim == 0) ? new QueryAtom(p, var, id) : new QueryAtom(p, id, var);
		GroundTerm[] resultArgs = new GroundTerm[2];
		resultArgs[1-dim] = id;
		results = db.executeQuery(new DatabaseQuery(q));
		for (int i = 0; i < results.size(); i++) {
			GroundTerm y = results.get(i, var);
			resultArgs[dim] = y;
			double val = db.getAtom(p, resultArgs).getValue();
			avg += val;
		}
		if (results.size() > 0)
			avg /= results.size();

		return avg;
	}

}
