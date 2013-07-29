package edu.umd.cs.bachuai13.vision;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.bachuai13.vision.PatchStructure.Patch;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabaseQuery;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.Atom;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.QueryAtom;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.predicate.Predicate;

public class ImagePatchUtils {

	static Logger log = LoggerFactory.getLogger(ImagePatchUtils.class);

	public static void insertFromPatchMap(Predicate relation, Database data, Map<Patch, Patch> map) {
		for (Map.Entry<Patch, Patch> e : map.entrySet()) {
			UniqueID A = data.getUniqueID(e.getKey().uniqueID());
			UniqueID B = data.getUniqueID(e.getValue().uniqueID());
			RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(relation, A, B);
			atom.setValue(1.0);
			data.commit(atom);
		}
	}

	/** do not use **/
	public static void insertChildren(Predicate relation, Database data, PatchStructure h) {
		//		for (Map.Entry<Patch, Patch> e : h.getParent().entrySet()) {
		//			UniqueID B = data.getUniqueID(e.getKey().uniqueID());
		//			UniqueID A = data.getUniqueID(e.getValue().uniqueID());
		//			RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(relation, A, B);
		//			atom.setValue(1.0);
		//			data.commit(atom);
		//		}
	}

	public static void insertPixelPatchChildren(Predicate children, Database data, PatchStructure h) {
		for (Patch p : h.getPatches().values()) {
			UniqueID patch = data.getUniqueID(p.uniqueID());
			for (int i : p.pixelList()) {
				UniqueID pixel = data.getUniqueID(i);
				RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(children, patch, pixel);
				atom.setValue(1.0);
				data.commit(atom);
			}
		}
	}

	public static void insertNeighbors(Predicate neighbor, Database data, PatchStructure ps) {
		RandomVariableAtom atom;
		for (Map.Entry<Patch, Patch> e : ps.getNorth().entrySet()) {
			UniqueID A = data.getUniqueID(e.getKey().uniqueID());
			UniqueID B = data.getUniqueID(e.getKey().uniqueID());
			atom = (RandomVariableAtom) data.getAtom(neighbor, A, B);
			atom.setValue(1.0);
			data.commit(atom);
			atom = (RandomVariableAtom) data.getAtom(neighbor, B, A);
			atom.setValue(1.0);
			data.commit(atom);
		}
		for (Map.Entry<Patch, Patch> e : ps.getEast().entrySet()) {	
			UniqueID A = data.getUniqueID(e.getKey().uniqueID());
			UniqueID B = data.getUniqueID(e.getKey().uniqueID());
			atom = (RandomVariableAtom) data.getAtom(neighbor, A, B);
			atom.setValue(1.0);
			data.commit(atom);
			atom = (RandomVariableAtom) data.getAtom(neighbor, B, A);
			atom.setValue(1.0);
			data.commit(atom);
		}
	}

	public static void insertPatchLevels(Database data, PatchStructure h, Predicate level) {
		for (Patch p : h.getPatches().values()) {
			UniqueID L = data.getUniqueID(p.getLevel());
			UniqueID A = data.getUniqueID(p.uniqueID());
			RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(level, A, L);
			atom.setValue(1.0);
			data.commit(atom);
		}
	}

	/**
	 * Assumes image is columnwise vectorized matrix of grayscale values in [0,1]
	 * @param brightness predicate of brightness
	 * @param imageID UniqueID of current image
	 * @param data database to insert pixel values
	 * @param hierarchy 
	 * @param width width of image
	 * @param height height of image
	 * @param image vectorized image
	 * @param mask vectorized mask of which entries to set ground truth on. If null, all entries are entered
	 */
	public static void setPixels(Predicate brightness, UniqueID imageID, Database data, PatchStructure hierarchy, int width, int height, double [] image, boolean [] mask) {
		int k = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (mask == null || mask[k]) {
					UniqueID pixel = data.getUniqueID(k);
					RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(brightness, pixel, imageID);
					atom.setValue(image[k]);
					data.commit(atom);		
				}
				k++;
			}
		}
	}

	public static void populatePixels(int width, int height, Predicate pixelBrightness, Database data, UniqueID imageID) {
		int rv = 0;
		int ov = 0;
		int k = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				UniqueID pixel = data.getUniqueID(k);
				Atom atom = data.getAtom(pixelBrightness, pixel, imageID);
				if (atom instanceof RandomVariableAtom) {
					((RandomVariableAtom) atom).setValue(0.0);
					data.commit((RandomVariableAtom) atom);		
					rv++;
				} else
					ov++;
				k++;
			}
		}

		log.debug("Saw {} random variables, {} observed variables", rv, ov);
	}

	/**
	 * Populate all patches in hierarchy for given image
	 * @param brightness
	 * @param imageID
	 * @param data
	 * @param hierarchy
	 */
	public static void populateAllPatches(Predicate brightness, UniqueID imageID, Database data, PatchStructure hierarchy) {
		log.debug("Populating " + brightness + " on image " + imageID);
		for (Patch p : hierarchy.getPatches().values()) {
			UniqueID patch = data.getUniqueID(p.uniqueID());
			Atom atom = data.getAtom(brightness, patch, imageID);
			if (atom instanceof RandomVariableAtom) {
				data.commit((RandomVariableAtom) atom);
			}
		}
	}
	//	/**
	//	 * Populate all patches in hierarchy for given image
	//	 * @param brightness
	//	 * @param imageID
	//	 * @param data
	//	 * @param hierarchy
	//	 */
	//	public static void populateAllPatches(List<Predicate> brightnessList, UniqueID imageID, Database data, PatchStructure hierarchy) {
	//		for (Patch p : hierarchy.getPatches().values()) {
	//			int level = p.getLevel();
	//			UniqueID patch = data.getUniqueID(p.uniqueID());
	//			Atom atom = data.getAtom(brightnessList.get(level), patch, imageID);
	//			if (atom instanceof RandomVariableAtom) {
	//				data.commit((RandomVariableAtom) atom);
	//			}
	//		}
	//	}

	/**
	 * Loads vectorized image file
	 * @param filename
	 * @param width
	 * @param height
	 * @return
	 */
	public static List<double []> loadImages(String filename, int width, int height) {
		Scanner imageScanner;
		List<double []> images = new ArrayList<double []>();
		try {
			imageScanner = new Scanner(new FileReader(filename));
			while (imageScanner.hasNext() && images.size() < 9999) {
				String line = imageScanner.nextLine();
				String [] tokens = line.split("\t");

				assert(tokens.length == width * height);

				double [] image = new double[width * height];

				for (int i = 0; i < tokens.length; i++) {
					image[i] = Double.parseDouble(tokens[i]);
				}
				images.add(image);
			}
			imageScanner.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return images;
	}


	/**
	 * precomputes patch brightness from fully-observed database
	 */
	public static void computePatchBrightness(Predicate brightness, Predicate pixelBrightness, Database data, UniqueID imageID, PatchStructure ps, double [] image) {
		log.debug("Computing patch brightness for image {}", imageID);
		for (Patch p : ps.getPatches().values()) {
			UniqueID patch = data.getUniqueID(p.uniqueID());
			RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(brightness, patch, imageID);
			double sum = 0.0;
			for (Integer pixel : p.pixelList()) {
				sum += image[pixel];
			}
			atom.setValue(sum / p.pixelList().size());
		}
	}

	/**
	 * 
	 */
	public static void computeNeighborBrightness(Predicate neighborBrightness, Predicate brightness, Predicate neighbors, Database data, UniqueID imageID, PatchStructure ps) {
		log.debug("Computing neighbor brightness for image {}", imageID);
		for (Patch p : ps.getPatches().values()) {
			UniqueID patch = data.getUniqueID(p.uniqueID());
			RandomVariableAtom atom = (RandomVariableAtom) data.getAtom(neighborBrightness, patch, imageID);
			double sum = 0.0;
			Variable neigh = new Variable("neighbor");
			QueryAtom q = new QueryAtom(neighbors, patch, neigh);
			ResultList list = data.executeQuery(new DatabaseQuery(q));
			if (list.size() > 0) {
				for (int i = 0; i < list.size(); i++) {
					GroundAtom nb = data.getAtom(brightness, list.get(i)[0], imageID);
					sum += nb.getValue();
				}
				atom.setValue(sum / list.size());
			}
		}
	}
}
