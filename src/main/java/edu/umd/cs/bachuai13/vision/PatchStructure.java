package edu.umd.cs.bachuai13.vision;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.config.ConfigBundle;

public class PatchStructure {

	Logger log = LoggerFactory.getLogger(PatchStructure.class);

	static boolean useStringIDs = false;

	public PatchStructure(int width, int height, int branching, int depth, ConfigBundle config) {
		this.width = width;
		this.height = height;
		this.branching = branching;
		this.depth = depth;

		parent = new HashMap<Patch, Patch>();
		north = new HashMap<Patch, Patch>();
		east = new HashMap<Patch, Patch>();
		mirrorHorizontal = new HashMap<Patch, Patch>();
		mirrorVertical = new HashMap<Patch, Patch>();
		patches = new HashMap<String, Patch>();
		patchCounter = 0;

		useStringIDs = config.getBoolean("rdbmsdatastore.usestringids", false);
	}

	public void generatePixels() {
		int k = 0;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				Patch current = new Patch(k, x, y, 1, 1, width, height);
				patches.put(x + "," + y, current);
				k++;
			}
		}
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				Patch current = patches.get(x + "," + y);
				Patch hm = patches.get((width - x - 1) + "," + y);
				Patch vm = patches.get(x + "," + (height - y - 1));
				mirrorHorizontal.put(current, hm);
				mirrorVertical.put(current, vm);
				if (x < width - 1) {
					Patch neighbor = patches.get((x + 1) + "," + y);
					east.put(current,  neighbor);
				} 
				if (y < height - 1) {
					Patch neighbor = patches.get(x + "," + (y + 1));
					north.put(current,  neighbor);
				} 
			}
		}
	}
	
	public void generateGridResolution(int resolution) {
		ArrayList<Integer> startXs = new ArrayList<Integer>(resolution+1);
		ArrayList<Integer> startYs = new ArrayList<Integer>(resolution+1);
		for (int i = 0; i < resolution; i++) {
			startXs.add(i, (int) (0.5 + i * width / resolution));
			startYs.add(i, (int) (0.5 + i * height / resolution));
		}
		startXs.add(resolution, width);
		startYs.add(resolution, height);
		
		for (int i = 0; i < resolution; i++) {
			for (int j = 0; j < resolution; j++) {
				int startX = startXs.get(i);
				int startY = startYs.get(j);
				int myWidth = startXs.get(i+1) - startX;
				int myHeight = startYs.get(j+1) - startY;
				Patch current = new Patch(startX, startY, resolution, myWidth, myHeight, height, width);
				patches.put("(" + resolution + "," + i + "," + j + ")", current);
			}
		}
		for (int i = 0; i < resolution; i++) {
			for (int j = 0; j < resolution; j++) {
				Patch current = patches.get("(" + resolution + "," + i + "," + j + ")");
				if (j < resolution - 1) {
					Patch n = patches.get("(" + resolution + "," + i + "," + (j+1) + ")");
					north.put(current, n);
				}
				if (i < resolution - 1) {
					Patch e = patches.get("(" + resolution + "," + (i+1) + "," + j + ")");
					east.put(current, e);
				}
				
				Patch hm = patches.get("(" + resolution + "," + (resolution - i - 1) + "," + j + ")");
				Patch vm = patches.get("(" + resolution + "," + i + "," + (resolution - j - 1) + ")");
				mirrorHorizontal.put(current, hm);
				mirrorVertical.put(current, vm);
			}
		}
	}

	public void generateHierarchy() {
//		generateGridResolution(2);
//		generateGridResolution(4);
//		generateGridResolution(8);
//		generateGridResolution(16);
//		generateGridResolution(32);
//		generateGridResolution(48);
		//generateGridResolution(64);
//		generateGridResolution(100);

		generateGridResolution(width);
//		generateRowPatches();
//		generateColumnPatches();
		log.debug("Generated {} patches", patchCounter);
	}

	public void generateRowPatches() {
		for (int i = 0; i < height; i++) {
			// as a temporary hack, use negative 2 as "level" for rows
			Patch row = new Patch(0, i, -2, width, 1, height, width);
			patches.put(locationID(0, i, -2), row);

			if (i > 0) {
				Patch neighbor = patches.get(locationID(0, i-1, -2));
				north.put(neighbor, row);
			}
		}
		for (int i = 0; i < height; i++) {
			Patch current = patches.get(locationID(0, i, -2));
			Patch mirror = patches.get(locationID(0, height - 1 - i, -2));
			mirrorVertical.put(current, mirror);
		}
	}


	public void generateColumnPatches() {
		for (int i = 0; i < width; i++) {
			// as a temporary hack, use negative 1 as "level" for columns
			Patch col = new Patch(i, 0, -1, 1, height, height, width);
			patches.put(locationID(i, 0, -1), col);

			if (i > 0) {
				Patch neighbor = patches.get(locationID(i-1, 0, -1));
				east.put(neighbor, col);
			}
		}
		for (int i = 0; i < width; i++) {
			Patch current = patches.get(locationID(i, 0, -1));
			Patch mirror = patches.get(locationID(height - 1 - i, 0, -1));
			mirrorHorizontal.put(current, mirror);
		}
	}

	public void generateGrid() {
		int level = 0;
		double myWidth = width;
		double myHeight = height;
		while (level < depth) {
			// connect up all neighbors at level 
			double x = 0;
			while (x < width) {
				double y = 0;
				while (y < height) {
					log.trace("Adding patch starting at "+ x + "," + y);
					Patch current = patches.get(locationID(x, y, level));
					// north
					if (y + myHeight < height) {
						Patch neighbor = patches.get(locationID(x, y + myHeight, level));
						north.put(current, neighbor);
					}
					// east
					if (x + myWidth < width) {
						Patch neighbor = patches.get(locationID(x + myWidth, y, level));
						east.put(current, neighbor);
						if (neighbor == null) {
							log.debug("found null at level {}, depth {}", level, depth);
						}
					}

					y += myHeight;
				}
				x += myWidth;
			}

			level++;

			myWidth /= branching;
			myHeight /= branching;
		}
	}

	public void generateMirrors() {
		int level = 0;
		double myWidth = width;
		double myHeight = height;
		while (level < depth) {
			// connect up all mirror patches at level 
			double x = 0;
			while (x < width) {
				double y = 0;
				while (y < height) {
					Patch current = patches.get(locationID(x, y, level));
					Patch mh = patches.get(locationID(width - myWidth - x, y, level));
					Patch mv = patches.get(locationID(x, height - myHeight - y, level));
					mirrorHorizontal.put(current, mh);
					mirrorVertical.put(current, mv);

					y += myHeight;
				}
				x += myWidth;
			}

			level++;
			if (level > depth || myWidth <= 1.0 || myHeight <= 1.0)
				break;
			myWidth /= branching;
			myHeight /= branching;
		}
	}

	public void generateHierarchy(double x, double y, int level, Patch parentPatch) {
		double myWidth = width / Math.pow(branching, level);
		double myHeight = height / Math.pow(branching, level);
		double childWidth = width / Math.pow(branching, level + 1);
		double childHeight = height / Math.pow(branching, level + 1);

		if (level < depth) {
			// create current patch
			Patch current = new Patch(x, y, level, myWidth, myHeight, height, width);

			if (parentPatch != null) 
				parent.put(current, parentPatch);
			patches.put(current.toString(), current);

			if (childWidth >= 1 && childHeight >= 1) {
				double childx = x;
				while (childx < x + myWidth) {
					double childy = y;
					while (childy < y + myHeight) {
						generateHierarchy((int) childx, (int) childy, level + 1, current);

						childy += childHeight; 
					}
					childx += childWidth;
				}
			}
		}
	}

	public Map<Patch, Patch> getParent() {
		return parent;
	}

	public Map<Patch, Patch> getNorth() {
		return north;
	}

	public Map<Patch, Patch> getEast() {
		return east;
	}

	public Map<Patch, Patch> getMirrorHorizontal() {
		return mirrorHorizontal;
	}

	public Map<Patch, Patch> getMirrorVertical() {
		return mirrorVertical;
	}

	public Map<String, Patch> getPatches() {
		return patches;
	}

	public Patch pixelPatch(int x, int y) {
		return patches.get(locationID(x, y, depth));
	}

	public int getDepth() {
		return depth;
	}


	private int branching;
	private int width;
	private int height;
	private int depth;
	private int patchCounter;

	private Map<Patch, Patch> parent;
	private Map<Patch, Patch> north;
	private Map<Patch, Patch> east;
	private Map<Patch, Patch> mirrorHorizontal;
	private Map<Patch, Patch> mirrorVertical;	
	private Map<String, Patch> patches;


	public class Patch {
		public Patch(int id, int startx, int starty, int myWidth, int myHeight, int width, int height) {
			this.startx = startx;
			this.starty = starty;
			this.id = id;
			this.patchWidth = myWidth;
			this.patchHeight = myHeight;
			this.imageHeight = height;
			this.imageWidth = width;
		}
		public Patch(double startx, double starty, int level, double myWidth, double myHeight, double height, double width) {
			this.startx = startx;
			this.starty = starty;
			this.patchWidth = myWidth;
			this.patchHeight = myHeight;
			this.imageHeight = height;
			this.imageWidth = width;
			string = locationID(startx, starty, level);
			id = useStringIDs ? string : patchCounter;
			this.level = level;
			patchCounter++;
		}

		double startx;
		double starty;
		double patchWidth;
		double patchHeight;
		double imageHeight;
		double imageWidth;
		String string;
		Object id;
		int level;
		
		public boolean hasNorth() {
			return starty + patchHeight < imageHeight;
		}
		public boolean hasSouth() {
			return starty > 0;
		}
		public boolean hasEast() {
			return startx + patchWidth < imageWidth;
		}
		public boolean hasWest() {
			return startx > 0;
		}
		
		public String toString() {
			return string;
		}

		public Object uniqueID() {
			return id;
		}

		public int getLevel() {
			return level;
		}

		public List<Integer> pixelList() {
			List<Integer> pixels = new ArrayList<Integer>();
			for (double x = startx; x < (int) (startx + patchWidth); x += 1.0)
				for (double y = starty; y < (int) (starty + patchHeight); y += 1.0)
					pixels.add(vectorizeCoordinate(x, y));
			if (startx + patchHeight > width) 
				pixels.contains(startx);
			return pixels;
		}

		private int vectorizeCoordinate(double x, double y) {
			return (int) (((int) x) * imageHeight + y);
		}
	}

	public static String locationID(double x, double y, int level) {
		return "(" + (int) (x + 0.5) + "," + (int) (y + 0.5) + "," + level + ")";  
	}
}
