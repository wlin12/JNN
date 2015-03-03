package util;

import java.util.ArrayList;
import java.util.HashMap;

public class DropoutMatrixGen {
	public static HashMap<Integer, HashMap<Integer, HashMap<Double, DropoutMatrixGen>>> generators = new HashMap<Integer, HashMap<Integer,HashMap<Double,DropoutMatrixGen>>>();
	
	public static boolean[][] gen(int sizeX, int sizeY, double dropoutRate){
		if(!generators.containsKey(sizeX)){
			generators.put(sizeX, new HashMap<Integer, HashMap<Double, DropoutMatrixGen>>());
		}
		HashMap<Integer, HashMap<Double, DropoutMatrixGen>> genX = generators.get(sizeX);
		if(!genX.containsKey(sizeY)){
			genX.put(sizeY, new HashMap<Double, DropoutMatrixGen>());
		}
		HashMap<Double, DropoutMatrixGen> genY = genX.get(sizeY);
		if(!genY.containsKey(dropoutRate)){
			genY.put(dropoutRate, new DropoutMatrixGen(sizeX, sizeY, dropoutRate));
		}
		DropoutMatrixGen gen = genY.get(dropoutRate);
		return gen.gen();
	}
	
	public static int maxSize = 10000;
	public int sizeX;
	public int sizeY;
	public double dropoutRate;
	XorRandom rand = new XorRandom(2);
	
	public ArrayList<boolean[][]> matrixes = new ArrayList<boolean[][]>();

	public DropoutMatrixGen(int sizeX, int sizeY, double dropoutRate) {
		super();
		this.sizeX = sizeX;
		this.sizeY = sizeY;
		this.dropoutRate = dropoutRate;
		for(int i = 0; i < maxSize; i++){
			matrixes.add(genNew());
		}	
	}
	
	public boolean[][] gen(){
		if(rand.nextDouble() < 0.0001){
			matrixes.add(genNew());
			matrixes.remove(0);
		}
		return matrixes.get(rand.nextInt(maxSize));
	}
	
	public boolean[][] genNew(){
		boolean[][] matrix = new boolean[sizeX][sizeY];
		for(int i = 0; i < sizeX; i++){
			for(int j = 0; j < sizeY; j++){
				if(rand.nextDouble() < dropoutRate){
					matrix[i][j]=true;
				}
			}
		}
		return matrix;
	}
}
