package util;

import java.io.PrintStream;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableSet;
import java.util.TreeMap;

public class MapUtils {

	public static NavigableSet<String> sortStringDoubleMap(final HashMap<String,Double> map){
		return sortStringDoubleMap(map, false);
	}

	public static NavigableSet<String> sortStringDoubleMap(final HashMap<String,Double> map, final boolean reverse){
		TreeMap<String,Double> sorted_map = new TreeMap<String,Double>(new Comparator<String>(){

			@Override
			public int compare(String arg0, String arg1) {
				int compare = map.get(arg0).compareTo(map.get(arg1));
				if (compare == 0) return 1;
				else return compare;				

			}
		});
		sorted_map.putAll(map);
		if(reverse){
			return sorted_map.navigableKeySet();
		}
		return sorted_map.descendingKeySet();
	}
	
	public static NavigableSet<String> sortStringIntMap(final HashMap<String,Integer> map){
		TreeMap<String,Integer> sorted_map = new TreeMap<String,Integer>(new Comparator<String>(){

			@Override
			public int compare(String arg0, String arg1) {
				int compare = map.get(arg0).compareTo(map.get(arg1));
				if (compare == 0) return 1;
				else return compare;

			}
		});
		sorted_map.putAll(map);
		return sorted_map.descendingKeySet();
	}
	
	public static LinkedList<String> getTopN(HashMap<String, Double> map, int topN){
		NavigableSet<String> sorted = sortStringDoubleMap(map);

		LinkedList<String> topK = new LinkedList<String>();
		
		int i = 0;
		for(String s : sorted){
			topK.addLast(s);
			if(++i >= topN){
				break;
			}
		}
		return topK;
	}
	
	public static LinkedList<String> getBottomN(HashMap<String, Double> map, int topN){
		NavigableSet<String> sorted = sortStringDoubleMap(map, true);

		LinkedList<String> bottomK = new LinkedList<String>();
		
		int i = 0;
		for(String s : sorted){
			bottomK.addLast(s);
			if(++i >= topN){
				break;
			}
		}
		return bottomK;
	}
	
	public static void add(HashMap<String, Double> map, String key, double val){
		if(!map.containsKey(key)){
			map.put(key, val);
		}
		else{
			map.put(key, map.get(key) + val);
		}
	}
	
	public static void add(HashMap<String, Integer> map, String key, int val){
		if(!map.containsKey(key)){
			map.put(key, val);
		}
		else{
			map.put(key, map.get(key) + val);
		}
	}
	
	public static void normalize(HashMap<String, Double> map){
		double normalizer = 0;
		for(Double d : map.values()){
			normalizer += d;
		}
		if(normalizer!=0){
			for(Entry<String,Double> entry : map.entrySet()){
				entry.setValue(entry.getValue()/normalizer);
			}
		}
	}
	
	public static void printTable(PrintStream out, String orig,
			Map<String, Double> hypProbs) {
		for(Entry<String, Double> e : hypProbs.entrySet()){
			out.println(orig +  " ||| " + e.getKey() + " ||| " + e.getValue());
		}
	}
	
	public static void main(String[] args){
		HashMap<Integer, Double> test = new HashMap<Integer, Double>();
		HashMap<Integer, Double> test2 = new HashMap<Integer, Double>();
		for(int i = 0; i < 100000; i++){
			test.put(i, 0.0+i);
		}
		
		for(int i = 0; i < 1000000; i++){
			test2.put(i, 0.0+i);
		}
		
		System.err.println("testing");
		
		long start = System.currentTimeMillis();
		for(int i = 0; i < 10000; i++){
			test.get(Math.random()*1000000);
		}
		System.err.println("took " + (System.currentTimeMillis() - start) + " milis for test 1");
		
		start = System.currentTimeMillis();
		for(int i = 0; i < 10000; i++){
			test2.get(Math.random()*1000000);
		}
		System.err.println("took " + (System.currentTimeMillis() - start) + " milis for test 2");
	}

}
