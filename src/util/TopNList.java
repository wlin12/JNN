package util;

import java.util.Iterator;
import java.util.LinkedList;

public class TopNList<Type> {

	LinkedList<Type> objectList = new LinkedList<Type>();
	LinkedList<Double> objectScore = new LinkedList<Double>();
	
	int numberOfElements = 10;
	
	public TopNList(int num) {
		this.numberOfElements = num;
	}
	
	public void add(Type obj, double score){
		Iterator<Double> it = objectScore.iterator();
		int index = 0;
		while(it.hasNext()){
			double next = it.next();
			if(next < score){
				break;
			}
			index++;
		}
		objectScore.add(index, score);
		objectList.add(index, obj);
		if(objectList.size() > numberOfElements){
			objectList.removeLast();
			objectScore.removeLast();
		}
	}
	
	public boolean containsKey(Type key){
		return objectList.contains(key);
	}
	
	public LinkedList<Type> getObjectList() {
		return objectList;
	}
	
	public LinkedList<Double> getObjectScore() {
		return objectScore;
	}
	
	public static void main(String[] args) {
		TopNList<String> list = new TopNList<String>(5);
		list.add("a", 5);
		list.add("b", 3);
		list.add("c", 4);
		list.add("d", 2);
		list.add("e", 6);
		list.add("f", 9);
		list.add("g", 1);
		for(String s: list.objectList){
			System.err.println(s);
		}
	}

	public void print(String string) {
		Iterator<Type> objectIt = objectList.iterator();
		Iterator<Double> objectScoreIt = objectScore.iterator();
		while(objectIt.hasNext()){
			System.err.println(objectIt.next() + " -> " + objectScoreIt.next());
			
		}
	}
	
	@Override
	public String toString() {
		String ret = "";
		Iterator<Type> objectIt = objectList.iterator();
		while(objectIt.hasNext()){
			ret += objectIt.next();
			if(objectIt.hasNext()){
				ret += ",";
			}
		}
		return ret;
	}

	public int size(){
		return objectList.size();
	}
}
