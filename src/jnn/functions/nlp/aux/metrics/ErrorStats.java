package jnn.functions.nlp.aux.metrics;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedList;

public class ErrorStats{
	public static final int MAX_ERROR_LIST = 5;
	public double errorSum;
	public int norm;

	public double minError;
	public int minErrorIteration;
	public LinkedList<Double> errorsPerIteration;
	
	boolean reverse = false;

	public void init(){
		minError = Double.MAX_VALUE;
		minErrorIteration = -1;
		errorsPerIteration = new LinkedList<Double>();
	}

	public double commitError(){
		double error = errorSum/norm;
		if(minError > error){
			minErrorIteration = errorsPerIteration.size();
			minError = error;
		}
		errorsPerIteration.add(error);
		return error;
	}

	public void addError(double error){
		errorSum += error;
		norm++;
	}
	
	public void addError(ErrorStats error){
		errorSum += error.errorSum;
		norm+= error.norm;
	}

	public void initError(){
		errorSum = 0;
		norm = 0;
	}

	public String errorPerIterationStr(boolean prune){
		double prev = Double.MAX_VALUE;
		LinkedList<String> ret = new LinkedList<String>();
		int start = 0;
		if(prune){
			start = errorsPerIteration.size() - Math.min(MAX_ERROR_LIST, errorsPerIteration.size());
		}
		
		for(int i = start ; i < errorsPerIteration.size(); i++){
			double d = errorsPerIteration.get(i);
			String el = ""+d;
			if(reverse){
				el = ""+(1-d);
			}
			if(prev > d){		
				el += "(+" + ((prev-d)*100/prev) +"%) ";
			}
			else{
				el += "(-" + ((d-prev)*100/prev) +"%) ";
			}
			ret.addFirst(el);
			prev = d;
		}
		return ret.toString();
	}

	public void loadFromErrorString(String error){
		String[] errors = error.split("\\s+");
		init();
		for(String s : errors){
			String iterError = s.split("\\(")[0];
			double iterErrorVal = Double.parseDouble(iterError);
			initError();
			addError(iterErrorVal);
			commitError();
		}
	}

	public void loadFromFile(BufferedReader reader){
		try {
			String title = reader.readLine();
			String best = reader.readLine();
			String errors = reader.readLine();
			loadFromErrorString(errors);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public String bestIterationStr(){
		if(reverse){
			return "best iteration " + minErrorIteration + " -> " + (1-minError);
		}
		else{
			return "best iteration " + minErrorIteration + " -> " + minError;			
		}
	}


	public void displayResults(String title){
		displayResults(title, System.err);
	}
	
	public void displayResults(String title, boolean prune){
		displayResults(title, System.err, prune);
	}

	public void displayResults(String title, PrintStream out){
		out.println(title);
		out.println(bestIterationStr());
		out.println(errorPerIterationStr(false));
	}
	
	public void displayResults(String title, PrintStream out, boolean prune){
		out.println(title);
		out.println(bestIterationStr());
		out.println(errorPerIterationStr(prune));
	}

	public double getImprovement(int iterations){
		int init = errorsPerIteration.size()-iterations;
		if(init < 0){
			return 1;
		}
		return (errorsPerIteration.get(init)-errorsPerIteration.getLast())/errorsPerIteration.get(init);
	}

	public boolean isBestIteration() {
		return minErrorIteration == errorsPerIteration.size() - 1;
	}

	public void setReverse() {
		reverse = true;		
	}
	
	public double getScoreAtIteration(int iteration){
		if(reverse){
			return 1 - errorsPerIteration.get(iteration);
		}
		else{
			return errorsPerIteration.get(iteration);
		}
	}
}
