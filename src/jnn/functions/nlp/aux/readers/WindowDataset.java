package jnn.functions.nlp.aux.readers;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Scanner;

import org.apache.commons.math3.util.FastMath;

import vocab.Vocab;
import vocab.WordEntry;

public class WindowDataset {
	String file;
	Scanner reader;
	int windowSize;
	int numberOfWords;
	LinkedList<String> currentWindow = new LinkedList<String>();
	boolean isEnd = false;

	public WindowDataset(String file, int windowSize) {
		this.file = file;
		this.windowSize = windowSize;
		try {
			reader = new Scanner(new File(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
		numberOfWords = windowSize*2+1;
		for(int i = 0; i < numberOfWords; i++){
			currentWindow.addLast(reader.next());
		}
	}

	public String[][][] readWindowsEquivalent(int batchSize, int numberOfBatches, Vocab vocab){
		String[][][] ret = new String[numberOfBatches][][];
		for(int i = 0; i < numberOfBatches; i++){
			ret[i] = new String[batchSize][];
			for(int j = 0; j < batchSize; j++){
				ret[i][j] = readWindow();

				WordEntry centerEntry = vocab.getEntry(ret[i][j][windowSize]);
				while(centerEntry == null){
					ret[i][j] = readWindow();
					centerEntry =vocab.getEntry(ret[i][j][windowSize]);					
				}
			}
		}
		return ret;
	}

	public String[][][] readWindowsEquivalentSkip(int batchSize, int numberOfBatches, Vocab vocab){
		String[][][] ret = new String[numberOfBatches][][];
		for(int i = 0; i < numberOfBatches; i++){
			ret[i] = new String[batchSize][];
			for(int j = 0; j < batchSize; j++){
				ret[i][j] = readWindow();
				WordEntry centerEntry =vocab.getEntry(ret[i][j][windowSize]);
				while(centerEntry == null || 
						(centerEntry.count > 1 && 
								FastMath.random() < 1-((FastMath.log(centerEntry.count)+1)/centerEntry.count))){
					ret[i][j] = readWindow();
					centerEntry =vocab.getEntry(ret[i][j][windowSize]);					
				}
			}
		}
		return ret;
	}

	public String[] readWindow(){
		String[] ret = new String[numberOfWords];
		Iterator<String> it = currentWindow.iterator();
		for(int i = 0; i < numberOfWords; i++){
			ret[i] = it.next();
		}
		String word = reader.next();
		currentWindow.removeFirst();
		currentWindow.addLast(word);
		if(!reader.hasNext()){
			try {
				reader = new Scanner(new File(file));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}		
			isEnd = true;
		}
		return ret;
	}
	
	public void reset(){
		try {
			reader.close();
			reader = new Scanner(new File(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
		for(int i = 0; i < numberOfWords; i++){
			currentWindow.addLast(reader.next());
		}
	}
}
