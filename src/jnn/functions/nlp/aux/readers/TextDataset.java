package jnn.functions.nlp.aux.readers;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;

import jnn.functions.nlp.aux.input.InputSentence;
import util.IOUtils;
import util.LangUtils;
import util.MathUtils;

public class TextDataset{
	public ArrayList<String> files = new ArrayList<String>();
	public ArrayList<TextReader> readers = new ArrayList<TextReader>();
	BufferedReader currentReader = null;
	int currentFileIndex = 0;
	boolean isEnd = false;
	long numberOfSentences = -1;

	public long countSentences(){
		if(numberOfSentences == -1){
			numberOfSentences = 0;
			reset();
			while(!isEnd()){
				read();
				numberOfSentences++;
			}
		}
		return numberOfSentences;
	}

	public void add(String file, TextReader reader){
		if(files.size()==0){
			currentFileIndex = 0;
			currentReader = IOUtils.getReader(file);
		}
		files.add(file);
		readers.add(reader);
	}

	public InputSentence[] readBatch(int size){
		InputSentence[] batch = new InputSentence[size];
		for(int i = 0; i < size; i++){
			batch[i] = read();
		}
		return batch;
	}

	public InputSentence[][] readBatchesEquivalent(int size, int numBatches){
		InputSentence[][] batch = new InputSentence[numBatches][size];
		int[] batchSizes = new int[numBatches];
		int[] batchCount = new int[numBatches];
		for(int i = 0; i < size*numBatches; i++){
			InputSentence sent = read();
			int minIndex = MathUtils.minIndex(batchSizes);
			batch[minIndex][batchCount[minIndex]++] = sent;
			if(batchCount[minIndex] == size){
				batchSizes[minIndex] = Integer.MAX_VALUE;
			}
			else{
				batchSizes[minIndex] += sent.tokens.length;
			}
		}
		return batch;
	}

	public InputSentence[][] readBatchesEquivalent(int size, int numBatches, LinkedList<int[]> indexes){
		InputSentence[][] batch = new InputSentence[numBatches][size];
		int[] batchSizes = new int[numBatches];
		int[] batchCount = new int[numBatches];
		for(int i = 0; i < size*numBatches; i++){
			InputSentence sent = read();
			int minIndex = MathUtils.minIndex(batchSizes);
			int count = batchCount[minIndex];
			batch[minIndex][count] = sent;
			indexes.addLast(new int[]{minIndex,count});
			batchCount[minIndex]++;
			if(batchCount[minIndex] == size){
				batchSizes[minIndex] = Integer.MAX_VALUE;
			}
			else{
				batchSizes[minIndex] += sent.tokens.length;
			}
		}
		return batch;
	}

	public InputSentence read(){
		try {
			isEnd = false;
			String line = currentReader.readLine();
			if(LangUtils.isStringEmpty(line)){
				return read();
			}
			else{
				InputSentence ret = readers.get(currentFileIndex).readSentence(line);
				if(!currentReader.ready()){
					currentReader.close();
					currentFileIndex = (currentFileIndex+1)%files.size();			
					currentReader = IOUtils.getReader(files.get(currentFileIndex));
					if(currentFileIndex == 0){
						isEnd = true;
					}
				}
				return ret;
			}

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public int size(){
		return files.size();
	}

	public boolean isEnd(){
		return isEnd;
	}

	public void reset(){
		isEnd = false;
		if(currentReader != null){
			try {
				currentReader.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		currentFileIndex = 0;
		currentReader = IOUtils.getReader(files.get(currentFileIndex));
	}
}
