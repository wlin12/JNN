package jnn.functions.nlp.aux.input;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedList;

import util.IOUtils;
import util.MathUtils;

public class LabelledDataList {
	public ArrayList<LabelledData> taggedDatasets = new ArrayList<LabelledData>();
	int datasetId = 0;
	int sampleId = 0;
	public int numberOfSamples = 0;
	public boolean scramble = true;
	
	public void add(LabelledData data){
		taggedDatasets.add(data);
		numberOfSamples+=data.getSize();
	}
	
	public void extract(LabelledDataList dataset, double ratio) {
		for(LabelledData data : dataset.taggedDatasets){
			int numberToExtract = (int)(data.sentences.size()*ratio);
			LabelledData newData = new LabelledData();
			dataset.numberOfSamples -= numberToExtract;
			while(numberToExtract > 0){
				numberToExtract--;
				newData.sentences.add(data.sentences.remove(numberToExtract));
			}
			add(newData);
		}
	}
	
	public void extract(LabelledDataList dataset, int numberOfSentences) {
		for(LabelledData data : dataset.taggedDatasets){
			int numberToExtract = numberOfSentences;
			LabelledData newData = new LabelledData();
			dataset.numberOfSamples -= numberToExtract;
			while(numberToExtract > 0){
				numberToExtract--;
				newData.sentences.add(data.sentences.remove(numberToExtract));
			}
			add(newData);
		}
	}
	
	public ArrayList<LabelledData> getTaggedDatasets() {
		return taggedDatasets;
	}
	
	public LabelledSentence getNextSentence(){
		LabelledSentence ret = taggedDatasets.get(datasetId).getSentences().get(sampleId);
		advance();
		return ret;
	}
	
	public LabelledSentence[] getNextBatch(int size){
		LabelledSentence[] batch = new LabelledSentence[size];
		for(int i = 0 ; i < size; i++){
			batch[i] = getNextSentence();
		}
		return batch;
	}
	
	public LabelledSentence[][] readBatchesEquivalent(int size, int numBatches){
		LabelledSentence[][] batch = new LabelledSentence[numBatches][size];
		int[] batchSizes = new int[numBatches];
		int[] batchCount = new int[numBatches];
		for(int i = 0; i < size*numBatches; i++){
			LabelledSentence sent = getNextSentence();
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
	
	public LabelledSentence[][] readBatchesEquivalent(int size, int numBatches, LinkedList<int[]> order){
		LabelledSentence[][] batch = new LabelledSentence[numBatches][size];
		int[] batchSizes = new int[numBatches];
		int[] batchCount = new int[numBatches];
		for(int i = 0; i < size*numBatches; i++){
			LabelledSentence sent = getNextSentence();
			int minIndex = MathUtils.minIndex(batchSizes);
			int num = batchCount[minIndex]++;
			batch[minIndex][num] = sent;
			order.add(new int[]{minIndex,num});
			if(batchCount[minIndex] == size){
				batchSizes[minIndex] = Integer.MAX_VALUE;
			}
			else{
				batchSizes[minIndex] += sent.tokens.length;
			}
		}
		return batch;
	}
	
	public void advance(){
		sampleId++;
		if(sampleId == taggedDatasets.get(datasetId).getSize()){
			sampleId=0;
			if(scramble){
				taggedDatasets.get(datasetId).scramble();
			}
			datasetId++;
			if(datasetId == taggedDatasets.size()){
				datasetId=0;
			}
		}
	}
	
	public void printToFile(String file, boolean printTags){
		PrintStream out = IOUtils.getPrintStream(file);
		for(LabelledData dataset : taggedDatasets){
			for(LabelledSentence sent : dataset.sentences){
				String sentForm = "";
				for(int i = 0; i < sent.tokens.length; i++){
					sentForm+=sent.tokens[i];
					if(printTags){
						sentForm+="_"+sent.tags[i];
					}
					sentForm+=" ";
				}
				out.println(sentForm.trim());
			}
		}
		out.close();
	}
	
	public void printToFile(PrintStream out, boolean printTags){
		for(LabelledData dataset : taggedDatasets){
			for(LabelledSentence sent : dataset.sentences){
				String sentForm = "";
				for(int i = 0; i < sent.tokens.length; i++){
					sentForm+=sent.tokens[i];
					if(printTags){
						sentForm+="_"+sent.tags[i];
					}
					sentForm+=" ";
				}
				out.println(sentForm.trim());
			}
		}
		
	}
	
	public int getNumberOfSamples() {
		return numberOfSamples;
	}

	
}
