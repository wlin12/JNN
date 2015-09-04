package jnn.functions.nlp.app.pretraining;

import jnn.functions.nlp.aux.readers.PlainTextReader;
import jnn.functions.nlp.aux.readers.TextDataset;
import jnn.functions.nlp.aux.readers.WindowDataset;

public class StructuredSkipngramSpecification {
	public int windowSize = 5;
	public int negativeSamples = 10;
	public int maxTypeInput = 1000000;
	public int maxTypeOutput = 1000000;
	public int batchSize = 100;
	public int minCount = 5;
	
	public int charProjectionDim = 50;
	public int charStateDim = 150;
	public int wordProjectionDim = 50;
	public int contextStateDim = 50;
	public String outputDir;

	public String wordFeatures = "words,capitalization";
	public String word2vecEmbeddings;
	
	WindowDataset trainingData;
	WindowDataset devData;
	String trainFile;
	String devFile;
	boolean useShortTermMemory = true;
	
	public StructuredSkipngramSpecification() {
	}
	
	public void setDataset(String file){
		trainFile = file;
		trainingData = new WindowDataset(file, windowSize);
	}
	
	public void setDevDataset(String file){
		devFile = file;
		devData = new WindowDataset(file, windowSize);
	}
}
