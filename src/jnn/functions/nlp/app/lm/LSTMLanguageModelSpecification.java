package jnn.functions.nlp.app.lm;

import jnn.functions.nlp.aux.readers.PlainTextReader;
import jnn.functions.nlp.aux.readers.TextDataset;

public class LSTMLanguageModelSpecification {
	public int maxType = 10000000;
	
	public int charProjectionDim = 50;
	public int charStateDim = 150;
	public int wordProjectionDim = 50;
	
	public int softmaxCharDim  = 50;
	public int softmaxStateDim = 150;

	public int decoderStateDim = 150;

	public String wordFeatures = "words,capitalization";
	public String softmaxType = "words";
	public String word2vecEmbeddings;
	
	public String outputDir;
	
	TextDataset trainingData = new TextDataset();
	TextDataset validationData = new TextDataset();
	TextDataset testData = new TextDataset();
	
	public LSTMLanguageModelSpecification() {
	}
	
	public void addDataset(String file){
		trainingData.add(file, new PlainTextReader());
	}
	
	public void addValidationDataset(String file){
		validationData.add(file, new PlainTextReader());
	}
	
	public void addTestDataset(String file){
		testData.add(file, new PlainTextReader());
	}
}
