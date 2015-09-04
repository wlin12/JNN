package jnn.functions.nlp.app.pos;

import java.util.LinkedList;

import jnn.functions.nlp.aux.input.LabelledData;
import jnn.functions.nlp.aux.input.LabelledDataList;
import jnn.functions.nlp.aux.metrics.WordAccuracyMetric;
import jnn.functions.nlp.aux.metrics.WordBasedEvalMetric;

public class PosSpecification {
	public int charProjectionDim = 50;
	public int charStateDim = 150;
	public int wordProjectionDim = 50;
	public int contextStateDim = 50;
	
	public LabelledDataList taggedDatasets = new LabelledDataList();
	public LabelledDataList validationDatasets = new LabelledDataList();
	public LabelledDataList testDatasets = new LabelledDataList();

	LinkedList<WordBasedEvalMetric> wordErrorMetricsTrain = new LinkedList<WordBasedEvalMetric>();
	LinkedList<WordBasedEvalMetric> wordErrorMetricsValidation = new LinkedList<WordBasedEvalMetric>();
	LinkedList<WordBasedEvalMetric> wordErrorMetricsTest = new LinkedList<WordBasedEvalMetric>();

	public String word2vecEmbeddings;
	public String wordFeatures;
	public String contextModel;
	public String skipngramModel;
	public String skipngramModelFeatures;
	public String outputDir;
	public int sequenceActivation=0;
	
	public PosSpecification() {
		taggedDatasets.scramble = true;
		validationDatasets.scramble = false;
		testDatasets.scramble = false;
	}
	
	public void addDatasetTrain(String file, String type){
		LabelledData dataset = new LabelledData(file, type);
		taggedDatasets.add(dataset);		
		System.err.println("added dataset with " + dataset.sentences.size() + " sentences ");		
	}

	public void addDatasetValidation(String file, String type){
		LabelledData dataset = new LabelledData(file,type);
		validationDatasets.add(dataset);
		System.err.println("added validation dataset with " + dataset.sentences.size() + " sentences ");
	}
	
	public void addDatasetValidation(double ratio){
		validationDatasets.extract(taggedDatasets, ratio);		
	}
	
	public void addDatasetValidation(int numberOfSentences){
		validationDatasets.extract(taggedDatasets, numberOfSentences);		
	}

	public void addDatasetTest(String file, String type){
		LabelledData dataset = new LabelledData(file,type);
		testDatasets.add(dataset);	
		System.err.println("added test dataset with " + dataset.sentences.size() + " sentences ");
	}
	
	public void addMetricTrain(WordBasedEvalMetric metric){
		wordErrorMetricsTrain.add(metric);
	}
	
	public void addMetricValidation(WordBasedEvalMetric metric){
		wordErrorMetricsValidation.add(metric);
	}
	
	public void addMetricTest(WordBasedEvalMetric metric){
		wordErrorMetricsTest.add(metric);
	}
	
	public void addWordErrorMetrics(){
		addMetricTrain(new WordAccuracyMetric("word acc train"));
		addMetricValidation(new WordAccuracyMetric("word acc dev"));
		addMetricTest(new WordAccuracyMetric("word acc test"));
	}
}
