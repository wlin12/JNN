package jnn.functions.nlp.app.labeller;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import jnn.functions.composite.SoftmaxObjectiveLayer;
import jnn.functions.nlp.app.pos.PosTaggerInstance;
import jnn.functions.nlp.aux.input.LabelledData;
import jnn.functions.nlp.aux.input.LabelledSentence;
import jnn.functions.nlp.aux.metrics.WordAccuracyMetric;
import jnn.functions.nlp.aux.metrics.WordBasedEvalMetric;
import jnn.functions.nlp.labeling.WordTaggingLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nlp.words.WordWithContextRepresentation;
import jnn.functions.nlp.words.features.CapitalizationWordFeatureExtractor;
import jnn.functions.nlp.words.features.CharSequenceExtractor;
import jnn.functions.nlp.words.features.DoubleFeatureExtractor;
import jnn.functions.nlp.words.features.FeatureExtractor;
import jnn.functions.nlp.words.features.LowercasedWordFeatureExtractor;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.training.GlobalParameters;
import util.IOUtils;
import vocab.Vocab;

public class LabelledTagger{	

	LabelledSpecification spec;

	Vocab tagVocab;
	Vocab wordVocab;

	WordTaggingLayer taggerLayer;
	WordRepresentationLayer wordRepLayer;
	WordWithContextRepresentation contextRepLayer;

	//for efficiency only (caching these words so that running over them every few iterations is faster)
	HashSet<String> wordsInHeldout = new HashSet<String>();

	public LabelledTagger(LabelledSpecification spec) {
		this.spec = spec;

		buildLabelTags();
		buildTagger();
	}

	public void buildLabelTags(){
		tagVocab = new Vocab();
		wordVocab = new Vocab();
		for(LabelledData dataset : spec.taggedDatasets.getTaggedDatasets()){
			for(LabelledSentence sent : dataset.getSentences()){
				for(String tag : sent.tags){
					tagVocab.addWordToVocab(tag);					
				}
				for(String token : sent.tokens){
					wordVocab.addWordToVocab(token);
				}
			}
		}

		//adding the dev and test tags to vocab
		for(LabelledData dataset : spec.validationDatasets.getTaggedDatasets()){
			for(LabelledSentence sent : dataset.getSentences()){
				for(String tag : sent.tags){
					tagVocab.addWordToVocab(tag);
				}
				for(String token : sent.tokens){
					wordsInHeldout.add(token);
				}
			}
		}
		for(LabelledData dataset : spec.testDatasets.getTaggedDatasets()){
			for(LabelledSentence sent : dataset.getSentences()){
				for(String tag : sent.tags){
					tagVocab.addWordToVocab(tag);					
				}
				for(String token : sent.tokens){
					wordsInHeldout.add(token);
				}
			}
		}
		wordVocab.sortVocabByCount();
		wordVocab.generateHuffmanCodes();
		tagVocab.sortVocabByCount();
		tagVocab.generateHuffmanCodes();
	}

	public void buildTagger(){
		WordRepresentationSetup setup = new WordRepresentationSetup(wordVocab, spec.wordProjectionDim, spec.charProjectionDim, spec.charStateDim);
		String[] wordFeatures = spec.wordFeatures.split(",");
		for(String feature : wordFeatures){
			if(feature.equals("words") || feature.equals("word")){
				if(spec.word2vecEmbeddings != null){
					setup.addFeatureExtractor(new LowercasedWordFeatureExtractor(),spec.word2vecEmbeddings);
				}
				else{
					setup.addFeatureExtractor(new LowercasedWordFeatureExtractor());
				}
			}
			else if(feature.equals("capitalization")){
				setup.addFeatureExtractor(new CapitalizationWordFeatureExtractor());
			}
			else if(feature.equals("characters")){
				setup.addSequenceExtractor(new CharSequenceExtractor());
			}
			else if(feature.startsWith("splited-sparse-")){
				final int index = Integer.parseInt(feature.replaceAll("splited-sparse-", ""));
				setup.addFeatureExtractor(new FeatureExtractor() {
					
					@Override
					public String extract(String word) {
						return word.split("/")[index];
					}
				});
			}
			else if(feature.startsWith("splited-double-")){
				final int index = Integer.parseInt(feature.replaceAll("splited-double-", ""));
				setup.addDoubleExtract(new DoubleFeatureExtractor() {
					
					@Override
					public double extract(String word) {
						return Double.parseDouble(word.split("/")[index]);
					}
				});			
			}
			else{
				throw new RuntimeException("unsupported feature " + feature + " (available features: words, characters and capitalization)");
			}
		}
		wordRepLayer = new WordRepresentationLayer(setup);

		contextRepLayer = new WordWithContextRepresentation(wordRepLayer.getOutputDim(), spec.contextStateDim);				
		if(spec.contextModel.equals("blstm")){
			contextRepLayer.setBLSTMModel();
		}
		else if(spec.contextModel.equals("window")){
			contextRepLayer.setWindowMode(5);				
		}
		else{
			throw new RuntimeException("unsupported model " + spec.contextModel + " (choose between blstm and window)");
		}

		SoftmaxObjectiveLayer labelSoftmaxLayer = new SoftmaxObjectiveLayer(tagVocab, spec.contextStateDim, "<unk>");
		this.taggerLayer = new WordTaggingLayer(wordRepLayer, contextRepLayer, labelSoftmaxLayer);
	}

	public void train(int batchSize, int threads) {
		threads = Math.min(threads, batchSize);
		long startTime = System.currentTimeMillis();
		int numberOfWords = 0;
		if(threads == 1){			
			LabelledSentence[] batchSents = spec.taggedDatasets.getNextBatch(batchSize);
			HashSet<String> words = new HashSet<String>();
			for(LabelledSentence sent : batchSents){
				numberOfWords+=sent.tokens.length;
				for(String word : sent.tokens){
					words.add(word);
				}
			}

			wordRepLayer.fillCache(words, threads, true);

			double norm = 1/(double)numberOfWords;
			//			double norm = 1;

			PosTaggerInstance trainInstance = new PosTaggerInstance(0, batchSents, taggerLayer, norm);

			trainInstance.train();

			//			trainInstance.commit(learningRate);

			for(WordBasedEvalMetric trainMetric : spec.wordErrorMetricsTrain){
				trainMetric.addSentenceScore(trainInstance.getSentences(), trainInstance.getHypothesis());
			}
		}
		else{
			LabelledSentence[][] batchesPerThread = spec.taggedDatasets.readBatchesEquivalent(batchSize/threads, threads); 

			numberOfWords = 0;
			HashSet<String> words = new HashSet<String>();
			for(int t = 0; t < threads; t++){
				for(LabelledSentence sent : batchesPerThread[t]){
					numberOfWords+=sent.tokens.length;
					for(String word : sent.tokens){
						words.add(word);
					}
				}
			}
			wordRepLayer.fillCache(words, threads, true);
			double norm = 1/(double)numberOfWords;
			//			double norm = 1;

			PosTaggerInstance[] trainInstances = new PosTaggerInstance[threads];
			Thread[] trainThreads = new Thread[threads];

			for(int t = 0; t < threads; t++){
				final int tFinal = t;
				trainInstances[t] = new PosTaggerInstance(t, batchesPerThread[t], taggerLayer, norm); 
				trainThreads[t] = new Thread(){
					@Override
					public void run() {
						trainInstances[tFinal].train();
					}
				};
				trainThreads[t].start();
			}
			for(int t = 0; t < threads; t++){
				try {
					trainThreads[t].join();
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				}
				for(WordBasedEvalMetric trainMetric : spec.wordErrorMetricsTrain){
					trainMetric.addSentenceScore(trainInstances[t].getSentences(), trainInstances[t].getHypothesis());
				}
			}
		}
		long computationTime = System.currentTimeMillis() - startTime;
		System.err.println("number of words per second (train) = " + (numberOfWords/(double)computationTime) + "k");
		taggerLayer.updateWeights(0, 0);
		//contextRepLayer.sequenceRNN.printCommitTimeAndReset();
		//System.err.println(wordRepLayer.sequenceEncoderLayer[0]);
	}

	public void validate(int batchSize, int maxThreads) {
		if(spec.validationDatasets.numberOfSamples > 0){
			int threads = Math.min(maxThreads, batchSize);			

			HashMap<String, Double> wordErrors = new HashMap<String, Double>();
			HashMap<String, Double> tagErrors = new HashMap<String, Double>();
			HashMap<String, Double> wordNorm = new HashMap<String, Double>();
			HashMap<String, Double> tagNorm = new HashMap<String, Double>();
			ArrayList<LabelledSentence> validationSentences = new ArrayList<LabelledSentence>();
			ArrayList<LabelledSentence> testSentences = new ArrayList<LabelledSentence>();
			ArrayList<LabelledSentence> validationHypothesis = new ArrayList<LabelledSentence>();
			ArrayList<LabelledSentence> testHypothesis = new ArrayList<LabelledSentence>();

			long startTime = System.currentTimeMillis();
			wordRepLayer.fillCache(wordsInHeldout, threads, false);
			if(threads == 1){
				for(int i = 0; i < spec.validationDatasets.numberOfSamples; i++){
					LabelledSentence[] batchSents = spec.validationDatasets.getNextBatch(1);		

					double norm = 1;
					PosTaggerInstance devInstance = new PosTaggerInstance(0, batchSents, taggerLayer, norm); 
					devInstance.test();

					for(WordBasedEvalMetric devMetric : spec.wordErrorMetricsValidation){
						devMetric.addSentenceScore(devInstance.getSentences(), devInstance.getHypothesis());
					}
					validationHypothesis.addAll(devInstance.getHypothesis());
					validationSentences.addAll(devInstance.getSentences());
				}

				for(int i = 0; i < spec.testDatasets.numberOfSamples; i++){
					LabelledSentence[] batchSents = spec.testDatasets.getNextBatch(1);		

					double norm = 1;
					PosTaggerInstance testInstance = new PosTaggerInstance(0, batchSents, taggerLayer, norm);
					testInstance.test();

					for(WordBasedEvalMetric testMetric : spec.wordErrorMetricsTest){
						testMetric.addSentenceScore(testInstance.getSentences(), testInstance.getHypothesis());										
					}

					testHypothesis.addAll(testInstance.getHypothesis());
					testSentences.addAll(testInstance.getSentences());
				}		
			}
			else{
				int numberOfSamplesLeftDev = spec.validationDatasets.numberOfSamples;
				while(numberOfSamplesLeftDev > 0){
					int numberOfSamples = Math.min(batchSize, numberOfSamplesLeftDev);
					threads =  Math.min(maxThreads, numberOfSamples);
					LabelledSentence[][] batchesPerThread = spec.validationDatasets.readBatchesEquivalent(numberOfSamples/threads, threads); 
					double norm = 1;
					PosTaggerInstance[] devInstances = new PosTaggerInstance[threads];
					Thread[] devThreads = new Thread[threads];

					for(int t = 0; t < threads; t++){
						final int tFinal = t;
						devInstances[t] = new PosTaggerInstance(t, batchesPerThread[t], taggerLayer, norm);
						devThreads[t] = new Thread(){
							@Override
							public void run() {
								devInstances[tFinal].test();
							}
						};
						devThreads[t].start();
					}
					for(int t = 0; t < threads; t++){
						try {
							devThreads[t].join();
						} catch (InterruptedException e) {
							throw new RuntimeException(e);
						}
						for(WordBasedEvalMetric devMetric : spec.wordErrorMetricsValidation){
							devMetric.addSentenceScore(devInstances[t].getSentences(), devInstances[t].getHypothesis());							
						}

						validationHypothesis.addAll(devInstances[t].getHypothesis());
						validationSentences.addAll(devInstances[t].getSentences());
					}
					numberOfSamplesLeftDev -= batchSize;
				}

				int numberOfSamplesLeftTest = spec.testDatasets.numberOfSamples;

				while(numberOfSamplesLeftTest > 0){
					int numberOfSamples = Math.min(batchSize, numberOfSamplesLeftTest);
					threads =  Math.min(maxThreads, numberOfSamples);

					LabelledSentence[][] batchesPerThread = spec.testDatasets.readBatchesEquivalent(numberOfSamples/threads, threads); 

					double norm = 1;
					PosTaggerInstance[] testInstances = new PosTaggerInstance[threads];
					Thread[] testThreads = new Thread[threads];

					for(int t = 0; t < threads; t++){
						final int tFinal = t;
						testInstances[t] = new PosTaggerInstance(t, batchesPerThread[t], taggerLayer, norm);
						testThreads[t] = new Thread(){
							@Override
							public void run() {
								testInstances[tFinal].test();
							}
						};
						testThreads[t].start();
					}
					for(int t = 0; t < threads; t++){
						try {
							testThreads[t].join();
						} catch (InterruptedException e) {
							throw new RuntimeException(e);
						}
						for(WordBasedEvalMetric testMetric : spec.wordErrorMetricsTest){
							testMetric.addSentenceScore(testInstances[t].getSentences(), testInstances[t].getHypothesis());							
						}

						testHypothesis.addAll(testInstances[t].getHypothesis());
						testSentences.addAll(testInstances[t].getSentences());

					}
					numberOfSamplesLeftTest -= batchSize;

				}
			}

			wordRepLayer.emptyCache();
			long computationTime = System.currentTimeMillis() - startTime;
			long numberOfWords = 0;
			for(LabelledSentence sent : testSentences) numberOfWords+=sent.tokens.length;
			for(LabelledSentence sent : validationSentences) numberOfWords+=sent.tokens.length;

			for(WordBasedEvalMetric trainMetric : spec.wordErrorMetricsTrain){
				trainMetric.commit();				
			}
			for(WordBasedEvalMetric devMetric : spec.wordErrorMetricsValidation){
				devMetric.commit();
			}
			for(WordBasedEvalMetric testMetric : spec.wordErrorMetricsTest){
				testMetric.commit();
			}

			for(WordBasedEvalMetric trainMetric : spec.wordErrorMetricsTrain){
				trainMetric.printShort(System.err);
			}
			for(WordBasedEvalMetric devMetric : spec.wordErrorMetricsValidation){
				devMetric.printShort(System.err);
			}
			
			Iterator<WordBasedEvalMetric> devIterator = spec.wordErrorMetricsValidation.iterator();
			for(WordBasedEvalMetric testMetric : spec.wordErrorMetricsTest){
				testMetric.printShort(System.err);
				System.err.println("best test score (highest on dev)="+testMetric.getScoreAtIteration(devIterator.next().getBestIteration()));
			}
			
			if(spec.wordErrorMetricsValidation.getFirst().isBestIteration()){
				IOUtils.mkdir(spec.outputDir);
				PrintStream devOut = IOUtils.getPrintStream(spec.outputDir+"/dev.hyp.gz");
				PrintStream testOut = IOUtils.getPrintStream(spec.outputDir+"/test.hyp.gz");
				for(LabelledSentence sent : validationHypothesis){
					devOut.println(sent);
				}
				for(LabelledSentence sent : testHypothesis){
					testOut.println(sent);
				}
				devOut.close();
				testOut.close();
			}

			System.err.println("number of words per second (test+val) = " + (numberOfWords/(double)computationTime) + "k");

			//			if(wordErrorMetricsValidation.getFirst().isBestIteration()){
			//				System.err.println("best dev score - printing results...");
			//				PrintStream validationOut = IOUtils.getPrintStream(dir + "validation.output");
			//				printOutputPredictions(generalFramework, validationSentences, validationHypothesis, validationOut);
			//				PrintStream testOut = IOUtils.getPrintStream(dir + "test.output");
			//				printOutputPredictions(generalFramework, testSentences, testHypothesis, testOut);
			//				for(WordBasedEvalMetric devMetric : wordErrorMetricsValidation){
			//					devMetric.print(validationOut);
			//				}
			//				for(WordBasedEvalMetric testMetric : wordErrorMetricsTest){
			//					testMetric.print(testOut);
			//				}
			//				validationOut.close();
			//				testOut.close();
			//			}
		}
	}

	public static void main(String[] args){
		Options options = new Options();
		options.addOption("train_file", true, "training file");
		options.addOption("validation_file", true, "validation file");
		options.addOption("test_file", true, "test file");
		options.addOption("lr", true, "learning rate");
		options.addOption("batch_size", true, "batch size");
		options.addOption("iterations", true, "iterations");
		options.addOption("validation_interval", true, "batches till validation");
		options.addOption("threads", true, "number of threads");
		options.addOption("word2vec_embeddings", true, "word2vec embeddings");
		options.addOption("input_format", true, "format of input (parallel, conll)");
		options.addOption("word_features", true, "features separated by commas (e.g. words,capitalization,characters)");
		options.addOption("context_model", true, "model used for encoding context (either blstm or window)");
		options.addOption("word_dim", true, "word dimension per feature set");
		options.addOption("output_dir", true, "output directory");

		if(args.length == 0){
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "java -jar [this program]", options );
			System.exit(0);
		}

		CommandLineParser parser = new BasicParser();
		CommandLine cmd;
		try {
			cmd = parser.parse( options, args);
		} catch (ParseException e) {
			throw new RuntimeException(e);
		}

		GlobalParameters.useMomentumDefault = true;
		GlobalParameters.learningRateDefault = Double.parseDouble(cmd.getOptionValue("lr"));
		GlobalParameters.l2regularizerLambdaDefault = 0;
		
		int batchSize = Integer.parseInt(cmd.getOptionValue("batch_size"));
		int threads = Integer.parseInt(cmd.getOptionValue("threads"));
		int validationInterval = Integer.parseInt(cmd.getOptionValue("validation_interval"));
		String train = cmd.getOptionValue("train_file");
		String validation = cmd.getOptionValue("validation_file");
		String test = cmd.getOptionValue("test_file");
		String outputDir = cmd.getOptionValue("output_dir");
		
		String inputFormat = "splited";		
		if(cmd.getOptionValue("input_format")!=null){
			inputFormat = cmd.getOptionValue("input_format"); 
		}

		LabelledSpecification spec = new LabelledSpecification();		
		spec.addDatasetTrain(train, inputFormat);
		spec.addDatasetValidation(validation, inputFormat);
		spec.addDatasetTest(test, inputFormat);
		spec.word2vecEmbeddings = cmd.getOptionValue("word2vec_embeddings");
		spec.wordFeatures = cmd.getOptionValue("word_features");
		spec.contextModel = cmd.getOptionValue("context_model");
		if(cmd.getOptionValue("word_dim") != null){
			spec.wordProjectionDim = Integer.parseInt(cmd.getOptionValue("word_dim"));
		}

		spec.addMetricTrain(new WordAccuracyMetric("train accuracy"));
		spec.addMetricValidation(new WordAccuracyMetric("dev accuracy"));
		spec.addMetricTest(new WordAccuracyMetric("test accuracy"));
		spec.outputDir = outputDir;
		
		int iterations = Integer.parseInt(cmd.getOptionValue("iterations"));
		
		LabelledTagger tagger = new LabelledTagger(spec);
		for(int i = 0; i < iterations; i++){
			tagger.train(batchSize, threads);
			if(i != 0 && i % validationInterval == 0){
				tagger.validate(batchSize, threads);
			}
		}
	}
}
