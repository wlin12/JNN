package jnn.functions.nlp.app.pos;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import jnn.functions.composite.SoftmaxObjectiveLayer;
import jnn.functions.nlp.app.pretraining.StructuredSkipngram;
import jnn.functions.nlp.app.pretraining.StructuredSkipngramSpecification;
import jnn.functions.nlp.aux.input.LabelledData;
import jnn.functions.nlp.aux.input.LabelledSentence;
import jnn.functions.nlp.aux.metrics.WordAccuracyMetric;
import jnn.functions.nlp.aux.metrics.WordBasedEvalMetric;
import jnn.functions.nlp.labeling.WordTaggingLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nlp.words.WordWithContextRepresentation;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.training.GlobalParameters;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import util.IOUtils;
import util.MapUtils;
import vocab.Vocab;

public class PosTagger{	

	PosSpecification spec;

	Vocab tagVocab;
	Vocab wordVocab;
	HashSet<String> inVocab; 

	WordTaggingLayer taggerLayer;
	WordRepresentationLayer wordRepLayer;
	WordWithContextRepresentation contextRepLayer;

	//for efficiency only (caching these words so that running over them every few iterations is faster)
	HashSet<String> wordsInHeldout = new HashSet<String>();

	public PosTagger(PosSpecification spec) {
		this.spec = spec;

		buildPOSTags();
		buildTagger();
	}

	public void buildPOSTags(){
		tagVocab = new Vocab();
		wordVocab = new Vocab();
		inVocab = new HashSet<String>();
		for(LabelledData dataset : spec.taggedDatasets.getTaggedDatasets()){
			for(LabelledSentence sent : dataset.getSentences()){
				for(String tag : sent.tags){
					tagVocab.addWordToVocab(tag);					
				}
				for(String token : sent.tokens){
					wordVocab.addWordToVocab(token);
					inVocab.add(token);
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
		setup.sequenceSigmoid = spec.sequenceActivation;
		setup.loadFromString(spec.wordFeatures, spec.word2vecEmbeddings);
		wordRepLayer = new WordRepresentationLayer(setup);
		if(spec.skipngramModel != null){
			System.err.println("loading skipngram embeddings");
			StructuredSkipngramSpecification skipsetup = new StructuredSkipngramSpecification();;
			if(spec.skipngramModelFeatures != null){
				skipsetup.wordFeatures = spec.skipngramModelFeatures;
				wordRepLayer.addFeatures(StructuredSkipngram.loadRep(spec.skipngramModel, skipsetup));
			}
			else{
				skipsetup.wordFeatures = spec.wordFeatures;
				wordRepLayer = StructuredSkipngram.loadRep(spec.skipngramModel, skipsetup);
			}
		}
		

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

		SoftmaxObjectiveLayer labelSoftmaxLayer = new SoftmaxObjectiveLayer(tagVocab, spec.contextStateDim, "UNK");
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
			PosTaggerInstance trainInstance = new PosTaggerInstance(0, batchSents, taggerLayer, norm);
			trainInstance.train();

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
					LinkedList<int[]> order = new LinkedList<int[]>();
					LabelledSentence[][] batchesPerThread = spec.validationDatasets.readBatchesEquivalent(numberOfSamples/threads, threads, order); 
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
						
					}
					for(int[] el : order){
						validationHypothesis.add(devInstances[el[0]].getHypothesis().get(el[1]));
						validationSentences.add(devInstances[el[0]].getSentences().get(el[1]));
					}
					numberOfSamplesLeftDev -= batchSize;
				}

				int numberOfSamplesLeftTest = spec.testDatasets.numberOfSamples;

				while(numberOfSamplesLeftTest > 0){
					int numberOfSamples = Math.min(batchSize, numberOfSamplesLeftTest);
					threads =  Math.min(maxThreads, numberOfSamples);

					LinkedList<int[]> order = new LinkedList<int[]>();
					LabelledSentence[][] batchesPerThread = spec.testDatasets.readBatchesEquivalent(numberOfSamples/threads, threads, order); 

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
					}
					for(int[] el : order){
						testHypothesis.add(testInstances[el[0]].getHypothesis().get(el[1]));
						testSentences.add(testInstances[el[0]].getSentences().get(el[1]));
					}
					numberOfSamplesLeftTest -= batchSize;

				}
			}

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


			System.err.println("number of words per second (test+val) = " + (numberOfWords/(double)computationTime) + "k");

			if(spec.wordErrorMetricsValidation.getFirst().isBestIteration()){
				printOutputs(spec.outputDir + "/validation", validationHypothesis, validationSentences);
				printOutputs(spec.outputDir + "/test", testHypothesis, testSentences);
				System.err.println("best dev score - printing results...");
				PrintStream validationOut = IOUtils.getPrintStream(spec.outputDir + "/validation.output");				
				PrintStream testOut = IOUtils.getPrintStream(spec.outputDir + "/test.output");
				for(WordBasedEvalMetric devMetric : spec.wordErrorMetricsValidation){
					devMetric.print(validationOut);
				}
				devIterator = spec.wordErrorMetricsValidation.iterator();
				for(WordBasedEvalMetric testMetric : spec.wordErrorMetricsTest){
					testMetric.print(testOut);
					testOut.println("best test score (highest on dev)="+testMetric.getScoreAtIteration(devIterator.next().getBestIteration()));
				}
				validationOut.close();
				testOut.close();

				PrintStream out = IOUtils.getPrintStream(spec.outputDir + "/vectors.gz");
				wordRepLayer.printVectors(wordVocab, 10000, out);

			}
		}
		wordRepLayer.emptyCache();
	}
	
	public void printOutputs(String prefix, ArrayList<LabelledSentence> hyp, ArrayList<LabelledSentence> ref){
		// pos tagged 
		PrintStream posOut = IOUtils.getPrintStream(prefix+".pos");
		for(LabelledSentence sent : hyp){
			posOut.println(sent.toStanford());
		}
		posOut.close();
		
		// stats
		PrintStream statsOut = IOUtils.getPrintStream(prefix+".stats");
		double oovCorrect = 0;
		double oov = 0;
		HashMap<String, Integer> correctWords = new HashMap<String, Integer>();
		HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
		Iterator<LabelledSentence> it = ref.iterator();
		HashMap<String, LabelledSentence> errorToHyp = new HashMap<String, LabelledSentence>(); 
		HashMap<String, LabelledSentence> errorToRef = new HashMap<String, LabelledSentence>(); 
		for(LabelledSentence sent : hyp){
			posOut.println(sent);
			LabelledSentence ret = it.next();
			for(int i = 0; i < sent.tokens.length; i++){
				String word = sent.tokens[i].toLowerCase();
				if(!inVocab.contains(word)){
					oov++;
					if(sent.tags[i].equals(ret.tags[i])) {
						oovCorrect++;
					}		
				}
				MapUtils.add(wordCount, word, 1);
				if(sent.tags[i].equals(ret.tags[i])) {
					MapUtils.add(correctWords, word, 1);					
				}
				else{
					errorToHyp.put(word, sent);
					errorToRef.put(word, ret);					
				}
			}
		}		
		
		PrintStream correctOut = IOUtils.getPrintStream(prefix+".correct");		
		PrintStream wrongOut = IOUtils.getPrintStream(prefix+".incorrect");		
		for(String word : wordCount.keySet()){
			
			int correct = 0;
			if(correctWords.containsKey(word)){
				correct = correctWords.get(word);
			}
			int count = wordCount.get(word);
			if(!inVocab.contains(word)){
				if(correct == count && count == 1){
					correctOut.println(word);
				}
			}
			if(correct != count){
				wrongOut.println(word + " ||| " + (count - correct) + " " + (count - correct)*100/count + "%" + " - " + inVocab.contains(word));
				wrongOut.println("hyp: "+errorToHyp.get(word));
				wrongOut.println("ref: "+errorToRef.get(word));
			}
		}
		correctOut.close();
		wrongOut.close();
						
		statsOut.println("oov acc = " + oovCorrect/oov);
		statsOut.close();
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
		options.addOption("skip_ngram_model", true, "skipngram model to initialize the embeddings");
		options.addOption("skip_ngram_model_features", true, "skipngram model features");
		options.addOption("sequence_activation", true, "character sequence activation (0=linear, 1=logistic, 2=tanh)");
		options.addOption("update_rep", true, "whether to update the word representation(true or false)");
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

		String inputFormat = "parallel";		
		if(cmd.getOptionValue("input_format")!=null){
			inputFormat = cmd.getOptionValue("input_format"); 
		}

		PosSpecification spec = new PosSpecification();		
		spec.addDatasetTrain(train, inputFormat);
		if(validation != null){
			spec.addDatasetValidation(validation, inputFormat);
		}
		else{
			spec.addDatasetValidation(100);
		}
		spec.addDatasetTest(test, inputFormat);		
		spec.word2vecEmbeddings = cmd.getOptionValue("word2vec_embeddings");
		spec.wordFeatures = cmd.getOptionValue("word_features");
		spec.contextModel = cmd.getOptionValue("context_model");
		spec.sequenceActivation = Integer.parseInt(cmd.getOptionValue("sequence_activation"));
		if(cmd.getOptionValue("word_dim") != null){
			spec.wordProjectionDim = Integer.parseInt(cmd.getOptionValue("word_dim"));
		}

		spec.addMetricTrain(new WordAccuracyMetric("train accuracy"));
		spec.addMetricValidation(new WordAccuracyMetric("dev accuracy"));
		spec.addMetricTest(new WordAccuracyMetric("test accuracy"));

		int iterations = Integer.parseInt(cmd.getOptionValue("iterations"));		
		spec.skipngramModel = cmd.getOptionValue("skip_ngram_model");		
		spec.skipngramModelFeatures = cmd.getOptionValue("skip_ngram_model_features");		
		spec.outputDir = cmd.getOptionValue("output_dir");
		PosTagger tagger = new PosTagger(spec);


		for(int i = 0; i < iterations; i++){
			tagger.train(batchSize, threads);
			if(i != 0 && i % validationInterval == 0){
				tagger.validate(batchSize, threads);				
			}
		}
	}
}
