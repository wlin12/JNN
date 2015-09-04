package jnn.functions.nlp.words.features;

import java.util.ArrayList;

import vocab.Vocab;

public class WordRepresentationSetup {
	public ArrayList<FeatureExtractor> featureExtractors = new ArrayList<FeatureExtractor>();
	public ArrayList<SequenceExtractor> sequenceExtractors = new ArrayList<SequenceExtractor>();
	public ArrayList<String> featurePreinitializationFile = new ArrayList<String>();
	public ArrayList<String> sequencePreinitializationFile = new ArrayList<String>();
	public ArrayList<DoubleFeatureExtractor> doubleFeatureExtractors = new ArrayList<DoubleFeatureExtractor>();
	
	public Vocab existingWords;
	public int projectionDim;
	public int sequenceProjectionDim;
	public int sequenceStateDim;
	public int maxOccurForDropout = 1;
	public double dropoutProbability = 0.5;
	public String sequenceType = "lstm";
	public int sequenceSigmoid = 0;
	public boolean sequenceForward = true;
	public boolean sequenceBackward = true;
	public int maxNumberOfSparseFeatures = 100000000;
	
	public WordRepresentationSetup(Vocab existingWords, int projectionDim,
			int sequenceProjectionDim, int sequenceStateDim) {
		super();
		this.existingWords = existingWords;
		this.projectionDim = projectionDim;
		this.sequenceProjectionDim = sequenceProjectionDim;
		this.sequenceStateDim = sequenceStateDim;
	}

	public void addFeatureExtractor(FeatureExtractor extractor, String preinitializationFile){
		featureExtractors.add(extractor);
		featurePreinitializationFile.add(preinitializationFile);
	}
	
	public void addFeatureExtractor(FeatureExtractor extractor){
		featureExtractors.add(extractor);
		featurePreinitializationFile.add(null);
	}
	
	public void addSequenceExtractor(SequenceExtractor extractor, String preinitializationFile){
		sequenceExtractors.add(extractor);
		sequencePreinitializationFile.add(preinitializationFile);
	}
	
	public void addSequenceExtractor(SequenceExtractor extractor){
		sequenceExtractors.add(extractor);
		sequencePreinitializationFile.add(null);
	}
	
	public void addDoubleExtract(DoubleFeatureExtractor extractor){
		doubleFeatureExtractors.add(extractor);
	}

	public void loadFromString(String wordFeatures, String word2vecEmbeddings) {
		String[] wordFeaturesArray = wordFeatures.split(",");
		for(String feature : wordFeaturesArray){
			if(feature.equals("words") || feature.equals("word")){
				if(word2vecEmbeddings != null){
					addFeatureExtractor(new LowercasedWordFeatureExtractor(),word2vecEmbeddings);
				}
				else{
					addFeatureExtractor(new LowercasedWordFeatureExtractor());
				}
			}
			else if(feature.equals("shape")){
				addFeatureExtractor(new WordShapeFeatureExtractor());
			}
			else if(feature.equals("shape-no-repeat")){
				addFeatureExtractor(new WordShapeNoRepeatFeatureExtractor());
			}
			else if(feature.startsWith("prefix")){
				int size = Integer.parseInt(feature.split("-")[1]);
				addFeatureExtractor(new PrefixWordFeatureExtractor(size));
			}
			else if(feature.startsWith("suffix")){
				int size = Integer.parseInt(feature.split("-")[1]);
				addFeatureExtractor(new SuffixWordFeatureExtractor(size));
			}
			else if(feature.equals("capitalization")){
				addFeatureExtractor(new CapitalizationWordFeatureExtractor());
			}
			else if(feature.equals("casing")){
				addFeatureExtractor(new CaseFeatureExtractor());
			}
			else if(feature.startsWith("morfessor")){
				String file = feature.split("-")[1];
				addFeatureExtractor(new MorfessorExtractorPrefix(file));
				addFeatureExtractor(new MorfessorExtractorSuffix(file));
				addFeatureExtractor(new MorfessorExtractorStem(file));
			}
			else if(feature.equals("characters")){
				addSequenceExtractor(new CharSequenceExtractor());
			}
			else if(feature.equals("characters-lowercase")){
				addSequenceExtractor(new CharSequenceExtractorLc());
			}			
			else{
				throw new RuntimeException("unsupported feature " + feature + " (available features: words, characters and capitalization)");
			}
		}
	}

	public void add(WordRepresentationSetup loadRep) {
		
		for(FeatureExtractor feats : loadRep.featureExtractors){
			featureExtractors.add(feats);
		}
		for(SequenceExtractor feats : loadRep.sequenceExtractors){
			sequenceExtractors.add(feats);
		}
		for(String feats : loadRep.featurePreinitializationFile){
			featurePreinitializationFile.add(feats);
		}
		for(String feats : loadRep.sequencePreinitializationFile){
			sequencePreinitializationFile.add(feats);
		}
		for(DoubleFeatureExtractor feats : loadRep.doubleFeatureExtractors){
			doubleFeatureExtractors.add(feats);
		}

	}
	
}
