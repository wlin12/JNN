package jnn.functions.nlp.words.features;

public class LowercasedWordFeatureExtractor implements FeatureExtractor{
	@Override
	public String extract(String word) {
		return word.toLowerCase();
	}
}
