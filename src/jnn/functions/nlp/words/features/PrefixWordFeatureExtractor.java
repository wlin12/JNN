package jnn.functions.nlp.words.features;

public class PrefixWordFeatureExtractor implements FeatureExtractor{

	int size;
	public PrefixWordFeatureExtractor(int size) {
		this.size = size;
	}
	
	@Override
	public String extract(String word) {
		if(word.length() <= size) return word;
		return word.toLowerCase().substring(word.length()-size, word.length());
	}
}
