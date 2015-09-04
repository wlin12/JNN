package jnn.functions.nlp.words.features;

public class SuffixWordFeatureExtractor implements FeatureExtractor{

	int size;
	public SuffixWordFeatureExtractor(int size) {
		this.size = size;
	}
	
	@Override
	public String extract(String word) {
		if(word.length() <= size) return word;
		return word.toLowerCase().substring(0, size);
	}
}
