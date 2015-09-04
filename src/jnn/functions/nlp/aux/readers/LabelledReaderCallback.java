package jnn.functions.nlp.aux.readers;

import jnn.functions.nlp.aux.input.LabelledSentence;

public interface LabelledReaderCallback {
	public void cb(LabelledSentence sent);
}
