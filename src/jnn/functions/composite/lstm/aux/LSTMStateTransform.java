package jnn.functions.composite.lstm.aux;

import jnn.functions.composite.lstm.LSTMDecoderState;

public interface LSTMStateTransform {
	public void forward(LSTMMapping mapping);
	public void backward(LSTMMapping mapping);
}
