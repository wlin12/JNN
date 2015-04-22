package jnn.decoder;

import java.util.List;

import jnn.decoder.state.DecoderState;

public interface DecoderInterface {
	public List<DecoderState> expand(DecoderState state);
}
