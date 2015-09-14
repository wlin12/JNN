package jnn.functions.composite.lstm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;

import util.IOUtils;
import util.TopNList;
import jnn.decoder.DecoderInterface;
import jnn.decoder.stackbased.StackBasedDecoder;
import jnn.decoder.state.DecoderState;
import jnn.functions.composite.lstm.aux.LSTMBlock;
import jnn.functions.composite.lstm.aux.LSTMInputNeurons;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.composite.lstm.aux.LSTMOutputNeurons;
import jnn.functions.composite.lstm.aux.LSTMParameters;
import jnn.functions.composite.lstm.aux.LSTMStateTransform;
import jnn.functions.nlp.words.WordFromCharacterSoftmax.SequenceWordState;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.nonparametrized.SoftmaxLayer;
import jnn.functions.nonparametrized.SumLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.nonparametrized.WeightedSumLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.HadamardProductLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GraphInference;

public class LSTMDecoderWithAlignment extends Layer implements LSTMStateTransform{
	LSTMParameters parameters;
	DenseFullyConnectedLayer stateToAlignment;
	DenseFullyConnectedLayer sourceToAlignment;	
	DenseFullyConnectedLayer alignmentToScore;

	private static final String STATES = "states";
	private static final String ALIGNMENTS = "alignments";

	int inputDim;
	int stateDim;
	int sourceDim;
	int alignmentDim;

	private LSTMDecoderWithAlignment() {
	}

	public LSTMDecoderWithAlignment(int inputDim, int sourceDim, int alignmentDim, int stateDim) {

		this.inputDim = inputDim;
		this.stateDim = stateDim;
		this.sourceDim = sourceDim;
		this.alignmentDim = alignmentDim;
		parameters = new LSTMParameters(inputDim+sourceDim, stateDim);

		stateToAlignment = new DenseFullyConnectedLayer(stateDim, alignmentDim);
		sourceToAlignment = new DenseFullyConnectedLayer(sourceDim, alignmentDim);
		alignmentToScore = new DenseFullyConnectedLayer(alignmentDim, 1);
		alignmentToScore.useBias = false; //softmax does not need this

	}

	public LSTMBlock[] getForwardBlocks(int start, DenseNeuronArray[] input, DenseNeuronArray[] source, DenseNeuronArray[] alignmentSources, LSTMDecoderWithAlignmentState[] states, DenseNeuronArray[] nextFertilities, GraphInference inference){
		// add input units
		inference.addNeurons(0, input);
		inference.addNeurons(0, states[0].lstmState);
		inference.addNeurons(0, states[0].lstmCell);

		int startLevel = 3;

		LSTMBlock currentBlock = new LSTMBlock(states[0].lstmState, states[0].lstmCell, startLevel);
		LSTMBlock[] blocks = new LSTMBlock[input.length];

		for(int i = 0; i < input.length; i++){
			states[i].lstmState = currentBlock.hprevState;
			states[i].lstmCell = currentBlock.cprevMem;
			nextFertilities[i] = new DenseNeuronArray(source.length);
			buildForwardBlock(start+i, source, alignmentSources, states[i], currentBlock, nextFertilities[i], inference);			
			blocks[i] = currentBlock;
			currentBlock = currentBlock.nextState();			
		}

		return blocks;
	}

	public DenseNeuronArray[] buildSourceAlignments(DenseNeuronArray[] source, GraphInference inference){
		DenseNeuronArray[] alignmentSources = DenseNeuronArray.asArray(source.length, sourceToAlignment.getOutputDim(), "alignment source");
		inference.addNeurons(0, source);
		inference.addNeurons(alignmentSources);
		inference.addMapping(new OutputMappingDenseArrayToDenseArray(source, alignmentSources, sourceToAlignment));
		return alignmentSources;
	}

	public void buildForwardBlock(int targetPos, DenseNeuronArray[] source, DenseNeuronArray[] alignmentSources, LSTMDecoderWithAlignmentState state, LSTMBlock block, DenseNeuronArray nextFertility, GraphInference inference){
		int start = block.start;
		DenseNeuronArray alignmentState = new DenseNeuronArray(stateToAlignment.getOutputDim());
		alignmentState.setName("alignment state");
		inference.addNeurons(start, alignmentState);
		inference.addMapping(new OutputMappingDenseToDense(block.hprevState, alignmentState, stateToAlignment));
		start++;

		DenseNeuronArray[] alignment = DenseNeuronArray.asArray(alignmentSources.length, stateToAlignment.getOutputDim(), "alignment");		
		inference.addNeurons(start, alignment);
		start++;
		DenseNeuronArray[] alignmentTan = DenseNeuronArray.asArray(alignmentSources.length, stateToAlignment.getOutputDim(), "alignment tanh");		
		inference.addNeurons(start, alignmentTan);
		start++;
		state.alignments = new DenseNeuronArray(source.length);
		state.alignments.setName("alignment scores");
		inference.addNeurons(start, state.alignments);
		start++;

		//		DenseNeuronArray posNeurons = new DenseNeuronArray(source.length);
		//		inference.addNeurons(0, posNeurons);
		//		posNeurons.init();
		//		for(int i = 0; i < source.length; i++){
		//			posNeurons.addNeuron(i, i+1);
		//		}
		//		DenseNeuronArray weightsPos = new DenseNeuronArray(source.length);
		//		inference.addNeurons(start, weightsPos);
		//		if(prevState != null){
		//			inference.addMapping(new OutputMappingDenseArrayToDense(new DenseNeuronArray[]{posNeurons, prevState.alignmentsSoftmax}, weightsPos, HadamardProductLayer.singleton));
		//			start++;
		//		}
		//
		//		DenseNeuronArray relPos = new DenseNeuronArray(2);
		//		inference.addNeurons(start, relPos);
		//		inference.addMapping(new OutputMappingDenseToDense(weightsPos, relPos, SumLayer.singleton));
		//		start++;

		for(int a = 0; a < alignment.length; a++){
			DenseNeuronArray pos = new DenseNeuronArray(2);
			pos.init();
			pos.addNeuron(0, targetPos+1);
			pos.addNeuron(1, a+1);
			//			pos.addNeuron(1, a-targetPos);
			inference.addNeurons(0, pos);

			//			inference.addMapping(new OutputMappingDenseToDense(relPos, alignment[a], relPosToAlignment));
			inference.addMapping(new OutputMappingDenseToDense(alignmentSources[a], alignment[a], CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(alignmentState, alignment[a], CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(alignment[a], alignmentTan[a], TanSigmoidLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0, alignmentTan[a].size-1, a, a, alignmentTan[a], state.alignments, alignmentToScore));

		}		
		start++;

		state.alignmentsSoftmax = new DenseNeuronArray(alignmentSources.length);
		state.alignmentsSoftmax.setName("alignment softmax");

		inference.addNeurons(start, state.alignmentsSoftmax);
		inference.addMapping(new OutputMappingDenseToDense(state.alignments, state.alignmentsSoftmax, SoftmaxLayer.singleton));
		start++;

		DenseNeuronArray inputWithAlignment = new DenseNeuronArray(inputDim+sourceDim);
		inputWithAlignment.setName("lstm block input with alignment");
		inference.addNeurons(start, inputWithAlignment);
		start++;		
		DenseNeuronArray[] sourcesWithWeight = new DenseNeuronArray[source.length+1];
		for(int a = 0; a < source.length; a++){
			sourcesWithWeight[a] = source[a];
		}
		sourcesWithWeight[source.length] = state.alignmentsSoftmax;
		inference.addMapping(new OutputMappingDenseArrayToDense(0,sourceDim-1,0,sourceDim-1,sourcesWithWeight, inputWithAlignment, WeightedSumLayer.singleton));
		inference.addMapping(new OutputMappingDenseToDense(0,inputDim-1,sourceDim, sourceDim+inputDim-1,state.input, inputWithAlignment, CopyLayer.singleton));		

		block.start = start;
		block.addToInference(inference, inputWithAlignment, parameters);		
	}

	public void buildInference(DenseNeuronArray[] input, DenseNeuronArray[] source, DenseNeuronArray initialState, DenseNeuronArray initialCell, Mapping map){
		GraphInference inference = map.getSubInference();
		DenseNeuronArray[] sourceAlignment = buildSourceAlignments(source, inference);
		DenseNeuronArray[] nextFertilities = new DenseNeuronArray[input.length];
		LSTMDecoderWithAlignmentState[] states = new LSTMDecoderWithAlignmentState[input.length];
		for(int i = 0; i < states.length; i++){
			states[i] = new LSTMDecoderWithAlignmentState(-1, false, null, null, null, input[i]);
			if(i > 0){
				states[i].prevState = states[i-1];
			}
		}
		states[0].lstmState = initialState;
		states[0].lstmCell = initialCell;

		LSTMBlock[] blocksForward = getForwardBlocks(0, input, source, sourceAlignment,states,nextFertilities, inference);

		map.setForwardParam(ALIGNMENTS, states);
		map.setForwardParam(STATES, blocksForward);
	}

	@Override
	public void forward(LSTMMapping map) {
		GraphInference inference = map.getSubInference();

		DenseNeuronArray[] x = LSTMDecoderState.getInputs(map.states);
		buildInference(x, map.sources, map.initialState, map.initialCell, map);
		inference.init();
		inference.forward();

		DenseNeuronArray[] lstmStates = LSTMDecoderState.getStates(map.states);
		DenseNeuronArray[] lstmCells = LSTMDecoderState.getCells(map.states);

		LSTMBlock[] blocks = (LSTMBlock[])map.getForwardParam(STATES);
		for(int i = 0; i < lstmStates.length; i++){
			DenseNeuronArray outputI = lstmStates[i];
			DenseNeuronArray cellI = lstmCells[i];
			for(int d = 0; d < stateDim; d++){				
				outputI.addNeuron(d, blocks[i].hState.getNeuron(d));
				cellI.addNeuron(d, blocks[i].cMen.getNeuron(d));
			}
		}

		//		for(int i = 0; i < getAlignments(map).length; i++){
		//			System.err.println(getAlignments(map)[i].fertilitySig);
		//		}
	}

	@Override
	public void backward(LSTMMapping map) {
		LSTMBlock[] blocks = (LSTMBlock[])map.getForwardParam(STATES);

		DenseNeuronArray[] lstmStates = LSTMDecoderState.getStates(map.states);
		DenseNeuronArray[] lstmCells = LSTMDecoderState.getCells(map.states);

		GraphInference inference = map.getSubInference();
		for(int i = 0; i < lstmStates.length; i++){
			DenseNeuronArray outputI = lstmStates[i];
			DenseNeuronArray cellI = lstmCells[i];
			for(int d = 0; d < stateDim; d++){	
				blocks[i].hState.addError(d, outputI.getError(d));
				blocks[i].cMen.addError(d, cellI.getError(d));
			}
		}
		inference.backward();
	}

	public LSTMDecoderWithAlignmentState decode(LSTMDecoderWithAlignmentState initialDecoderState, DenseNeuronArray[] source, int beam, DecoderInterface scorer){

		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray[] alignmentSources = buildSourceAlignments(source, inference);
		inference.init();
		inference.forward();

		StackBasedDecoder decoder = new StackBasedDecoder(new DecoderInterface() {

			@Override
			public List<DecoderState> expand(DecoderState state) {
				LSTMDecoderWithAlignmentState lstmState = (LSTMDecoderWithAlignmentState) state;
				GraphInference inference = new GraphInference(0, false);
				inference.addNeurons(0,alignmentSources);
				inference.addNeurons(0,source);
				lstmState.alignments = new DenseNeuronArray(source.length);
				DenseNeuronArray[] nextFertility = new DenseNeuronArray[1];
				LSTMBlock block = getForwardBlocks(state.numberOfPrevStates,new DenseNeuronArray[]{lstmState.input}, source,alignmentSources, new LSTMDecoderWithAlignmentState[]{lstmState},nextFertility, inference)[0];
				inference.init();
				inference.forward();
				lstmState.output = block.hState;

				List<DecoderState> states = scorer.expand(state);
				for(DecoderState nextState : states){
					LSTMDecoderWithAlignmentState nextLstmState = (LSTMDecoderWithAlignmentState) nextState;
					nextLstmState.lstmState = block.hState;
					nextLstmState.lstmCell = block.cMen;
				}
				return states;
			}
		}, initialDecoderState);
		decoder.stackSize = beam;
		decoder.decode();
		return (LSTMDecoderWithAlignmentState)decoder.getBestState();
	}
	
	public TopNList<LSTMDecoderWithAlignmentState> decode(LSTMDecoderWithAlignmentState initialDecoderState, DenseNeuronArray[] source, int beam, int topN, DecoderInterface scorer){

		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray[] alignmentSources = buildSourceAlignments(source, inference);
		inference.init();
		inference.forward();

		StackBasedDecoder decoder = new StackBasedDecoder(new DecoderInterface() {

			@Override
			public List<DecoderState> expand(DecoderState state) {
				LSTMDecoderWithAlignmentState lstmState = (LSTMDecoderWithAlignmentState) state;
				GraphInference inference = new GraphInference(0, false);
				inference.addNeurons(0,alignmentSources);
				inference.addNeurons(0,source);
				lstmState.alignments = new DenseNeuronArray(source.length);
				DenseNeuronArray[] nextFertility = new DenseNeuronArray[1];
				LSTMBlock block = getForwardBlocks(state.numberOfPrevStates,new DenseNeuronArray[]{lstmState.input}, source,alignmentSources, new LSTMDecoderWithAlignmentState[]{lstmState},nextFertility, inference)[0];
				inference.init();
				inference.forward();
				lstmState.output = block.hState;

				List<DecoderState> states = scorer.expand(state);
				for(DecoderState nextState : states){					
					LSTMDecoderWithAlignmentState nextLstmState = (LSTMDecoderWithAlignmentState) nextState;
					nextLstmState.lstmState = block.hState;
					nextLstmState.lstmCell = block.cMen;
				}
				return states;
			}
		}, initialDecoderState);
		decoder.stackSize = beam;
		decoder.decode();
		LinkedList<DecoderState> finalStates = decoder.getBestStates(topN);
		TopNList<LSTMDecoderWithAlignmentState> ret = new TopNList<LSTMDecoderWithAlignmentState>(topN);			

		if(finalStates.size() == 0){
			return ret;
		}
		for(int k = 0; k < finalStates.size(); k++){
			LSTMDecoderWithAlignmentState state = (LSTMDecoderWithAlignmentState)finalStates.get(k);
			ret.add(state, state.score);
		}
		return ret;
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		parameters.update(learningRate, momentum);
		stateToAlignment.updateWeights(learningRate, momentum);
		sourceToAlignment.updateWeights(learningRate, momentum);
		alignmentToScore.updateWeights(learningRate, momentum);
		//alignmentToScore.normalizeWeights();

	}

	public int getStateDim() {
		return stateDim;
	}

	public void save(PrintStream out){
		out.println(inputDim);
		out.println(stateDim);
		out.println(sourceDim);
		out.println(alignmentDim);

		parameters.save(out);
		stateToAlignment.save(out);
		sourceToAlignment.save(out);
		alignmentToScore.save(out);
	}

	public static LSTMDecoderWithAlignment load(BufferedReader in){
		try{
			LSTMDecoderWithAlignment decoder = new LSTMDecoderWithAlignment();
			decoder.inputDim = Integer.parseInt(in.readLine());
			decoder.stateDim = Integer.parseInt(in.readLine());
			decoder.sourceDim = Integer.parseInt(in.readLine());
			decoder.alignmentDim = Integer.parseInt(in.readLine());
			decoder.parameters = LSTMParameters.load(in);
			decoder.stateToAlignment = DenseFullyConnectedLayer.load(in);
			decoder.sourceToAlignment = DenseFullyConnectedLayer.load(in);
			decoder.alignmentToScore = DenseFullyConnectedLayer.load(in);
			return decoder;
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}

	public LSTMDecoderWithAlignmentState[] getAlignments(Mapping mapping){
		return (LSTMDecoderWithAlignmentState[])mapping.getForwardParam(ALIGNMENTS);
	}
}
