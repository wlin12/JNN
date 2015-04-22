package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingDenseToString;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxDenseObjective;
import jnn.training.GraphInference;
import util.TopNList;
import vocab.Vocab;

abstract public class AbstractSofmaxObjectiveLayer extends Layer implements DenseToStringTransform, DenseArrayToStringArrayTransform{
	
	abstract public String decode(DenseNeuronArray input);

	abstract public void save(PrintStream out);

	abstract public TopNList<String> getTopN(DenseNeuronArray input, int n);

}
