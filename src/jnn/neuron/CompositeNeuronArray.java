package jnn.neuron;

import jnn.functions.parametrized.Layer;

abstract public class CompositeNeuronArray{
	abstract public NeuronArray[] getAtomicNeurons();
}
