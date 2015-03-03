package jnn.neuron;

import jnn.features.FeatureVector;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.INDArrayUtils;
import util.MathUtils;
import util.RandomUtils;


public class DenseNeuronArray extends NeuronArray{
	private INDArray outputs;
	private INDArray error;
	String name = null;

	public DenseNeuronArray(int size) {
		super(size);
	}
	
	@Override
	public void init() {
		this.outputs = Nd4j.zeros(size);
		this.error = Nd4j.zeros(size);
	}
	
	public DenseNeuronArray copy(){
		DenseNeuronArray copy = new DenseNeuronArray(size);
		copy.setName(name);
		copy.outputs = outputs.add(0);
		copy.error = error.add(0);
		return copy;
	}

	public DenseNeuronArray copyOutput(){
		DenseNeuronArray copy = new DenseNeuronArray(size);
		copy.setName(name);
		copy.outputs = outputs.add(0);
		copy.error = Nd4j.zeros(size);
		return copy;
	}

	@Override
	public String toString() {
		String ret = "";
		String size = "-";
		if(outputs!=null){
			size = outputs.size(0) + "";
		}
		if(name != null){
			ret = name + "(" + size + ")" + "\n";
		}
		ret += outputs;
		ret += "\n" + error;
		return ret;
	}

	public void link(DenseNeuronArray in){
		this.size = in.size;
		this.outputs = in.outputs;
		this.error = in.error;
	}

	public double[] copyAsArray() {
		double[] ret = new double[size];
		for(int i = 0; i < size; i ++){
			ret[i] = outputs.getDouble(i);
		}
		return ret;
	}
	
	public double[] copyErrorAsArray() {
		double[] ret = new double[size];
		for(int i = 0; i < size; i ++){
			ret[i] = error.getDouble(i);
		}
		return ret;
	}

	
	public void setName(String name) {
		this.name = name;
	}
	
	public void randInitialize() {
		init();
		for(int i = 0; i < size;  i++){
			outputs.putScalar(i, RandomUtils.initializeRandomNumber(-1, 1, 1));
		}
	}

	public void loadFromArray(double[] array) {
		for(int i = 0; i < size;  i++){
			outputs.putScalar(i, array[i]);
		}
	}

	public void computeErrorTan(double[] expected) {		
		for(int i = 0; i < size; i++){
			error.putScalar(i, (expected[i] - outputs.getDouble(i))/(size * 2));
		}
	}	
	
	public void computeErrorTan(double[] expected, double norm) {
		for(int i = 0; i < size; i++){
			error.putScalar(i, (expected[i] - outputs.getDouble(i))/(size * 2 * norm));
		}
	}	
	
	public double sqError(){		
		double ret = 0;
		for(int i = 0; i < size; i++){
			ret+=error.getDouble(i)*error.getDouble(i);
		}
		return ret/size;
	}
	
	public boolean isOutputInitialized(){
		return outputs!=null;
	}
	
	public boolean isErrorInitialized(){
		return error!=null;
	}
	
	public int len(){
		return size;
	}
	
	public double getNeuron(int index){
		return outputs.getDouble(index);
	}
	
	public void addNeuron(int index, double val){
		outputs.putScalar(index, val + outputs.getDouble(index));
	}
	
	public double getError(int index){
		return error.getDouble(index);
	}
	
	public void addError(int index, double val){
		error.putScalar(index, val + error.getDouble(index));
	}
	
	public void addError(DenseNeuronArray anotherDenseArray){
		error.addi(anotherDenseArray.error);
	}

	public int maxIndex() {
		return MathUtils.maxIndex(outputs);
	}

	public double getMax() {
		return MathUtils.max(outputs);
	}

	public INDArray getOutputRange(int start, int end){
		if(start == 0 && end == size-1){
			return outputs;
		}
		INDArray ret = Nd4j.create(end - start + 1);
		for(int i = start; i <= end; i++){
			ret.putScalar(i-start, outputs.getDouble(i));
		}
		return ret;
	}
	
	public void setOutputRange(int start, int end, INDArray vals){
		if(start == 0 && end == size-1){
			outputs.addi(vals);
		}
		else{
			for(int i = start; i <= end; i++){
				outputs.putScalar(i, outputs.getDouble(i) + vals.getDouble(i-start));
			}
		}
	}
	
	public INDArray getErrorRange(int start, int end){
		if(start == 0 && end == size-1){
			return error;
		}
		INDArray ret = Nd4j.create(end - start + 1);
		for(int i = start; i <= end; i++){
			ret.putScalar(i-start, error.getDouble(i));
		}
		return ret;
	}
	
	public void setErrorRange(int start, int end, INDArray vals){
		if(start == 0 && end == size-1){
			error.addi(vals);		
		}
		else{
			for(int i = start; i <= end; i++){
				error.putScalar(i, error.getDouble(i) + vals.getDouble(i-start));
			}
		}
	}
	
	@Override
	public void beforeBackward() {
		INDArrayUtils.capValues(error, -FeatureVector.maxError, FeatureVector.maxError);
	}	
}

