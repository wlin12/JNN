package jnn.neuron;

import jnn.training.GlobalParameters;

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
		if(name != null){
			ret = name + "(" + size + ")" + "\n";
		}
		else{
			ret = "unamed" + "(" + size + ")" + "\n";
		}
		ret += outputs;
		ret += error;
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

	public String getName() {
		return name;
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
	
	public void addNeuron(DenseNeuronArray neurons, int startSource, int startTarget){
		for(int i = 0; i < size; i++){
			outputs.putScalar(i+startTarget, outputs.getDouble(i+startTarget) + neurons.getNeuron(i+startSource));			
		}
	}
	
	public void addNeuron(DenseNeuronArray neurons, int startSource, int startTarget, int len){
		for(int i = 0; i < len; i++){
			outputs.putScalar(i+startTarget, outputs.getDouble(i+startTarget) + neurons.getNeuron(i+startSource));			
		}
	}
	
	public double getError(int index){
		return error.getDouble(index);
	}

	public void addError(DenseNeuronArray neurons, int startSource, int startTarget){
		for(int i = 0; i < size; i++){
			error.putScalar(i + startTarget, error.getDouble(i + startTarget) + neurons.getError(i + startSource));
		}
	}
	
	public void addError(DenseNeuronArray neurons, int startSource, int startTarget, int len){
		for(int i = 0; i < len; i++){
			error.putScalar(i + startTarget, error.getDouble(i + startTarget) + neurons.getError(i + startSource));
		}
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
			INDArrayUtils.addi(outputs, vals);
		}
		else{
			for(int i = start; i <= end; i++){
				outputs.putScalar(i, outputs.getDouble(i) + vals.getDouble(i-start));
			}
		}
	}
	
	public void setOutputRangeAfterMmul(int start, int end, INDArray x, INDArray y){
		if(start == 0 && end == size-1){
			x.mmuli(y, outputs);
		}
		else{
			INDArray vals = x.mmul(y);
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
			INDArrayUtils.addi(error, vals);
		}
		else{
			for(int i = start; i <= end; i++){
				error.putScalar(i, error.getDouble(i) + vals.getDouble(i-start));
			}
		}
	}
	
	public void setErrorRangeAfterMmul(int start, int end, INDArray x, INDArray y){
		if(start == 0 && end == size-1){
			x.mmuli(y, error);
		}
		else{
			INDArray vals = x.mmul(y);
			for(int i = start; i <= end; i++){
				error.putScalar(i, error.getDouble(i) + vals.getDouble(i-start));
			}
		}
	}
	
	@Override
	public void beforeBackward() {
		INDArrayUtils.capValues(error, -GlobalParameters.maxError, GlobalParameters.maxError);
	}

	@Override
	public void capValues() {		
	}

	public static DenseNeuronArray[] asArray(int length, int letterProjectionDim) {
		DenseNeuronArray[] ret = new DenseNeuronArray[length];
		for(int i = 0; i < length; i++){
			ret[i] = new DenseNeuronArray(letterProjectionDim);
		}
		return ret;
	}	
	
	public static DenseNeuronArray[] asArray(int length, int letterProjectionDim, String name) {
		DenseNeuronArray[] ret = new DenseNeuronArray[length];
		for(int i = 0; i < length; i++){
			ret[i] = new DenseNeuronArray(letterProjectionDim);
			ret[i].setName(name + "[" + i + "]");
		}
		return ret;
	}

	public void checkForNaN() {
		for(int i = 0 ; i < size; i++){
			if(Double.isNaN(outputs.getDouble(i))){
				throw new RuntimeException("found output nan in index " + i);
			}
			if(Double.isNaN(error.getDouble(i))){
				throw new RuntimeException("found error nan in index " + i);
			}
		}
	}

}

