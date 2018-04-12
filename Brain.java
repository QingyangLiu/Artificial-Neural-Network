import java.util.ArrayList;

public class Brain{
	double alpha;
	int numinput;
	int numoutput;
	int numhidden;
	int numhiddenlayer;
	double[][] output;
	ArrayList<double[][]> weights=new ArrayList<>();
	ArrayList<double[][]> layers=new ArrayList<>();
	
	public Brain(int numinput, int numoutput, int numhidden, int numhiddenlayer, double alpha){
		this.numinput=numinput;
		this.numoutput=numoutput;
		this.numhidden=numhidden;
		this.numhiddenlayer=numhiddenlayer;
		this.alpha=alpha;
		weights.add(new double[numinput][numhidden]);
		for (int i=1;i<numhiddenlayer;i++){
			weights.add(new double[numhidden][numhidden]);
		}
		weights.add(new double[numhidden][numoutput]);
		for (int i=0;i<weights.size();i++){
			for (int j=0;j<weights.get(i).length;j++){
				for (int k=0;k<weights.get(i)[j].length;k++){
					weights.get(i)[j][k]=Math.random()*2-1;
				}
			}
		}
		
	}
	
	public double[][] go(double[][] inputs){
		this.layers.add(inputs);
		for (int i=0;i<this.numhiddenlayer;i++){
			this.layers.add(nonlinear(matrixmultiply(layers.get(i), weights.get(i)),false));	
		}
		this.layers.add(nonlinear(matrixmultiply(layers.get(layers.size()-1), weights.get(weights.size()-1)),false));
		this.output=layers.get(layers.size()-1);
		
		return output;
		
		
	}
	public void train(double[][] desiredoutput){
		double[][] error=new double[desiredoutput.length][desiredoutput[0].length];
		double[][] delta=new double[desiredoutput.length][desiredoutput[0].length];
	for (int i=numhiddenlayer;i>=0;i--){	
		if (i==numhiddenlayer){
			for(int j=0;j<error.length;j++){
				for (int k=0;k<error[j].length;k++){
					error[j][k]=desiredoutput[j][k]-layers.get(i+1)[j][k];
				}
			}
			
			double[][] sigmoid=nonlinear(layers.get(layers.size()-1), true);
			for (int j=0;j<delta.length;j++){
				for (int k=0;k<delta[j].length;k++){
					delta[j][k]=error[j][k]*sigmoid[j][k];
				}
			}
		}else{
			error=matrixmultiply(delta, transpose(weights.get(i+1)));
			delta=new double[error.length][error[0].length];
			for (int j=0;j<delta.length;j++){
				for (int k=0;k<delta[j].length;k++){
					delta[j][k]=error[j][k]*nonlinear(layers.get(i+1),true)[j][k];
				}
			}
		}
		for (int j=0;j<weights.get(i).length;j++){
			for (int k=0;k<weights.get(i)[j].length;k++){
				weights.get(i)[j][k]+=matrixmultiply(transpose(layers.get(i)),delta)[j][k]*alpha;
			}
		}
		
		
	}
		
			
		layers.clear();	
	}
	
	private double[][] nonlinear(double[][] x, boolean deriv){
		double[][] returnlist=new double[x.length][x[0].length];
		if (deriv==true){
			for (int i=0;i<x.length;i++){
				for (int j=0;j<x[i].length;j++){
					returnlist[i][j]=x[i][j]*(1-x[i][j]);
				}
			}
			return returnlist;
		}else{
			for (int i=0;i<x.length;i++){
				for (int j=0;j<x[i].length;j++){
					returnlist[i][j]=1/(1+Math.exp(-x[i][j]));
				}
			}
			return returnlist;
		}
	}
	private double[][] matrixmultiply(double[][] matrix1, double[][] matrix2){
		double[][] newmatrix=new double[matrix1.length][matrix2[0].length];
		for (int i=0;i<matrix1.length;i++){
			for (int j=0;j<matrix2[0].length;j++){
				for (int k=0;k<matrix1[0].length;k++){
					newmatrix[i][j]+=matrix1[i][k]*matrix2[k][j];
				}
			}
		}
		return newmatrix;
	}
	
	private double[][] transpose(double[][] matrix){
		double[][] returnmatrix=new double[matrix[0].length][matrix.length];
		for (int i=0;i<matrix.length;i++){
			for (int j=0;j<matrix[i].length;j++){
				returnmatrix[j][i]=matrix[i][j];
			}
		}
		return returnmatrix;
	}
}