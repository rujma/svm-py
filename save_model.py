import numpy as np

def save_model_linear(SV, Alphas, Bias):

    with open('linear.out', 'a') as f:
        dataToWrite = SV
        np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
        f.write("EOP\n")
        dataToWrite = Alphas.transpose()
        np.savetxt(f, dataToWrite, '%5.3f', delimiter=',')
        f.write("EOP\n")
        dataToWrite = Bias
        np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
        f.write("EOP\n")

def save_model_poly (SV, Alphas, Bias, Gamma, Degree):
	
	with open('poly.out', 'a') as f:
		dataToWrite = SV
		np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
		f.write("EOP\n")
		dataToWrite = Alphas.transpose()
		np.savetxt(f, dataToWrite, '%5.3f', delimiter=',')
		f.write("EOP\n")
		dataToWrite = Bias
		np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
		f.write("EOP\n")
		f.write(str(Gamma) + "\n")
		f.write("EOP\n")
		f.write(str(Degree) + "\n")
		f.write("EOP\n")

def save_model_rbf (SV, Alphas, Bias, Gamma):
	
	with open('rbf.out', 'a') as f:
		dataToWrite = SV
		np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
		f.write("EOP\n")
		dataToWrite = Alphas.transpose()
		np.savetxt(f, dataToWrite, '%5.3f', delimiter=',')
		f.write("EOP\n")
		dataToWrite	 = Bias
		np.savetxt(f, dataToWrite,'%5.3f', delimiter=',')
		f.write("EOP\n")
		f.write(str(Gamma) + "\n")
		f.write("EOP\n")