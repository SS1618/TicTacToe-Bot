// TicTacToeNeuralNet.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <vector>
#include "math.h"
#include <cstdlib>
#include <assert.h>

using namespace std;
struct Connection {
	double weight;
	double deltaWeight;
};
class Neuron;
typedef vector<Neuron> Layer;
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { n_outputVal = val; }
	double getOutputVal(void) const { return n_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	double sumDOW(const Layer &nextLayer) const;
	void updateInputWeights(Layer &prevLayer);
private:
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	static double eta; // 0.0 to 1.0
	static double alpha; //0.0 to n
	double n_outputVal;
	vector<Connection> n_outputWeights;
	unsigned n_myIndex;
	double n_gradient;
};
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
void Neuron::updateInputWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.n_outputWeights[n_myIndex].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal()*n_gradient + alpha * oldDeltaWeight;
		//eta = learning rate, alpha = takes fraction of previous change in weight
		neuron.n_outputWeights[n_myIndex].deltaWeight = newDeltaWeight;
		neuron.n_outputWeights[n_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += n_outputWeights[n].weight * nextLayer[n].n_gradient;
	}
	return sum;
}
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	n_gradient = dow * Neuron::transferFunctionDerivative(n_outputVal);
}
void Neuron::calcOutputGradients(double targetVals) {
	double delta = targetVals - n_outputVal;
	n_gradient = delta * Neuron::transferFunctionDerivative(n_outputVal);
}
double Neuron::transferFunction(double x) {
	return tanh(x);
}
double Neuron::transferFunctionDerivative(double x) {
	return 1.0 - (x*x); //aprox. for domain 
}
void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].n_outputWeights[n_myIndex].weight;
	}
	n_outputVal = Neuron::transferFunction(sum); //activation function
}
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; c++) {
		n_outputWeights.push_back(Connection());
		n_outputWeights.back().weight = randomWeight();
	}
	n_myIndex = myIndex;
}
class Net {
public:
	Net(const vector<unsigned> &toplogy);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;

private:
	vector<Layer> m_layers; //m_layer[layerNumber][neuronInLayer]
	double n_error;
	double n_recentAverageError;
	static double n_recentAverageSmoothingFactor;
};
double Net::n_recentAverageSmoothingFactor = 18000.0;
void Net::getResults(vector<double> &resultVals) const {
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
void Net::backProp(const vector<double> &targetVals) {
	//calculated overall net error
	Layer &outputLayer = m_layers.back();
	n_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		n_error += delta * delta;
	}
	n_error /= outputLayer.size() - 1;
	n_error = sqrt(n_error);

	//checks average error

	//cout << "error: " << n_recentAverageError << endl;
	n_recentAverageError = (n_recentAverageError * n_recentAverageSmoothingFactor * n_error) / (n_recentAverageSmoothingFactor + 1.0);
	cout << "Average Error: " << (n_recentAverageError * n_recentAverageSmoothingFactor * n_error) / (n_recentAverageSmoothingFactor + 1.0) << endl;
	//cout << "error2: " << n_recentAverageError << endl;

	//calc output gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	//calc hidden gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	//update weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}
void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1); //makes sure number of inputs and equal to number of input neurons
	//makes input neurons output a certain data input
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
Net::Net(const vector<unsigned> &topology) {
	unsigned numLayers = topology.size();
	n_recentAverageError = 1.0;
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer()); //adds a layer
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];//if layerNum is the output layer then set to 0 otherwise set to number of neurons in next layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "added neuron for layer " << layerNum << endl;
		}
		m_layers.back().back().setOutputVal(1.0);
	}
}
class GameBoard {
private:
	vector<int> boardState;
public:
	GameBoard();
	void printBoard();
	bool inputSymbol(int player, unsigned pos);
	int stateWin();
	void getBoardState(vector<int> &state);
	int getValAtPos(int pos);
	void resetBoard();
	bool checkIfFull();

};
GameBoard::GameBoard() {
	for (int i = 0; i < 9; i++) {
		boardState.push_back(0);
	}
}
bool GameBoard::checkIfFull() {
	for (int i = 0; i < 9; i++) {
		if (boardState[i] == 0) {
			return false;
		}
	}
	return true;
}
void GameBoard::printBoard() {
	for (int i = 0; i < 9; i++) {
		if ((i + 1) % 3 != 0) {
			cout << boardState[i] << "\t|\t";
		}
		else {
			cout << boardState[i] << endl;
			cout << "-------------------" << endl;
		}
	}
}
bool GameBoard::inputSymbol(int player, unsigned pos) {
	if (boardState[pos] != 0) {
		return false;
	}
	if (player == 1) {
		boardState[pos] = 1;
	}
	else if (player == -1) {
		boardState[pos] = -1;
	}
	return true;
}
int GameBoard::stateWin() {
	//vector<vector<double>> boardSetUp;
	for (int i = 0; i < 9; i += 3) {
		if (boardState[i] == boardState[i+1] && boardState[i] == boardState[i+2] && boardState[i+2] == boardState[i+1] && boardState[i]!=0) {
			return boardState[i];
		}
	}
	for (int i = 0; i < 3; i++) {
		if (boardState[i] == boardState[i + 3] && boardState[i] == boardState[i + 6] && boardState[i+6]==boardState[i+3] && boardState[i] != 0) {
			return boardState[i];
		}
	}
	if (boardState[0] == boardState[4] && boardState[0] == boardState[8] && boardState[4] == boardState[8] && boardState[0] != 0) {
		return boardState[0];
	}
	else if (boardState[2] == boardState[4] && boardState[2] == boardState[6] && boardState[6] == boardState[4] && boardState[2] != 0) {
		return boardState[2];
	}
	return 0;
}
void GameBoard::getBoardState(vector<int> &state) {
	state.clear();
	for (int i = 0; i < 9; i++) {
		state.push_back(boardState[i]);
	}
}
int GameBoard::getValAtPos(int pos) {
	return boardState[pos];
}
void GameBoard::resetBoard() {
	boardState.clear();
	for (int i = 0; i < 9; i++) {
		boardState.push_back(0);
	}
}
/*
0 1 2
3 4 5
6 7 8
*/
int main()
{
	GameBoard board;
	vector<unsigned> topology;
	topology.push_back(18);
	topology.push_back(9);

	vector<double> inputVals;
	vector<int> boardVals;
	vector<double>targetVals;
	vector<double> resultVals;
	int answer = 0;
	Net net(topology);
	while (answer == 0) {
		board.resetBoard();
		while (board.stateWin() == 0 && board.checkIfFull() == false) {
			//input current board status into network
			inputVals.clear();
			board.getBoardState(boardVals);
			for (int i = 0; i < boardVals.size(); i++) {
				if (boardVals[i] == 1) {
					inputVals.push_back(1);
					inputVals.push_back(0);
				}
				else if (boardVals[i] == -1) {
					inputVals.push_back(0);
					inputVals.push_back(1);
				}
				else {
					inputVals.push_back(0);
					inputVals.push_back(0);
				}
			}
			net.feedForward(inputVals);
			//get net's outut
			net.getResults(resultVals);
			double min = 1.0;
			int pos = 0;
			for (int i = 0; i < resultVals.size(); i++) {
				if (resultVals[i] < min && boardVals[i] == 0) {
					pos = i;
					min = resultVals[i];
				}
			}
			board.inputSymbol(1, pos);
			board.printBoard();
			if (board.stateWin() == 1) {
				cout << "Win to computer" << endl;
				break;
			}
			else if (board.checkIfFull() == true) {
				cout << "Draw" << endl;
				break;
			}
			//make net predict player's best move
			board.getBoardState(boardVals);
			inputVals.clear();
			for (int i = 0; i < boardVals.size(); i++) {
				if (boardVals[i] == 1) {
					inputVals.push_back(1);
					inputVals.push_back(0);
				}
				else if (boardVals[i] == -1) {
					inputVals.push_back(0);
					inputVals.push_back(1);
				}
				else {
					inputVals.push_back(0);
					inputVals.push_back(0);
				}
			}
			net.feedForward(inputVals);
			//get player's move
			cout << "Enter position: ";
			int userInput;
			cin >> userInput;
			board.inputSymbol(-1, userInput);
			board.printBoard();
			for (int i = 0; i < 9; i++) {
				if (i == userInput) {
					targetVals.push_back(1);
				}
				else {
					targetVals.push_back(0);
				}
			}
			net.backProp(targetVals);
			if (board.stateWin() == -1) {
				cout << "Win to player" << endl;
			}
			cout << endl;
		}
		cout << "New Game? (0 = yes, 1 = no): ";
		cin >> answer;
	}
	
	

	answer = 0;
	while (answer == 0) {
		board.resetBoard();
		while (board.stateWin() == 0 && board.checkIfFull() == false) {
			//get player's move
			cout << "Enter position: ";
			int userInput;
			cin >> userInput;
			board.inputSymbol(1, userInput);
			board.printBoard();
			for (int i = 0; i < 9; i++) {
				if (i == userInput) {
					targetVals.push_back(1);
				}
				else {
					targetVals.push_back(0);
				}
			}
			if (board.stateWin() == 1) {
				cout << "Win to player" << endl;
			}
			else if (board.checkIfFull() == true) {
				cout << "Draw" << endl;
				break;
			}

			//input current board status into network
			inputVals.clear();
			board.getBoardState(boardVals);
			for (int i = 0; i < boardVals.size(); i++) {
				if (boardVals[i] == 1) {
					inputVals.push_back(1);
					inputVals.push_back(0);
				}
				else if (boardVals[i] == -1) {
					inputVals.push_back(0);
					inputVals.push_back(1);
				}
				else {
					inputVals.push_back(0);
					inputVals.push_back(0);
				}
			}
			net.feedForward(inputVals);
			//get net's outut
			net.getResults(resultVals);
			double min = 1.0;
			int pos = 0;
			for (int i = 0; i < resultVals.size(); i++) {
				if (resultVals[i] < min && boardVals[i] == 0) {
					pos = i;
					min = resultVals[i];
				}
			}
			board.inputSymbol(-1, pos);
			board.printBoard();
			if (board.stateWin() == -1) {
				cout << "Win to computer" << endl;
			}
			else if (board.checkIfFull() == true) {
				cout << "Draw" << endl;
			}
			cout << endl;
		}
		cout << "Play again?" << endl;
		cin >> answer;
	}



	

 
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
