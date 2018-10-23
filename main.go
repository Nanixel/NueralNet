package main

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Regular Nueral Nets: https://www.datadan.io/building-a-neural-net-from-scratch-in-go/
// Great Video Resource: https://www.youtube.com/watch?v=Ilg3gGewQ5U

// Architecture
// 3 Layers
// 1st = input layer
// 2nd = hidden layer
// 3rd = output layer

// Inputs: Measurements of Flowers (Independent Variable)
// What we are trying to predict: Species of flower (based on measurement)

/*
Key Concepts for Context
Neuron
Activation: Is that nueron "on" or "off"
Weights and Biases
Back Propogation: An algorithm for how a SINGLE training example would like to "nudge" the outputted weights and bias based on whats expected.
	Activation Function: The purpose of an activation function is to produce non-linearity into the output of a neuron.  Most real world data is non-linear.
		Many choices but using: Sigmoid Function
			Sigmoid: Takes a real-valued input and squashes it to range between 0 and 1
			Others:
				Tanh: Takes a real-valued input and squashes it to the range -1 and 1 -> tanh(x) = 2y(2x)
				ReLu: Takes a real-valued input and thresholds it at zero (replaces negative values with zero) -> f(x) {return max(0,x)}
			Dirivative: used for backpropogation
Back Propogation Method
	Three ways to affect the output of a neuron:
		1. Increase bias b
		2. Increase weights[i]
			- increase weights that have the highest activation for the desired output in proportion to a(i)
		3. Change activation function
			- All weights that are positive get brighter, all weights that are negative get dimmer
			- Change a(i) in proportion to w(i)
	Gives you a list of nudges that you want to happen to the second to last layer (by adding together all of the desires for each output neuron aka how each output nueron wants the weights and biases to be adjusted)
		- you can recrusively apply the same process to the weights and values of the previous layers (second to last layer acts as the output layer at this point)

Stochastic Gradient Descent (SGD) to determine updates to weights and bias: Imaging a 3D field with random high and lowpoints this is known as a gradient.  We use stochastic gradient decent to find the lowest "valley" in this gradient.  The lowest "valley" corresponds to the lowest amount of error

Feed Forward
*/

// What we mean by "Learning": We want to find the weights and bias that minimize the cost function
// Cost = Output your network gives vs the output you wanted it to give
// Add the squares of the differences between each component
// A component in this regard being a singular output neuron
// The negative gradient of the cost function is what tells you how you need to change all the weights and biases

// Propogating Backwards = adding tegether all the list of desired "nudges"

// Keep in mind: the adjustments to the weights and biases NEED to consider every training example (in the digits example, we need to consider the numbers 1,2,3,4,5,6,7,8,9)
// We cant directly change activations -> we can only influence weights and biases
type nueralNet struct {
	config  nueralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// defines nueral network architecture
type nueralNetConfig struct {
	inputNuerons  int
	outputNuerons int
	hiddenNuerons int
	numEpochs     int
	learningRate  float64
}

func main() {
	//fmt.Println("Hello World")
}

func newNetwork(config nueralNetConfig) *nueralNet {
	return &nueralNet{config: config}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return math.Exp(x) / math.Pow(1+math.Exp(x), 2)
}

// x = features of our dataset aka flower features
// y = representation of what we are trying to predict
func (nn *nueralNet) train(x, y *mat.Dense) error {

	randSource := rand.NewSource(time.Now().UnixNano())
	randGenerator := rand.New(randSource)

	//NOTE I dont yet understand why these are needed
	// think about this in the case of the 3 later network described above (L1 = 4N, L2 = 3N, L3 = 3N)
	wHidden := mat.NewDense(nn.config.inputNuerons, nn.config.hiddenNuerons, nil) // 4x3
	bHidden := mat.NewDense(1, nn.config.hiddenNuerons, nil)                      // 1x3
	wOut := mat.NewDense(nn.config.hiddenNuerons, nn.config.outputNuerons, nil)   // 3x3
	bOut := mat.NewDense(1, nn.config.outputNuerons, nil)                         // 1x3

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	// fill the matricies with random values (non-influential aka starting with a hypothesis)
	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGenerator.Float64()
		}
	}

	output := new(mat.Dense)

	if err := nn.backPropogate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// set properties
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

func (nn *nueralNet) backPropogate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	for i := 0; i < nn.config.numEpochs; i++ {
		hiddenLayerInput := new(mat.Dense)
		// Mul takes the product of a and b and places the result in the reciever
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}

		// Apply the function fn to each of the elements in the referenced matrix
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)

		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}

		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		networkError := new(mat.Dense)
		networkError.Sub(y, output) // subtract desired output from the actual output

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }

		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}
