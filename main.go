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

// Inputs: Measurements of Flowers (Independent Variable?)
// What we are trying to predict: Species of flower (based on measurement)

/*
Key Concepts for Context
Neuron
Activation: Is that nueron "on" or "off"
Weights and Biases
Back Propogation: An algorithm for how a SINGLE training example would like to "nudge" the outputted weights and bias based on whats expected.
	Activation Function
		Many choices but using: Sigmoid Function
			Dirivative: used for backpropogation
Back Propogation Method

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

func (nn *nueralNet) backPropogate(x, y, wHidden, bHidden, wOut, bOut, output) error {
	return nil
}
