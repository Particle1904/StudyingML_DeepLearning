using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;

using Numpy;

using static Tensorflow.Binding;

namespace ClassificationNeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //tf.debugging.set_log_device_placement(true);
            //tf.device(@"/CPU:0");

            // Loading the FashionMNIST dataset using Keras.
            ((NDarray, NDarray), (NDarray, NDarray)) dataset = FashionMNIST.LoadData();

            // Splitting it into Train and Test data, and into Features and Labels.
            NDarray xTrain = dataset.Item1.Item1;
            NDarray yTrain = dataset.Item1.Item2;
            NDarray xTest = dataset.Item2.Item1;
            NDarray yTest = dataset.Item2.Item2;

            // Normalizing the Features(images) to be a pixel value between 0f and 1.0f(from 0 to 255).
            // Neural Nets love Normalized/Scaled data!!!
            NDarray? xTrainNormalized = xTrain / xTrain.max();
            NDarray? xTestNormalized = xTest / xTest.max();

            // Creating a human readable Labels List. Unused in final code.
            string[] labels = new string[] { "T-shirt/top", "Trouser", "Pullover", "Dress",
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" };

            tf.set_random_seed(42);

            // Creating the model.
            Sequential model = new Sequential();
            model.Add(new Flatten());
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(10, activation: "softmax"));

            // Compiling the model.
            model.Compile(optimizer: new Adam(), loss: "sparse_categorical_crossentropy", new string[] { "accuracy" });

            // Traning the model using the Normalized Data and Validation data(test data).
            Keras.Callbacks.History? nonNormHistory = model.Fit(xTrainNormalized, yTrain, batch_size: 512, epochs: 500, verbose: 1,
                validation_data: new NDarray[] { xTestNormalized, yTest });

            // Making predictions using the normalized Test data(not the best, but did it anyway for learning purposes).
            NDarray? yProbs = model.Predict(xTestNormalized, batch_size: 512);
            NDarray? yProbsInt = yProbs.argmax(axis: 1);
        }
    }
}