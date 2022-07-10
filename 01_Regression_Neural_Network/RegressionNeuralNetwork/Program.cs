using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;

using Numpy;

namespace RegressionNeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Generating a Features(X) array of numbers starting from -100.0 and stepping by +4.0 until +100.0.
            NDarray X = np.arange(-100.0f, 100.0f, 4.0f);
            // Generating a Labels(y) array thats equal X + 10.0f at all indexes.
            NDarray y = X + 10.0f;

            // Separating the generated Features and Labels into Train and Test(Validation) data.
            NDarray XTrain = X[":40"];
            NDarray XTest = X["40:"];

            NDarray yTrain = y[":40"];
            NDarray yTest = y["40:"];

            // Creating Multiple unique models to compare how the **Hyperparameters** can
            // influence the **predictions** that the model will be able to make.
            #region Creating, Compiling and Training Model 1.
            // Creating the Neural Network Model.
            Sequential model1 = new Sequential();
            model1.Add(new Dense(100, activation: "relu"));
            model1.Add(new Dense(1));

            // Compiling the Neural Network Model.
            model1.Compile(optimizer: new Adam(lr: 0.001f), "mae", new string[] { "mae" });

            // Training(fitting) the Neural Network Model using the Features and Labels TRAIN data.
            model1.Fit(np.expand_dims(XTrain, axis: -1), yTrain, batch_size: 5, epochs: 100);
            #endregion

            #region Creating, Compiling and Training Model 2.
            // Creating the Neural Network Model.
            Sequential model2 = new Sequential();
            model2.Add(new Dense(100, activation: "relu"));
            model2.Add(new Dense(100, activation: "relu"));
            model2.Add(new Dense(1));

            // Compiling the Neural Network Model.
            model2.Compile(optimizer: new Adam(lr: 0.001f), "mae", new string[] { "mae" });

            // Training(fitting) the Neural Network Model using the Features and Labels TRAIN data.
            model2.Fit(np.expand_dims(XTrain, axis: -1), yTrain, batch_size: 5, epochs: 100);
            #endregion

            #region Creating, Compiling and Training Model 3.
            // Creating the Neural Network Model.
            Sequential model3 = new Sequential();
            model3.Add(new Dense(100, activation: "relu"));
            model3.Add(new Dense(100, activation: "relu"));
            model3.Add(new Dense(100, activation: "relu"));
            model3.Add(new Dense(1));

            // Compiling the Neural Network Model.
            model3.Compile(optimizer: new Adam(lr: 0.001f), "mae", new string[] { "mae" });

            // Training(fitting) the Neural Network Model using the Features and Labels TRAIN data.
            model3.Fit(np.expand_dims(XTrain, axis: -1), yTrain, batch_size: 5, epochs: 100);
            #endregion

            #region Creating, Compiling and Training Model 4.
            // Creating the Neural Network Model.
            Sequential model4 = new Sequential();
            model4.Add(new Dense(100, activation: "relu"));
            model4.Add(new Dense(100, activation: "relu"));
            model4.Add(new Dense(100, activation: "relu"));
            model4.Add(new Dense(100, activation: "relu"));
            model4.Add(new Dense(100, activation: "relu"));
            model4.Add(new Dense(1));

            // Compiling the Neural Network Model.
            model4.Compile(optimizer: new Adam(lr: 0.001f), "mae", new string[] { "mae" });

            // Training(fitting) the Neural Network Model using the Features and Labels TRAIN data.
            model4.Fit(np.expand_dims(XTrain, axis: -1), yTrain, batch_size: 10, epochs: 500);
            #endregion

            // Making predictions with our Models, using the Test Features Data.
            NDarray prediction1 = model1.Predict(XTest);
            NDarray prediction2 = model2.Predict(XTest);
            NDarray prediction3 = model3.Predict(XTest);
            NDarray prediction4 = model4.Predict(XTest);

            // Printing the original yTest data to compare with the models
            // predictions to see how close it was to the "real" values.

            Console.WriteLine($"\nTest Labels:\n{yTest}\n");

            // Showing the predictions(Labels) the model made on the Test Data..
            Console.WriteLine($"\nModel1 Pred:\n{prediction1}\n");
            Console.WriteLine($"\nModel2 Pred:\n{prediction2}\n");
            Console.WriteLine($"\nModel3 Pred:\n{prediction3}\n");
            Console.WriteLine($"\nModel4 Pred:\n{prediction4}\n");

            // Saving the Model
            //model2.Save(@"E:\dev\Models\Model.h5");

            Console.ReadKey();
        }
    }
}