using Emgu;
using Emgu.CV;

using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;

using KerasNET_Generative_Adversarial_Network.src;

using Numpy;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

using System.Diagnostics;

namespace KerasNET_Generative_Adversarial_Network
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Keras.Utils.Util.DisablePySysConsoleLog = true;
            int epochs = 50000;
            int batchSize = 12;
            int samplePeriod = 1;
            int latentDimensions = 100;
            Tuple<int, int> kernelSize = new Tuple<int, int>(3, 3);
            Tuple<int, int> strides = new Tuple<int, int>(2, 2);
            Tuple<int, int> upsampleSize = new Tuple<int, int>(2, 2);
            Tuple<int, int> imageTargetSize = new Tuple<int, int>(224, 224);

            #region Data
            string dataPath = @"E:\dev\StudyingML_DeepLearning\Datasets\225xPokemonPNG";

            List<string>? imagesInDirectory = Directory.EnumerateFiles($@"{dataPath}\train").ToList();
            Console.WriteLine($"Found {imagesInDirectory.Count()} in directory!");

            Console.WriteLine("Creating Data Generator to scale data between -1 and +1...");
            ImageDataGenerator imageDataGenerator = new ImageDataGenerator(featurewise_center: true,
                samplewise_center: true, rescale: 2.0f / 255.0f, horizontal_flip: true);
            KerasIterator? imageDataset = imageDataGenerator.FlowFromDirectory(dataPath, imageTargetSize,
                class_mode: "categorical", batch_size: batchSize);

            #endregion

            #region Discriminator Model
            var inputD = new Input(shape: new Shape(224, 224, 3));
            var x2 = new Conv2D(32, kernelSize, strides, padding: "same").Set(inputD);
            x2 = new LeakyReLU(alpha: 0.2f).Set(x2);
            x2 = new Conv2D(16, kernelSize, strides, padding: "same").Set(x2);
            x2 = new LeakyReLU(alpha: 0.2f).Set(x2);
            x2 = new Conv2D(8, kernelSize, strides, padding: "same").Set(x2);
            x2 = new LeakyReLU(alpha: 0.2f).Set(x2);
            x2 = new Dropout(0.10f).Set(x2);
            x2 = new Flatten().Set(x2);
            var outputD = new Dense(1, activation: "sigmoid").Set(x2);

            Model discriminatorModel = new Model(new BaseLayer[] { inputD }, new BaseLayer[] { outputD });

            discriminatorModel.Compile(new Adam(0.0002f, 0.5f),
                loss: "binary_crossentropy",
                metrics: new string[] { "accuracy" });
            #endregion

            #region Generator Model
            var inputG = new Input(shape: latentDimensions);
            var x = new Dense(128 * 28 * 28).Set(inputG);
            x = new LeakyReLU(0.2f).Set(x);
            x = new Reshape(new Shape(28, 28, 128)).Set(x);
            x = new Conv2DTranspose(64, upsampleSize, strides, "same").Set(x);
            x = new LeakyReLU(0.2f).Set(x);
            x = new Conv2DTranspose(32, upsampleSize, strides, "same").Set(x);
            x = new LeakyReLU(0.2f).Set(x);
            x = new Conv2DTranspose(16, upsampleSize, strides, "same").Set(x);
            x = new LeakyReLU(0.2f).Set(x);
            var outputG = new Conv2D(3, new Tuple<int, int>(28, 28), padding: "same", activation: "tanh").Set(x);

            Model generatorModel = new Model(new BaseLayer[] { inputG }, new BaseLayer[] { outputG });
            #endregion

            #region Combined Model
            BaseLayer inputC = new Input(shape: new Shape(latentDimensions));
            BaseLayer? image = generatorModel.Call(inputC);

            discriminatorModel.SetTrainable(false);

            BaseLayer? fakePrediction = discriminatorModel.Call(image);

            Model combinedModel = new Model(new BaseLayer[] { inputC }, new BaseLayer[] { fakePrediction });

            combinedModel.Compile(new Adam(0.0002f, 0.5f), loss: "binary_crossentropy");
            #endregion

            NDarray? ones = np.ones(batchSize);
            NDarray? zeros = np.zeros(batchSize);

            Console.Clear();

            generatorModel.Summary();
            discriminatorModel.Summary();
            combinedModel.Summary();

            Stopwatch stopwatch = new Stopwatch();

            // Training the model
            for (int i = 0; i <= epochs; i++)
            {
                stopwatch.Start();

                NDarray? noise = np.random.randn(batchSize, latentDimensions);
                NDarray? fakeImages = generatorModel.Predict(noise);

                //Python.Runtime.PyTuple arguments = new Python.Runtime.PyTuple();
                //dynamic realImagesDynamic = imageDataset.PyObject.InvokeMethod("__next__", arguments);
                //Console.WriteLine($"{realImagesDynamic}");
                //NDarray? realImages = realImagesDynamic;

                Keras.Callbacks.History? historyDReal = discriminatorModel.FitGenerator(imageDataset, epochs: 1, verbose: 0);
                //discriminatorModel.TrainOnBatch(realImages, ones);
                double[]? historyDFake = discriminatorModel.TrainOnBatch(fakeImages, zeros);

                NDarray? noiseTrain = np.random.randn(batchSize, latentDimensions);
                double[]? historyC = combinedModel.TrainOnBatch(noiseTrain, ones);

                if (i % 100 == 0 && i != 0)
                {
                    SaveModelCheckPoint(i);
                }
                if (i % samplePeriod == 0)
                {
                    SaveSamples(5, i);
                }

                Console.WriteLine($"EPOCH: {i + 1} | TIME: {stopwatch.Elapsed.TotalSeconds.ToString("F3")} | D REAL LOSS: {historyDReal.HistoryLogs.Values.First().First().ToString("F7")} | D FAKE LOSS: {historyDFake.First().ToString("F7")} | COMBINED LOSS: {historyC.Last().ToString("F4")}");
                stopwatch.Restart();
            }

            stopwatch.Stop();

            void SaveSamples(int numberOfSamples, int epoch)
            {
                NDarray? noise = np.random.randn(new int[] { numberOfSamples, latentDimensions });
                NDarray? images = generatorModel.Predict(noise, batch_size: numberOfSamples);

                if (!Directory.Exists(@".\GeneratedImages"))
                {
                    Directory.CreateDirectory(@".\GeneratedImages");
                }

                for (int i = 0; i < images.len; i++)
                {
                    var imageCSharp = images[i].GetData<float>();

                    byte[] imageBytes = new byte[224 * 224 * 3];
                    for (int j = 0; j < imageCSharp.Length; j++)
                    {
                        imageBytes[j] = Convert.ToByte((0.5f * imageCSharp[j] + 0.5f) * 255);
                    }

                    using (Image<Rgb24> image = Image.LoadPixelData<Rgb24>(imageBytes, 224, 224))
                    {
                        image.Save(@$".\GeneratedImages\epoch{epoch}-{i}.png", new PngEncoder());
                    }
                }
            }

            void SaveModelCheckPoint(int epochs)
            {
                if (!Directory.Exists(@".\Models"))
                {
                    Directory.CreateDirectory(@".\Models");
                }

                combinedModel.Save($@".\Models\model-epoch{epochs}");
            }
        }
    }
}