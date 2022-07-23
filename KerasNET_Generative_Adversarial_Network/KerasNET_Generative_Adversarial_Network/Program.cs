using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;

using KerasNET_Generative_Adversarial_Network.src;

using Numpy;

using System.Drawing;
using System.Drawing.Imaging;

namespace KerasNET_Generative_Adversarial_Network
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int epochs = 50000;
            int batchSize = 32;
            int samplePeriod = 200;

            string dataPath = @"E:\dev\StudyingML_DeepLearning\Datasets\225xPokemonJPG";

            List<string>? imagesInDirectory = Directory.EnumerateFiles($@"{dataPath}\train").ToList();
            Console.WriteLine($"Found {imagesInDirectory.Count()} in directory!");

            Console.WriteLine("Allocating NDarray to store dataset...");
            NDarray trainImages = np.array(new float[imagesInDirectory.Count, 224, 224, 3]);

            for (int i = 0; i < imagesInDirectory.Count(); i++)
            {
                var imageFromFile = ImageUtil.LoadImg(imagesInDirectory[i], target_size: new Shape(224, 224, 3));
                trainImages[i] = ImageUtil.ImageToArray(imageFromFile);
                Console.WriteLine($"Loaded {i + 1} out of {imagesInDirectory.Count()}");
            }
            Console.WriteLine("Finished loading stage...");

            Console.WriteLine("Preprocessing data...");
            trainImages = trainImages / 255.0f * 2 - 1;
            Console.WriteLine("Finished preprocessing stage...");

            Console.WriteLine($"Shape of trainImages: {trainImages.shape}");

            int latentDimensions = 200;

            //var trainShape = dataGenerator;

            // Building the Generator
            var inputG = new Input(shape: latentDimensions);
            var x = new Dense(256).Set(inputG);
            x = new LeakyReLU(0.2f).Set(x);
            x = new BatchNormalization(momentum: 0.8f).Set(x);
            x = new Dense(512).Set(x);
            x = new LeakyReLU(0.2f).Set(x);
            x = new BatchNormalization(momentum: 0.8f).Set(x);
            x = new Dense(1024).Set(x);
            x = new LeakyReLU(0.2f).Set(x);
            x = new BatchNormalization(momentum: 0.8f).Set(x);
            var outputG = new Dense(224 * 224, activation: "tanh").Set(x);

            Model generatorModel = new Model(new BaseLayer[] { inputG }, new BaseLayer[] { outputG });

            // Building the Discriminator
            var inputD = new Input(shape: new Shape(224 * 224));
            var x2 = new Dense(512).Set(inputD);
            x2 = new LeakyReLU(alpha: 0.2f).Set(x2);
            x2 = new Dense(256).Set(x2);
            x2 = new LeakyReLU(alpha: 0.2f).Set(x2);
            var outputD = new Dense(1, activation: "sigmoid").Set(x2);

            Model discriminatorModel = new Model(new BaseLayer[] { inputD }, new BaseLayer[] { outputD });

            discriminatorModel.Compile(new Adam(0.0002f, 0.5f),
                loss: "binary_crossentropy",
                metrics: new string[] { "accuracy" });

            BaseLayer z = new Input(shape: new Shape(latentDimensions));
            BaseLayer? image = generatorModel.Call(z);

            discriminatorModel.SetTrainable(false);

            BaseLayer? fakePrediction = discriminatorModel.Call(image);

            Model combinedModel = new Model(new BaseLayer[] { z }, new BaseLayer[] { fakePrediction });

            combinedModel.Compile(new Adam(), loss: "binary_crossentropy");

            NDarray? ones = np.ones(batchSize);
            NDarray? zeros = np.zeros(batchSize);

            generatorModel.Summary();
            discriminatorModel.Summary();
            combinedModel.Summary();

            // Training the model
            for (int i = 0; i <= epochs; i++)
            {
                NDarray<int>? randomIndexTrain = np.random.randint(0, trainImages.shape[0]);

                NDarray? realImages = trainImages[randomIndexTrain];

                NDarray? noise = np.random.rand(batchSize, latentDimensions);
                NDarray? fakeImages = generatorModel.Predict(noise);

                discriminatorModel.TrainOnBatch(realImages, ones);
                discriminatorModel.TrainOnBatch(fakeImages, zeros);

                noise = np.random.rand(batchSize, latentDimensions);
                combinedModel.TrainOnBatch(noise, ones);

                if (i % 100 == 0)
                {
                    Console.WriteLine($"Epoch: {i + 1}");
                }
                if (i % samplePeriod == 0)
                {
                    SaveSamples();
                }
            }

            if (!Directory.Exists(@".\Models"))
            {
                Directory.CreateDirectory(@".\Models");
            }

            combinedModel.Save(@".\Models\model");

            void SaveSamples()
            {
                int rows = 5;
                int columns = 5;
                NDarray? noise = np.random.randn(new int[] { rows * columns, latentDimensions });

                NDarray? images = generatorModel.Predict(noise);
                images = 0.5f * images + 0.5f;

                if (!Directory.Exists(@".\GeneratedImages"))
                {
                    Directory.CreateDirectory(@".\GeneratedImages");
                }

                for (int i = 0; i < images.len; i++)
                {
                    byte[] imageBytes = new byte[100];
                    MemoryStream memoryStream = new MemoryStream(imageBytes);
                    Bitmap bitmap = new Bitmap(memoryStream);
                    string[]? existingFiles = Directory.GetFiles(@".\GeneratedImages");
                    if (existingFiles.Length <= 0)
                    {
                        bitmap.Save(@".\0001", ImageFormat.Png);
                    }
                    else
                    {
                        string lastFileSaved = existingFiles.Last();
                        string[]? splitPath = lastFileSaved.Split(@"\");
                        string? fileName = splitPath.Last().ToString().Replace(".png", "");
                        bool fileNumber = Int32.TryParse(fileName, out int result);
                        if (fileNumber == true)
                        {
                            bitmap.Save($@".\{result++}", ImageFormat.Png);
                        }
                    }
                }
            }
        }
    }
}