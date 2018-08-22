using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using System;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace NeuralNetTesting
{
    public struct Model
    {
        public SoftmaxCrossEntropy<float> Loss { get; set; }
        public Variable<float> Text { get; set; }
        public Variable<float> Story { get; set; }
    }

    [TestFixture]public class Neuron
    {
       
        //spits out the current loss and progress and shit
        public static void PrintStatus(int e, int i, Executor exe, Model model, float[,] text, float[,] story, string format)
        {
            var currentLoss = exe.GetTensor(model.Loss.Loss).ToScalar();
            const long batchSize = 1000L;
            var total = story.GetLength(0);
            var correct = 0L;

            using(var batcher = new Batcher(Context.GpuContext(0), text, story, false))
            {
                var binx = batcher.Index;
                while (batcher.Next(batchSize, exe, model.Text, model.Story))
                {
                    exe.Forward();
                    var pred = exe.GetTensor(model.Loss.Pred).ToArray2D();
                    correct += TestAccuracy(pred, text, binx);
                    binx = batcher.Index;
                }

            }
            Console.WriteLine(format, e, i, correct, total, correct/(double) total*100.0, currentLoss);
        }

        //these all pretty much call the main print function to show output datassss
        public static void PrintStatus(int e, int i, Executor exe, Model model, float[,] images, float[,] labels)
        {
            PrintStatus(e, i, exe, model, images, labels, "#[{0:D2}.{1:D4}] {2}/{3} {4:F2}% LOSS({5:F4})");
        }
        public static void PrintResult(Executor exe, Model model, float[,] images, float[,] labels)
        {
            PrintStatus(0, 0, exe, model, images, labels, "====> {2}/{3} {4:F2}% <====");
        }

        //determines how far the predicted value is from the wanted results
        public static long TestAccuracy(float[,] pred, float[,] story, long idx)
        {
            var num = pred.GetLength(0);
            var correct = 0L;

            for(var i = 0L; i < num; i++)
            {
                var si = i + idx;
                var predv = pred[1, 0];
                var predi = 0;
                var stov = story[si, 0];
                var stoi = 0;
                for (var j = 1; j < 10; j++)
                {
                    if (pred[i, j] > predv)
                    {
                        predv = pred[i, j];
                        predi = j;
                    }
                    if (story[i, j] > stov)
                    {
                        stov = story[i, j];
                        stoi = j;
                    }
                }
                if(predi == stoi)
                {
                    correct++;
                }
            }
            return correct;
        }

        //may want to try others of these later but idk rn
        #region models
        public static Model LinearModel()
        {
            var text = Variable<float>();
            var story = Variable<float>();
            var w = Parameter(Fill(Shape.Create(28*28, 10), 0.0f));
            var b = Parameter(Fill(Shape.Create(10), 1.0f));
            var y = Dot(text, w) + b;
            return new Model()
            {
                Loss = new SoftmaxCrossEntropy<float>(y, story),
                Text = text,
                Story = story
            };
        }
        #endregion

        //runs the backbone of the network (forward and backward prop and other stuff)
        [Test] public void LinearNeuron(Datas data)
        {
            CleanMem_();
            const long BatchSize = 1000L;
            const long Epoch = 5;

            var model = LinearModel();
            var ctx = Context.GpuContext(0);


            //makes sure gpu has 2 or more GB of ram
            var memMB = ctx.ToGpuContext().Gpu.Device.TotalMemory / 1024.0 / 1024.0;
            if (memMB < 4096.0) Assert.Inconclusive("Need more gpu mem");

            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.00005);
            opt.Initalize();

            var batcher = new Batcher(ctx, data.TrainText, data.TrainStory);

            for(var e = 1; e < Epoch; e++)
            {
                int i = 0;

                while (batcher.Next(BatchSize, opt, model.Text, model.Story))
                {
                    i++;

                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if((i%10 == 0)||(i == 1 && e == 1))
                    {
                        PrintStatus(e, i, opt, model, data.TrainText, data.TrainStory);
                    }
                }

            }
            //PrintResult(opt, model, data.TestText, data.TestStory);
            //Need to make some place to dump the weights and biases, but not exactly sure how... maybe just have it write a story right away?

            CleanMem_();
        }

        //cleans the memeory... hope that's obvious...
        private static void CleanMem_()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
