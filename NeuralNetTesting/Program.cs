using System;
using System.Threading;

namespace NeuralNetTesting
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("The network is being started 0w0");
            //change this path to change read location
            Datas d = new Datas(@"C:\Users\David\Desktop\bookData\HitchHikersGuide.txt");
            Neuron neruon = new Neuron();
            neruon.LinearNeuron(d);
        }
    }
}
