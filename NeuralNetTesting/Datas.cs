using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;

namespace NeuralNetTesting
{
    public class Datas
    {
        public string path;

        //arrays that get used to train the network and store final output data
        public float[,] TrainText { get; }
        public float[,] TrainStory { get; }
      //public float[,] ValText { get; }
      //public float[,] ValStory { get; }
        public float[,] TestText { get; }
        public float[,] TestStory { get; }
        public long numTest;

        //dictionarys to deal with text to number "conversions"
        //basically I assicate each word in the .txt file with a number in the dictionary and use these to give the weight and biases meaning
        Dictionary<int, string> dict;
        Dictionary<int, string> noRepeats;
        Dictionary<string, int> noRepeatsRev;
        Dictionary<int, string> outPuts;

        private StreamReader stream1;

        public Datas(string path)
        {
            stream1 = new StreamReader(path);

            dict = new Dictionary<int, string>();
            noRepeats = new Dictionary<int, string>();
            noRepeatsRev = new Dictionary<string, int>();
            outPuts = new Dictionary<int, string>();

            numTest = (long)dict.Count;
            TrainText = new float[numTest, 1];
            TrainStory = new float[numTest, 1];
        }

        //extracts all words from book/script/whatever to and adds them to the basic dictionary
        private void ParseWords()
        {
            string s = "";
            int x = 0;
            while (stream1.Peek() != -1)
            {
                if(" ".Equals((char)stream1.Peek()))
                {
                    s = s.ToLower();
                    stream1.Read();
                    dict.Add(x, s);
                    x++;
                    s = "";
                }
                s = s + (char)stream1.Read();
                
            }
            x++;
            dict.Add(x,s);
        }
        
        public void Init()
        {
            ParseWords();
            SortRepeats();
            ReverseNoRepeats();
            fillTraining();
        }

        //generates a dictionary that only contains original words from the text
        private void SortRepeats()
        {
            string s = "";
            int y = 0;

            for (int x = 0; x < dict.Count + 1; x++)
            {
                dict.TryGetValue(x, out s);
                if (!noRepeats.ContainsValue(s))
                {
                    noRepeats.Add(y, s);
                    y++;
                }
            }

            ReverseNoRepeats();
        }

        //flips the association of int/string to string/int to make filling the training arrays easier
        private void ReverseNoRepeats()
        {
            for(int x = 0; x < noRepeats.Count+1; x++)
            {
                noRepeats.TryGetValue(x, out string s);
                noRepeatsRev.Add(s, x);
            }
        }

        //creates a "conversion" from the strings to ints to put into the training array
        private void fillTraining()
        {
            for (int i = 0; i < dict.Count + 1; i++)
              {

                //assigns input value
                dict.TryGetValue(i, out string s);
                noRepeatsRev.TryGetValue(s, out int y);
                TrainText[i , 1] = y;

                //assigns wanted output value
                dict.TryGetValue(i + 1, out s);
                noRepeatsRev.TryGetValue(s, out y);
                TrainStory[i, 1] = y;

            }
        }

        public void NumToWord()
        {

        }
    }
}
