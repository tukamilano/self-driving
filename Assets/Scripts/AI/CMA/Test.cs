using System;
using System.Collections.Generic;

namespace Practice1
{
    internal static class Program
    {
        private static double Quadratic(double x1, double x2)
        {
            return Math.Pow(x1 - 3, 2) + Math.Pow(10 * (x2 + 2), 2);
        }

        private static void Main()
        {
            var optimizer = new CMA(new[] { 0.0, 0.0 }, 1.3);

            for (int generation = 0; generation < 50; generation++)
            {
                var solutions = new List<(double[] Parameters, double Value)>();
                for (int i = 0; i < optimizer.PopulationSize; i++)
                {
                    var x = optimizer.Ask();
                    double value = Quadratic(x[0], x[1]);
                    solutions.Add((x, value));
                    Console.WriteLine($"#{generation} {value} (x1={x[0]}, x2={x[1]})");
                }

                optimizer.Tell(solutions);
            }
        }
    }
}
