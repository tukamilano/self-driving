using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Practice1
{
    // CMA-ES stochastic optimizer translated from the original Python implementation.
    // Requires MathNet.Numerics for dense matrix and vector operations.
    public class CMA
    {
        private const double EPS = 1e-8;
        private const double MEAN_MAX = 1e32;
        private const double SIGMA_MAX = 1e32;

        private readonly int _nDim;
        private readonly int _popSize;
        private readonly int _mu;
        private readonly double _muEff;
        private readonly double _cc;
        private readonly double _c1;
        private readonly double _cmu;
        private readonly double _cSigma;
        private readonly double _dSigma;
        private readonly double _cm;
        private readonly double _chiN;
        private readonly double[] _weights;

        private Vector<double> _pSigma;
        private Vector<double> _pc;
        private Vector<double> _mean;
        private Matrix<double> _C;
        private double _sigma;
        private Vector<double>? _D;
        private Matrix<double>? _B;

        private double[,]? _bounds;
        private readonly int _nMaxResampling;
        private int _generation;
        private Random _rng;

        // learning rate adaptation support
        private readonly bool _lrAdapt;
        private readonly double _alpha;
        private readonly double _betaMean;
        private readonly double _betaSigma;
        private readonly double _gamma;
        private Vector<double> _Emean;
        private Vector<double> _ESigma;
        private double _Vmean;
        private double _VSigma;
        private double _etaMean;
        private double _etaSigma;

        // termination criteria members
        private readonly double _tolx;
        private readonly double _tolxup;
        private readonly double _tolfun;
        private readonly double _tolconditioncov;
        private readonly int _funhistTerm;
        private readonly double[] _funhistValues;

        public CMA(
            double[] mean,
            double sigma,
            double[,]? bounds = null,
            int nMaxResampling = 100,
            int? seed = null,
            int? populationSize = null,
            double[,]? covariance = null,
            bool lrAdapt = false)
        {
            if (sigma <= 0)
            {
                throw new ArgumentException("sigma must be non-zero positive value", nameof(sigma));
            }

            if (mean.Any(x => Math.Abs(x) >= MEAN_MAX))
            {
                throw new ArgumentException($"Abs of all elements of mean vector must be less than {MEAN_MAX}", nameof(mean));
            }

            _nDim = mean.Length;
            if (_nDim == 0)
            {
                throw new ArgumentException("The dimension of mean must be positive", nameof(mean));
            }

            if (populationSize == null)
            {
                populationSize = 4 + (int)Math.Floor(3 * Math.Log(_nDim));
            }

            if (populationSize <= 0)
            {
                throw new ArgumentException("populationSize must be non-zero positive value", nameof(populationSize));
            }

            _popSize = populationSize.Value;
            _mu = _popSize / 2;

            var weightsPrime = DenseVector.Create(_popSize, i => Math.Log((_popSize + 1) / 2.0) - Math.Log(i + 1));
            _muEff = Math.Pow(weightsPrime.SubVector(0, _mu).Sum(), 2) /
                     weightsPrime.SubVector(0, _mu).PointwisePower(2).Sum();
            var muEffMinus = Math.Pow(weightsPrime.SubVector(_mu, _popSize - _mu).Sum(), 2) /
                             weightsPrime.SubVector(_mu, _popSize - _mu).PointwisePower(2).Sum();

            const double alphaCov = 2.0;
            _c1 = alphaCov / (Math.Pow(_nDim + 1.3, 2) + _muEff);
            _cmu = Math.Min(
                1 - _c1 - 1e-8,
                alphaCov * (_muEff - 2 + 1 / _muEff) /
                (Math.Pow(_nDim + 2, 2) + alphaCov * _muEff / 2.0));

            if (_c1 > 1 - _cmu)
            {
                throw new InvalidOperationException("invalid learning rate for the rank-one update");
            }

            if (_cmu > 1 - _c1)
            {
                throw new InvalidOperationException("invalid learning rate for the rank-μ update");
            }

            var minAlpha = Math.Min(
                1 + _c1 / _cmu,
                Math.Min(1 + (2 * muEffMinus) / (_muEff + 2), (1 - _c1 - _cmu) / (_nDim * _cmu)));

            var positiveSum = weightsPrime.Where(w => w > 0).Sum();
            var negativeSum = weightsPrime.Where(w => w < 0).Select(Math.Abs).Sum();

            _weights = weightsPrime.Select(
                w => w >= 0 ? w / positiveSum : minAlpha / negativeSum * w).ToArray();

            _cm = 1.0;

            _cSigma = (_muEff + 2) / (_nDim + _muEff + 5);
            _dSigma = 1 + 2 * Math.Max(0, Math.Sqrt((_muEff - 1) / (_nDim + 1)) - 1) + _cSigma;

            _cc = (4 + _muEff / _nDim) / (_nDim + 4 + 2 * _muEff / _nDim);

            _pSigma = DenseVector.Create(_nDim, 0.0);
            _pc = DenseVector.Create(_nDim, 0.0);
            _mean = DenseVector.OfArray(mean.ToArray());
            _sigma = sigma;

            if (covariance is null)
            {
                _C = DenseMatrix.CreateIdentity(_nDim);
            }
            else
            {
                if (covariance.GetLength(0) != _nDim || covariance.GetLength(1) != _nDim)
                {
                    throw new ArgumentException("Invalid shape of covariance matrix", nameof(covariance));
                }

                _C = DenseMatrix.OfArray(covariance);
            }

            _D = null;
            _B = null;

            if (!IsValidBounds(bounds, _mean))
            {
                throw new ArgumentException("invalid bounds", nameof(bounds));
            }

            _bounds = CloneBounds(bounds);
            _nMaxResampling = nMaxResampling;
            _generation = 0;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();

            _lrAdapt = lrAdapt;
            _alpha = 1.4;
            _betaMean = 0.1;
            _betaSigma = 0.03;
            _gamma = 0.1;
            _Emean = DenseVector.Create(_nDim, 0.0);
            _ESigma = DenseVector.Create(_nDim * _nDim, 0.0);
            _Vmean = 0.0;
            _VSigma = 0.0;
            _etaMean = 1.0;
            _etaSigma = 1.0;

            _chiN = Math.Sqrt(_nDim) * (1.0 - (1.0 / (4.0 * _nDim)) + 1.0 / (21.0 * Math.Pow(_nDim, 2)));

            _tolx = 1e-12 * sigma;
            _tolxup = 1e4;
            _tolfun = 1e-12;
            _tolconditioncov = 1e14;

            _funhistTerm = 10 + (int)Math.Ceiling(30.0 * _nDim / _popSize);
            _funhistValues = new double[_funhistTerm * 2];
            Array.Fill(_funhistValues, double.NaN);
        }

        public int Dim => _nDim;

        public int PopulationSize => _popSize;

        public int Generation => _generation;

        public double[] Mean => _mean.ToArray();

        public void ReseedRng(int seed)
        {
            _rng = new Random(seed);
        }

        public void SetBounds(double[,]? bounds)
        {
            if (!IsValidBounds(bounds, _mean))
            {
                throw new ArgumentException("invalid bounds", nameof(bounds));
            }

            _bounds = CloneBounds(bounds);
        }

        public double[] Ask()
        {
            for (int i = 0; i < _nMaxResampling; i++)
            {
                var x = SampleSolution();
                if (IsFeasible(x))
                {
                    return x;
                }
            }

            var fallback = SampleSolution();
            return RepairInfeasibleParams(fallback);
        }

        public void Tell(IReadOnlyList<(double[] Parameters, double Value)> solutions)
        {
            if (solutions.Count != _popSize)
            {
                throw new ArgumentException("Must tell population-size solutions.", nameof(solutions));
            }

            foreach (var solution in solutions)
            {
                if (solution.Parameters.Any(x => Math.Abs(x) >= MEAN_MAX))
                {
                    throw new ArgumentException($"Abs of all param values must be less than {MEAN_MAX} to avoid overflow errors.");
                }
            }

            _generation += 1;
            var ordered = solutions.OrderBy(s => s.Value).ToList();

            int funhistIdx = 2 * (_generation % _funhistTerm);
            _funhistValues[funhistIdx] = ordered[0].Value;
            _funhistValues[funhistIdx + 1] = ordered[^1].Value;

            var (B, D) = EigenDecomposition();
            _B = null;
            _D = null;

            Vector<double>? oldMean = null;
            double oldSigma = 0.0;
            Matrix<double>? oldSigmaMatrix = null;
            Matrix<double>? oldInvSqrtC = null;

            if (_lrAdapt)
            {
                oldMean = _mean.Clone();
                oldSigma = _sigma;
                oldSigmaMatrix = _C.Clone().Multiply(oldSigma * oldSigma);
                var invD = D.Map(d => 1.0 / d);
                var invDiagMatrix = DenseMatrix.OfDiagonalArray(invD.ToArray());
                oldInvSqrtC = B * invDiagMatrix * B.Transpose();
            }

            var currentMean = _mean.Clone();
            var xk = ordered.Select(s => DenseVector.OfArray(s.Parameters.ToArray())).ToArray();
            var yk = xk.Select(v => v.Subtract(currentMean).Divide(_sigma)).ToArray();

            Vector<double> yW = DenseVector.Create(_nDim, 0.0);
            for (int i = 0; i < _mu; i++)
            {
                yW += yk[i] * _weights[i];
            }

            _mean = _mean + yW * (_cm * _sigma);

            var invDiag = D.Map(d => 1.0 / d);
            var c2 = B * DenseMatrix.OfDiagonalArray(invDiag.ToArray()) * B.Transpose();
            _pSigma = _pSigma * (1 - _cSigma) + c2 * yW * Math.Sqrt(_cSigma * (2 - _cSigma) * _muEff);

            double normPSigma = _pSigma.L2Norm();
            _sigma *= Math.Exp((_cSigma / _dSigma) * (normPSigma / _chiN - 1));
            _sigma = Math.Min(_sigma, SIGMA_MAX);

            double hSigmaCondLeft = normPSigma / Math.Sqrt(1 - Math.Pow(1 - _cSigma, 2 * (_generation + 1)));
            double hSigmaCondRight = (1.4 + 2.0 / (_nDim + 1)) * _chiN;
            double hSigma = hSigmaCondLeft < hSigmaCondRight ? 1.0 : 0.0;

            _pc = _pc * (1 - _cc) + yW * (hSigma * Math.Sqrt(_cc * (2 - _cc) * _muEff));

            var wIo = new double[_popSize];
            for (int k = 0; k < _popSize; k++)
            {
                if (_weights[k] >= 0)
                {
                    wIo[k] = _weights[k];
                }
                else
                {
                    var transformed = c2 * yk[k];
                    double norm = transformed.L2Norm();
                    wIo[k] = _weights[k] * _nDim / (norm * norm + EPS);
                }
            }

            double deltaHSigma = (1 - hSigma) * _cc * (2 - _cc);
            Matrix<double> rankOne = _pc.OuterProduct(_pc);
            Matrix<double> rankMu = DenseMatrix.Create(_nDim, _nDim, 0.0);

            for (int k = 0; k < _popSize; k++)
            {
                rankMu += yk[k].OuterProduct(yk[k]) * wIo[k];
            }

            double weightSum = _weights.Sum();
            _C = _C * (1 + _c1 * deltaHSigma - _c1 - _cmu * weightSum) + rankOne * _c1 + rankMu * _cmu;

            if (_lrAdapt)
            {
                if (oldMean is null || oldSigmaMatrix is null || oldInvSqrtC is null)
                {
                    throw new InvalidOperationException("Learning rate adaptation requires previous state.");
                }

                LrAdaptation(oldMean, oldSigma, oldSigmaMatrix, oldInvSqrtC);
            }
        }

        public bool ShouldStop()
        {
            var (B, D) = EigenDecomposition();
            var diagC = _C.Diagonal().ToArray();

            if (_generation > _funhistTerm)
            {
                double max = double.NegativeInfinity;
                double min = double.PositiveInfinity;
                foreach (var v in _funhistValues)
                {
                    if (double.IsNaN(v))
                    {
                        continue;
                    }

                    if (v > max)
                    {
                        max = v;
                    }

                    if (v < min)
                    {
                        min = v;
                    }
                }

                if (max - min < _tolfun)
                {
                    return true;
                }
            }

            bool smallStd = diagC.All(v => _sigma * v < _tolx);
            bool smallPc = _pc.AsEnumerable().All(v => Math.Abs(_sigma * v) < _tolx);
            if (smallStd && smallPc)
            {
                return true;
            }

            if (_sigma * D.Max() > _tolxup)
            {
                return true;
            }

            for (int i = 0; i < _nDim; i++)
            {
                double delta = 0.2 * _sigma * Math.Sqrt(diagC[i]);
                if (_mean[i] == _mean[i] + delta)
                {
                    return true;
                }
            }

            int axisIndex = _generation % _nDim;
            double axisDelta = 0.1 * _sigma * D[axisIndex];
            bool noEffectAxis = true;
            for (int i = 0; i < _nDim; i++)
            {
                double proposed = _mean[i] + axisDelta * B[i, axisIndex];
                if (_mean[i] != proposed)
                {
                    noEffectAxis = false;
                    break;
                }
            }

            if (noEffectAxis)
            {
                return true;
            }

            double minD = D.Min();
            if (minD <= 0)
            {
                return true;
            }

            double conditionCov = D.Max() / minD;
            if (conditionCov > _tolconditioncov)
            {
                return true;
            }

            return false;
        }

        private (Matrix<double> B, Vector<double> D) EigenDecomposition()
        {
            if (_B != null && _D != null)
            {
                return (_B, _D);
            }

            _C = (_C + _C.Transpose()) * 0.5;
            var evd = _C.Evd(Symmetricity.Symmetric);
            var eigenValues = evd.D.Diagonal().ToArray();
            var dValues = eigenValues.Select(v => Math.Sqrt(v < EPS ? EPS : v)).ToArray();
            var diagVector = DenseVector.OfArray(dValues);
            var diagSq = DenseMatrix.OfDiagonalArray(dValues.Select(v => v * v).ToArray());

            var eigenVectors = evd.EigenVectors;
            _C = eigenVectors * diagSq * eigenVectors.Transpose();

            _B = eigenVectors;
            _D = diagVector;
            return (_B, _D);
        }

        private double[] SampleSolution()
        {
            var (B, D) = EigenDecomposition();
            var z = DenseVector.Create(_nDim, _ => NextGaussian());
            var Dy = D.PointwiseMultiply(z);
            var y = B * Dy;
            var x = _mean + y * _sigma;
            return x.ToArray();
        }

        private bool IsFeasible(double[] parameters)
        {
            if (_bounds == null)
            {
                return true;
            }

            for (int i = 0; i < _nDim; i++)
            {
                var low = _bounds[i, 0];
                var high = _bounds[i, 1];
                var value = parameters[i];
                if (value < low || value > high)
                {
                    return false;
                }
            }

            return true;
        }

        private double[] RepairInfeasibleParams(double[] parameters)
        {
            if (_bounds == null)
            {
                return parameters;
            }

            var repaired = parameters.ToArray();
            for (int i = 0; i < _nDim; i++)
            {
                double low = _bounds[i, 0];
                double high = _bounds[i, 1];
                if (repaired[i] < low)
                {
                    repaired[i] = low;
                }
                else if (repaired[i] > high)
                {
                    repaired[i] = high;
                }
            }

            return repaired;
        }

        private void LrAdaptation(
            Vector<double> oldMean,
            double oldSigma,
            Matrix<double> oldSigmaMatrix,
            Matrix<double> oldInvSqrtC)
        {
            var deltaMean = _mean - oldMean;
            var sigmaMatrix = _C.Clone().Multiply(_sigma * _sigma);
            var deltaSigma = sigmaMatrix - oldSigmaMatrix;

            var oldInvSqrtSigma = oldInvSqrtC.Multiply(1.0 / oldSigma);
            var locDeltaMean = oldInvSqrtSigma * deltaMean;
            var locDeltaSigmaMatrix = oldInvSqrtSigma * deltaSigma * oldInvSqrtSigma;
            var locDeltaSigma = DenseVector.OfArray(locDeltaSigmaMatrix.ToRowMajorArray()) / Math.Sqrt(2.0);

            _Emean = _Emean * (1 - _betaMean) + locDeltaMean * _betaMean;
            _ESigma = _ESigma * (1 - _betaSigma) + locDeltaSigma * _betaSigma;

            double locDeltaMeanNormSq = Math.Pow(locDeltaMean.L2Norm(), 2);
            double locDeltaSigmaNormSq = Math.Pow(locDeltaSigma.L2Norm(), 2);

            _Vmean = (1 - _betaMean) * _Vmean + _betaMean * locDeltaMeanNormSq;
            _VSigma = (1 - _betaSigma) * _VSigma + _betaSigma * locDeltaSigmaNormSq;

            double sqnormEmean = Math.Pow(_Emean.L2Norm(), 2);
            double denomMean = _Vmean - sqnormEmean;
            double hatSNRmean = denomMean == 0
                ? 0
                : (sqnormEmean - (_betaMean / (2 - _betaMean)) * _Vmean) / denomMean;

            double sqnormESigma = Math.Pow(_ESigma.L2Norm(), 2);
            double denomSigma = _VSigma - sqnormESigma;
            double hatSNRSigma = denomSigma == 0
                ? 0
                : (sqnormESigma - (_betaSigma / (2 - _betaSigma)) * _VSigma) / denomSigma;

            double beforeEtaMean = _etaMean;
            double relativeSNRmean = Math.Clamp(hatSNRmean / _alpha / _etaMean - 1, -1, 1);
            _etaMean *= Math.Exp(Math.Min(_gamma * _etaMean, _betaMean) * relativeSNRmean);

            double relativeSNRSigma = Math.Clamp(hatSNRSigma / _alpha / _etaSigma - 1, -1, 1);
            _etaSigma *= Math.Exp(Math.Min(_gamma * _etaSigma, _betaSigma) * relativeSNRSigma);

            _etaMean = Math.Min(_etaMean, 1.0);
            _etaSigma = Math.Min(_etaSigma, 1.0);

            _mean = oldMean + deltaMean * _etaMean;
            var sigmaMatrixUpdated = oldSigmaMatrix + deltaSigma * _etaSigma;

            var evd = sigmaMatrixUpdated.Evd(Symmetricity.Symmetric);
            var eigenValues = evd.D.Diagonal().Select(v => Math.Max(v, EPS)).ToArray();
            double logEigSum = eigenValues.Sum(Math.Log);
            _sigma = Math.Exp(logEigSum / (2.0 * _nDim));
            _sigma = Math.Min(_sigma, SIGMA_MAX);
            _C = sigmaMatrixUpdated.Multiply(1.0 / (_sigma * _sigma));

            _sigma *= beforeEtaMean / _etaMean;
            _B = null;
            _D = null;
        }

        private static bool IsValidBounds(double[,]? bounds, Vector<double> mean)
        {
            if (bounds is null)
            {
                return true;
            }

            if (bounds.GetLength(0) != mean.Count || bounds.GetLength(1) != 2)
            {
                return false;
            }

            for (int i = 0; i < mean.Count; i++)
            {
                if (bounds[i, 0] > mean[i] || mean[i] > bounds[i, 1])
                {
                    return false;
                }
            }

            return true;
        }

        private static double[,]? CloneBounds(double[,]? bounds)
        {
            if (bounds is null)
            {
                return null;
            }

            int rows = bounds.GetLength(0);
            int cols = bounds.GetLength(1);
            var clone = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    clone[i, j] = bounds[i, j];
                }
            }

            return clone;
        }

        private double NextGaussian()
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = 1.0 - _rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
