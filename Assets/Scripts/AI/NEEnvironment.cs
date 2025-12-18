// SerialID: [77a855b2-f53d-4b80-9c94-c40562952b74]
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using System;
using System.Linq;
using System.IO;
using Practice1;

#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

public class NEEnvironment : Environment
{
    private int tournamentSelection = 10;
    private int TournamentSelection { get { return tournamentSelection; } }

    private int eliteSelection = 4;
    private int EliteSelection { get { return eliteSelection; } }

    [SerializeField] public bool[] selectedInputs = new bool[46];
    [SerializeField] public List<double> sensorAngleConfig = new List<double>();

    private int InputSize { get; set; }

    private List<int> SelectedInputsList { get; set; }

    [SerializeField] private int hiddenSize = 16;
    private int HiddenSize { get { return hiddenSize; } }

    [SerializeField] private int hiddenLayers = 1;
    private int HiddenLayers { get { return hiddenLayers; } }

    private int outputSize = 2;
    private int OutputSize { get { return outputSize; } }

    private double paramCount;
    private int totalPopulation;

    private int TotalPopulation { get { return totalPopulation; } }

    [SerializeField] private int nAgents = 8;
    private int NAgents { get { return nAgents; } }

    [Header("Agent Prefab"), SerializeField] private GameObject gObject = null;
    private GameObject GObject => gObject;

    [SerializeField] private bool isChallenge4 = false;
    private bool IsChallenge4 { get { return isChallenge4; } }

    [Header("UI References"), SerializeField] private Text populationText = null;
    private Text PopulationText { get { return populationText; } }

    private float GenBestRecord { get; set; }

    private float SumReward { get; set; }
    private float AvgReward { get; set; }

    private List<NNBrain> Brains { get; set; } = new List<NNBrain>();
    private List<GameObject> GObjects { get; } = new List<GameObject>();
    private List<Agent> Agents { get; } = new List<Agent>();
    private int Generation { get; set; }

    private float BestRecord { get; set; }
    private float BestOverallReward { get; set; } = float.NegativeInfinity;
    private NNBrain BestOverallBrain { get; set; }
    private NNBrain BestLastGenerationBrain { get; set; }

    private List<AgentPair> AgentsSet { get; } = new List<AgentPair>();
    private Queue<NNBrain> CurrentBrains { get; set; }

    private List<Obstacle> Obstacles { get; } = new List<Obstacle>();

    [SerializeField] private int maxGenerations = 50;
    private int MaxGenerations { get { return maxGenerations; } }

    private bool use_CMA = true;

    [SerializeField] private string cmaSeedFileName = "";

    private CMA cmaOptimizer;
    private bool cmaInitialized;

    private const string PopulationDumpPath = "Assets/LearningData/NE/Record/PopulationDump.json";
    private const string CmaSeedDirectory = "Assets/LearningData/NE";
    private bool populationDumpInitialized;
    private bool trainingComplete;

    void Start() {
        InitializePopulationDump();
        // Calculate and set input size.
        int sensorCount = 0;
        foreach (bool value in selectedInputs)
        {
            if (value) sensorCount++;
        }
        InputSize = sensorCount;

        // Calculate and set sensors list.
        List<int> selectedInputsList = new List<int>();
        for (int i = 0; i < selectedInputs.Length; i++)
        {
            if (selectedInputs[i]) selectedInputsList.Add(i);
        }
        SelectedInputsList = selectedInputsList;

        // Derive population parameters now that InputSize is known.
        paramCount = InputSize * HiddenSize
                     + HiddenSize * HiddenSize * (HiddenLayers - 1)
                     + 4 * HiddenSize
                     + 3;
        var logArg = Math.Max(paramCount, 1.0);
        totalPopulation = 4 + (int)(3 * Math.Floor(Math.Log(logArg)));
        if (totalPopulation <= 0)
        {
            totalPopulation = 4; // fallback to minimal population when parameters underflow
        }

        // Initialize brain.
        for(int i = 0; i < TotalPopulation; i++) {
            Brains.Add(new NNBrain(InputSize, HiddenSize, HiddenLayers, OutputSize));
        }

        if (use_CMA) {
            try {
                InitializeCmaPopulation();
            }
            catch (Exception ex) {
                Debug.LogError($"Failed to initialize CMA optimizer: {ex.Message}\n{ex.StackTrace}");
                use_CMA = false;
            }
        }

        for(int i = 0; i < NAgents; i++) {
            var obj = Instantiate(GObject);
            obj.SetActive(true);
            GObjects.Add(obj);
            Agents.Add(obj.GetComponent<Agent>());
        }
        
        foreach(Agent agent in Agents)
        {
            agent.SetAgentConfig(sensorAngleConfig);
        }

        BestRecord = -9999;
        SetStartAgents();
        if (IsChallenge4) {
            Obstacles.AddRange(FindObjectsOfType<Obstacle>());
        }
    }

    void SetStartAgents() {
        CurrentBrains = new Queue<NNBrain>(Brains);
        AgentsSet.Clear();
        var size = Math.Min(NAgents, TotalPopulation);
        for(var i = 0; i < size; i++) {
            AgentsSet.Add(new AgentPair {
                agent = Agents[i],
                brain = CurrentBrains.Dequeue()
            });
        }
    }

    void FixedUpdate() {
        if (trainingComplete) {
            return;
        }

        foreach(var pair in AgentsSet.Where(p => !p.agent.IsDone)) {
            AgentUpdate(pair.agent, pair.brain);
        }

        AgentsSet.RemoveAll(p => {
            if(p.agent.IsDone) {
                p.agent.Stop();
                p.agent.gameObject.SetActive(false);
                float r = p.agent.Reward;
                BestRecord = Mathf.Max(r, BestRecord);
                GenBestRecord = Mathf.Max(r, GenBestRecord);
                p.brain.Reward = r;
                if (r > BestOverallReward)
                {
                    BestOverallReward = r;
                    BestOverallBrain = new NNBrain(p.brain);
                }
                SumReward += r;
            }
            return p.agent.IsDone;
        });

        if(CurrentBrains.Count == 0 && AgentsSet.Count == 0) {
            SetNextGeneration();
        }
        else {
            SetNextAgents();
        }
    }

    private void AgentUpdate(Agent a, NNBrain b) {
        var observation = a.GetAllObservations();
        var rearranged = RearrangeObservation(observation, SelectedInputsList);
        var action = b.GetAction(rearranged);
        a.AgentAction(action, false);
    }

    private void SetNextAgents() {
        int size = Math.Min(NAgents - AgentsSet.Count, CurrentBrains.Count);
        for(var i = 0; i < size; i++) {
            var nextAgent = Agents.First(a => a.IsDone);
            var nextBrain = CurrentBrains.Dequeue();
            nextAgent.Reset();
            AgentsSet.Add(new AgentPair {
                agent = nextAgent,
                brain = nextBrain
            });
        }
        UpdateText();
    }

    private void SetNextGeneration() {
        if (Generation >= MaxGenerations) {
            CompleteTraining();
            return;
        }

        AvgReward = SumReward / TotalPopulation;
        GenPopulation();
        SumReward = 0;
        GenBestRecord = -9999;
        Agents.ForEach(a => a.Reset());
        SetStartAgents();
        UpdateText();
    }

    private static int CompareBrains(Brain a, Brain b) {
        if(a.Reward > b.Reward) return -1;
        if(b.Reward > a.Reward) return 1;
        return 0;
    }

    private void GenPopulation() {
        var children = new List<NNBrain>();
        var bestBrains = Brains.ToList();

        // Elite selection
        bestBrains.Sort(CompareBrains);
        if(EliteSelection > 0) {
            children.AddRange(bestBrains.Take(EliteSelection));
        }

        DumpPopulationData(bestBrains);

        if(!use_CMA)
        {
            while(children.Count < TotalPopulation) {
                var tournamentMembers = Brains.AsEnumerable().OrderBy(x => Guid.NewGuid()).Take(tournamentSelection).ToList();
                tournamentMembers.Sort(CompareBrains);
                children.Add(tournamentMembers[0].Mutate(Generation));
                children.Add(tournamentMembers[1].Mutate(Generation));
            }
            Brains = children;
        } else
        {
            if (!cmaInitialized) {
                InitializeCmaPopulation();
            }

            var solutions = new List<(double[] Parameters, double Value)>(Brains.Count);
            foreach (var brain in Brains) {
                solutions.Add((brain.ToDNA(), -brain.Reward));
            }

            cmaOptimizer.Tell(solutions);
            ApplyCmaSamples();
        }

        var lastGenBest = bestBrains
            .OrderByDescending(b => b.Reward)
            .FirstOrDefault();
        BestLastGenerationBrain = lastGenBest != null ? new NNBrain(lastGenBest) : null;

        Generation++;
    }

    protected List<double> RearrangeObservation(List<double> observation, List<int> indexesToUse)
    {
        if(observation == null || indexesToUse == null) return null;

        List<double> rearranged = new List<double>();
        foreach(int index in indexesToUse)
        {
            if(index >= observation.Count)
            {
                rearranged.Add(0);
                continue;
            }
            rearranged.Add(observation[index]);
        }

        return rearranged;
    }

    private void UpdateText() {
        PopulationText.text = "Population: " + (TotalPopulation - CurrentBrains.Count) + "/" + TotalPopulation
            + "\nGeneration: " + (Generation + 1)
            + "\nBest Record: " + BestRecord
            + "\nBest this gen: " + GenBestRecord
            + "\nAverage: " + AvgReward;
    }

    private void CompleteTraining() {
        trainingComplete = true;
        Agents.ForEach(a => a.Stop());
#if UNITY_EDITOR
        SaveFinalBestBrain();
#endif
    }

    private struct AgentPair
    {
        public NNBrain brain;
        public Agent agent;
    }

    private void InitializePopulationDump() {
        if (populationDumpInitialized) return;
        File.WriteAllText(PopulationDumpPath, string.Empty);
        populationDumpInitialized = true;
    }

    private void DumpPopulationData(List<NNBrain> brains) {
        InitializePopulationDump();
        var dump = LoadPopulationDump();
        var generationDump = new GenerationDump {
            generation = Generation,
            individuals = brains.Select(b => new IndividualDump {
                parameters = b.ToDNA().ToList(),
                fitness = b.Reward
            }).ToList()
        };
        dump.generations.Add(generationDump);
        var json = JsonUtility.ToJson(dump, true);
        File.WriteAllText(PopulationDumpPath, json);
    }

    private PopulationDump LoadPopulationDump() {
        if (!File.Exists(PopulationDumpPath)) {
            return new PopulationDump();
        }
        var json = File.ReadAllText(PopulationDumpPath);
        if (string.IsNullOrEmpty(json)) {
            return new PopulationDump();
        }
        var dump = JsonUtility.FromJson<PopulationDump>(json);
        return dump ?? new PopulationDump();
    }

    [Serializable]
    private class PopulationDump {
        public List<GenerationDump> generations = new List<GenerationDump>();
    }

    [Serializable]
    private class GenerationDump {
        public int generation;
        public List<IndividualDump> individuals = new List<IndividualDump>();
    }

    [Serializable]
    private class IndividualDump {
        public List<double> parameters;
        public float fitness;
    }

    private void InitializeCmaPopulation() {
        if (Brains == null || Brains.Count == 0) {
            throw new InvalidOperationException("Brains must be initialized before CMA initialization.");
        }

        var sampleDna = Brains[0].ToDNA();
        var initialMean = new double[sampleDna.Length];

        var seedPath = ResolveCmaSeedPath();
        try
        {
            if (File.Exists(seedPath))
            {
                var json = File.ReadAllText(seedPath);
                if (!string.IsNullOrEmpty(json))
                {
                    var seedBrain = JsonUtility.FromJson<NNBrain>(json);
                    if (seedBrain != null && seedBrain.ParameterCount == sampleDna.Length)
                    {
                        initialMean = seedBrain.ToDNA();
                    }
                    else
                    {
                        Debug.LogWarning($"Seed brain parameter mismatch for {seedPath}; using default CMA mean.");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Failed to load CMA seed from {seedPath}: {ex.Message}");
            initialMean = new double[sampleDna.Length];
        }

        var sigma = string.IsNullOrEmpty(cmaSeedFileName) ? 0.5 : 0.1;
        cmaOptimizer = new CMA(initialMean, sigma, populationSize: TotalPopulation);
        ApplyCmaSamples();
        cmaInitialized = true;
    }

    private string ResolveCmaSeedPath()
    {
        var fileStem = string.IsNullOrEmpty(cmaSeedFileName)
            ? SceneManager.GetActiveScene().name
            : cmaSeedFileName;
        return Path.Combine(CmaSeedDirectory, fileStem + ".json");
    }

    private void ApplyCmaSamples() {
        if (cmaOptimizer == null) {
            throw new InvalidOperationException("CMA optimizer is not initialized.");
        }

        for (int i = 0; i < Brains.Count; i++) {
            var candidate = cmaOptimizer.Ask();
            Brains[i].FromDNA(candidate);
            Brains[i].Reward = 0f;
        }
    }
#if UNITY_EDITOR
    private void SaveFinalBestBrain()
    {
        if (BestOverallBrain == null && BestLastGenerationBrain == null)
        {
            return;
        }

        var sceneName = EditorSceneManager.GetActiveScene().name;
        if (string.IsNullOrEmpty(sceneName))
        {
            return;
        }

        if (BestOverallBrain != null)
        {
            var overallPath = string.Format("Assets/LearningData/NE/{0}.json", sceneName);
            BestOverallBrain.Save(overallPath);
        }

        if (BestLastGenerationBrain != null)
        {
            var lastGenPath = string.Format("Assets/LearningData/NE/{0}_LastGen.json", sceneName);
            BestLastGenerationBrain.Save(lastGenPath);
        }
    }
#endif
}
