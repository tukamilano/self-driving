// SerialID: [e4a22a75-a938-4302-8b9a-d405c01db428]
using System.Collections.Generic;
using UnityEngine;
using System;

public class CarAgent : Agent
{
    [SerializeField] private int currentStep = 0;
    private int CurrentStep { get { return currentStep; } set { currentStep = value; } }

    private const int DefaultCurrentStepMax = 20000;
    private int currentStepMax = DefaultCurrentStepMax;
    private int CurrentStepMax => currentStepMax;

    [SerializeField] private int localStep = 0;
    private int LocalStep { get { return localStep; } set { localStep = value; } }

    private const int DefaultLocalStepMax = 1000;
    private int localStepMax = DefaultLocalStepMax;
    private int LocalStepMax => localStepMax;

    private const float DefaultPenalty = 1000f;

    [SerializeField] private float penalty = DefaultPenalty;
    private float Penalty => penalty > 0f ? penalty : DefaultPenalty;

    [SerializeField] private bool allowPlusReward = true;
    private bool AllowPlusReward => allowPlusReward;

    [SerializeField] private bool isLearning = true;
    private bool IsLearning => isLearning;

    [SerializeField] private bool backUpOnCollision = false;
    private bool BackUpOnCollision => backUpOnCollision;

    private Sensor[] Sensors { get; set; }
    private CarController Controller { get; set; }
    private Rigidbody CarRb { get; set; }
    private Vector3 StartPosition { get; set; }
    private Quaternion StartRotation { get; set; }
    private Vector3 LastPosition { get; set; }
    public float TotalDistance { get; set; }
    private int WaypointIndex { get; set; }
    private bool passLastPoint=false;
    // 次のWaypointの方向（グローバル座標）
    private Vector3 NextWaypointDirection = Vector3.forward;

    private void Awake() {
        CarRb = GetComponent<Rigidbody>();
        Controller = GetComponent<CarController>();
        Sensors = GetComponentsInChildren<Sensor>();
        isBattle=false;
    }

    private void OnValidate() {
        if (penalty <= 0f) {
            penalty = DefaultPenalty;
        }
    }

    public void Start() {
        StartPosition = transform.position;
        StartRotation = transform.rotation;
        CurrentStep = 0;
        LocalStep = 0;
        LastPosition = StartPosition;
        TotalDistance = 0;
    }

    public override void AgentReset() {
        transform.position = StartPosition;
        transform.rotation = StartRotation;

        Controller.GasInput = 0;
        Controller.SteerInput = 0;
        Controller.BrakeInput = 0;

        gameObject.SetActive(false);
        gameObject.SetActive(true);

        CurrentStep = 0;
        LocalStep = 0;
        TotalDistance = 0;
        LastPosition = StartPosition;

        WaypointIndex = 0;
    }

    /// <summary>
    /// 取得可能なエージェントの状態をすべて返す．
    /// </summary>
    /// <remarks>
    /// 返り値のリスト（状態）の中身は以下の通り．
    /// 
    /// | インデックス | 内容 |
    /// | --- | --- |
    /// | 0--4 | 前方の対壁センサー（Sensors_0_Wall）
    /// | 5--9 | 右方の対壁センサー（Sensors_1_Wall）
    /// | 10--14 | 左方の対壁センサー（Sensors_2_Wall）
    /// | 15--19 | 後方の対壁センサー（Sensors_3_Wall）
    /// | 20--24 | 前方の対車センサー（Sensors_0_Player）
    /// | 25--29 | 前方の対車センサー（Sensors_1_Player）
    /// | 30--34 | 前方の対車センサー（Sensors_2_Player）
    /// | 35--39 | 前方の対車センサー（Sensors_3_Player）
    /// | --- | --- |
    /// | 40--42 | 自車のローカル速度 |
    /// | --- | --- |
    /// | 43--45 | コース上の前方向ベクトル（次のWaypointの方向）
    /// | --- | --- |
    /// 
    /// </remarks>
    /// <returns>状態</returns>
    public override List<double> GetAllObservations() {
        var results = new List<double>();
        // センサー
        Array.ForEach(Sensors, sensor => {
            results.AddRange(sensor.Hits());
        });
        // 速度
        Vector3 local_v = CarRb.transform.InverseTransformDirection(CarRb.velocity);
        for(int i = 0; i < 3; i++) {
            results.Add(local_v[i] / 5.0f);
        }
        // 前方向
        Vector3 localNextDirection = CarRb.transform.InverseTransformDirection(NextWaypointDirection);
        for(int i = 0; i < 3; i++) {
            results.Add(localNextDirection[i]);
        }
        return results;
    }

    /// <summary>
    /// センサーの角度を変更する．
    /// </summary>
    /// <param name="config">センサーの角度のリスト</param>
    public override void SetAgentConfig(List<double> config)
    {
        base.SetAgentConfig(config);

        if(config == null) return;

        int configIndex = 0;
        foreach(Sensor sensor in Sensors)
        {
            int sensorIndex = 0;
            while(configIndex < config.Count && sensorIndex < sensor.Angles.Length)
            {
                sensor.Angles[sensorIndex] = (float)config[configIndex];
                sensorIndex++;
                configIndex++;
            }
        }
    }

    public override int GetState() {
        var stateDivide = 3;
        var results = new List<double>();
        var r = 0;
        Array.ForEach(Sensors, sensor => {
            results.AddRange(sensor.Hits());
        });

        // Sensors to use (up to 7).
        int[] indices = { 0, 1, 2, 3, 4, 40, 42 };

        List<double> filteredResult = new List<double>();

        foreach (int index in indices)
        {
            if (index >= 0 && index < results.Count)
            {
                filteredResult.Add(results[index]);
            }
        }

        for(int i = 0; i < filteredResult.Count; i++) {
            var v = Mathf.FloorToInt(Mathf.Lerp(0, stateDivide - 1, (float)filteredResult[i]));
            if(filteredResult[i] >= 0.99f) {
                v = stateDivide - 1;
            }
            r += (int)(v * Mathf.Pow(stateDivide, i));
        }
        var numStates = (int)Mathf.Pow(stateDivide, filteredResult.Count);
        int n;
        if(CarRb.velocity.magnitude < 10) { n = 0; }
        else if(CarRb.velocity.magnitude < 15) { n = 1; }
        else { n = 2; }
        r += numStates * n;
        return r;
    }

    public override List<double> CollectObservations() {
        // センサーの距離をリストに追加する
        var results = new List<double>();
        Array.ForEach(Sensors, sensor => {
            results.AddRange(sensor.Hits());
        });
        Vector3 local_v = CarRb.transform.InverseTransformDirection(CarRb.velocity);
        results.Add(local_v.x / 5.0f);
        results.Add(local_v.z / 5.0f);
        return results;
    }

    /*************編集ポイント***************
        センサ情報等、あるフレームにおける環境情報を取得するための関数
    */

    public override List<double> OriginalObservations(){
        //センサ情報を取得
        var results = new List<double>();
        Array.ForEach(Sensors, sensor => {
            results.AddRange(sensor.Hits());
        });
        /*
            必要に応じて追加で格納する情報を追加
        */
        results.Add(CarRb.velocity.magnitude);
        return results;
    }

    public override double[] ActionNumberToVectorAction(int ActionNumber) {
        var action = new double[3];
        var steering = 0.0d;
        var braking = 0.0d;
        if(ActionNumber % 6 == 1) {
            steering = 1d;
        }
        else if(ActionNumber % 6 == 2) {
            steering = -1d;
        }
        else if(ActionNumber % 6 == 3) {
            steering = 0.5d;
        }
        else if(ActionNumber % 6 == 4) {
            steering = -0.5d;
        }
        else if(ActionNumber % 6 == 5) {
            braking = 0.5d;
        }

        var gasInput = 0.5d;
        action[0] = steering;
        action[1] = gasInput;
        action[2] = braking;
        return action;
    }

    public override void AgentAction(double[] vectorAction, bool inReverse) {
        CurrentStep++;
        LocalStep++;
        TotalDistance += (transform.position - LastPosition).magnitude;

        if(IsLearning) {
            if(CurrentStep > CurrentStepMax) {
                CompleteEpisode(TotalDistance);
                return;
            }

            if(LocalStep > LocalStepMax) {
                CompleteEpisode(TotalDistance);
                return;
            }
        } else if(isBattle) {
              // バトル中はLocalStepMax以内に次のWaypointに到達できなければGame Over
              if(LocalStep > LocalStepMax) {
                  agentExecutor.GameOver(agentIndex);
                  return;
                }
            }

        var steering = Mathf.Clamp((float)vectorAction[0], -1.0f, 1.0f);
        float gasInput = 0.0f;
        if (!inReverse) {
            gasInput = Mathf.Clamp((float)vectorAction[1], 0.0f, 1.0f);
        } else {
            gasInput = Mathf.Clamp((float)vectorAction[1], -0.3f, 0.0f);
        }
        var braking = Mathf.Clamp((float)vectorAction[2], 0.0f, 1.0f);

        Controller.SteerInput = steering;
        Controller.GasInput = gasInput;
        Controller.BrakeInput = braking;
        LastPosition = transform.position;
    }

    public override void GoStraight(){
        var gasInput = Mathf.Clamp(1.0f, 0.5f, 1.0f);
        Controller.GasInput = gasInput;
        LastPosition = transform.position;
    }

    public override float GetDistance()
    {
        return TotalDistance;
    }

    /// <summary>
    /// 衝突時に呼び出されるコールバック
    /// </summary>
    /// <param name="collision"></param>
    public void OnCollisionEnter(Collision collision) {
        if(collision.gameObject.tag == "wall") {
            if (BackUpOnCollision) {
                StartBackingUp();
            } else {
                CompleteEpisode(TotalDistance);
            }
        }
    }

    public void OnTriggerEnter(Collider other) {
        var waypoint = other.GetComponent<Waypoint>();
        if(waypoint == null) {
            return;
        }

        //逆走した時
        if (!BackUpOnCollision) {
            bool reverseRunFromStartPosition = waypoint.Index>WaypointIndex+1;
            bool reverseRunFromOtherPosition = waypoint.Index<=WaypointIndex;
            if( reverseRunFromOtherPosition|| reverseRunFromStartPosition){
                CompleteEpisode(TotalDistance);
                return;
            }
        }

        WaypointIndex = waypoint.Index;
        if(waypoint.IsLast) {
            WaypointIndex = 0;
            passLastPoint=true;
        }
        if(isBattle && WaypointIndex==1 && passLastPoint==true){
            agentExecutor.Win(agentIndex);
        }
        LocalStep = 0;

        NextWaypointDirection = waypoint.NextDirection;
    }

    public override void Stop() {
        CarRb.velocity = Vector3.zero;
        CarRb.angularVelocity = Vector3.zero;
        Controller.Stop();
    }

    private void CompleteEpisode(float baseReward, bool applyPenalty = true) {
        var finalReward = baseReward;
        if(applyPenalty) {
            finalReward -= Penalty;
        }

        if(finalReward > 0 && !AllowPlusReward) {
            finalReward = 0;
        }

        SetReward(finalReward);
        Done();
    }
}
