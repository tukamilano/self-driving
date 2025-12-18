using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "BattleBrain/NNBattleBrain")]
public class NNBattleBrain : BattleBrain
{
    [SerializeField] private TextAsset brainData = null;

    [HideInInspector] public NNBrain brain;

    public override void Initialize() {
        Debug.Log($"[NNBattleBrain] brainData: name={brainData?.name ?? "<null>"}, length={brainData?.text?.Length.ToString() ?? "<null>"}\n{brainData?.text ?? "<null>"}");
        base.Initialize();
        try
        {
            brain = NNBrain.Load(brainData);
        }
        catch (Exception ex)
        {
            throw new Exception("Failed to load NNBrain.", ex);
        }
    }

    /// <summary>
    /// `BattleBrain`の`GetAction()`関数をoverrideする.
    /// 入力のうち0, 1, 2, 3, 4, 40, 42番目のみを使う.
    /// </summary>
    public override double[] GetAction(List<double> observation)
    {
        return brain.GetAction(RearrangeObservation(observation, new List<int>{0, 1, 2, 3, 4, 40, 42}));
    }
}
