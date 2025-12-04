using System.Collections.Generic;
using UnityEngine;

// // 自分でAIを実装する場合 -------------------------------------------------------------
// [CreateAssetMenu(menuName = "BattleBrain/FujitaBattleBrain")]
// public class FujitaBattleBrain : BattleBrain {
// public virtual void Initialize() { }
//
//     public override void Initialize() {
//         // 最初に呼ばれる関数. 初期化等が必要な場合はこちらに記述する.
//     }
//
//     public override double[] GetAction(List<double> observation) {
//         // `observation`に基づき`action`を決定する関数を書く.
//         // 入力・出力の仕様の詳細はWikiに掲載.
//         return some_action_based_on_observation;
//     }
//
// }
// // ---------------------------------------------------------------------------------

// ニューラルネットワークのAIを使う場合 --------------------------------------------------
[CreateAssetMenu(menuName = "BattleBrain/FujitaBattleBrain")]
public class FujitaBattleBrain : NNBattleBrain
{
    /// <summary>
    /// 使用するセンサーの設定.
    /// </summary>
    public bool[] selectedInputs = new bool[46];

    /// <summary>
    /// `NNBattleBrain`の`GetAction()`をoverrideする.
    /// `NNBattleBrain`の`GetAction()`は0, 1, 2, 3, 4, 40, 42番目のセンサーを使うが、
    /// `FujitaBattleBrain`ではInspectorのSelected Inputsで選択したセンサーの値を使う.
    /// </summary>
    public override double[] GetAction(List<double> observation)
    {
        return brain.GetAction(RearrangeObservation(observation, CreateSelectedInputsList(selectedInputs)));
    }

    /// <summary>
    /// `RearrangeObservation()`に渡すため, 選択されたセンサーのインデックスのリストを作る.
    /// {0, 1, 1, 0, 1} -> {1, 2, 4}
    /// </summary>
    public List<int> CreateSelectedInputsList(bool[] selectedInputs) {
        List<int> selectedInputsList = new List<int>();
        for (int i = 0; i < selectedInputs.Length; i++)
        {
            if (selectedInputs[i]) selectedInputsList.Add(i);
        }
        return selectedInputsList;
    }
}
// ---------------------------------------------------------------------------------

// // Q学習のAIを使う場合 ----------------------------------------------------------------
// [CreateAssetMenu(menuName = "BattleBrain/FujitaBattleBrain")]
// public class FujitaBattleBrain : QBattleBrain
// {
//     /// <summary>
//     /// `QBattleBrain`の`GetAction()`をoverrideする.
//     /// </summary>
//     public override double[] GetAction(List<double> observation)
//     {
//         int state = GetState(observation);
//         int actionNumber = brain.GetActionWithoutEpsilon(state);
//         double[] action = ActionNumberToVectorAction(actionNumber);
//         return action;
//     }
// }
// // ---------------------------------------------------------------------------------