#!/usr/bin/env python3
"""
PopulationDump.jsonから世代ごとの適合度をプロットするスクリプト
- 最良適合度（1位個体）
- 平均適合度
- 第二位個体の適合度
"""

import json
import matplotlib.pyplot as plt
import os
import argparse

# 日本語フォントの設定（必要に応じて）
# plt.rcParams['font.family'] = 'DejaVu Sans'

def load_population_dump(json_path):
    """PopulationDump.jsonを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_fitness_data(data):
    """世代ごとの適合度データを抽出"""
    generations = []
    best_fitness = []
    second_best_fitness = []
    avg_fitness = []
    
    for gen_data in data['generations']:
        gen_num = gen_data['generation']
        individuals = gen_data['individuals']
        
        if len(individuals) == 0:
            continue
        
        # 最良適合度（1位個体）
        best = individuals[0]['fitness']
        
        # 第二位個体の適合度
        second_best = individuals[1]['fitness'] if len(individuals) > 1 else best
        
        # 平均適合度
        avg = sum(ind['fitness'] for ind in individuals) / len(individuals)
        
        generations.append(gen_num)
        best_fitness.append(best)
        second_best_fitness.append(second_best)
        avg_fitness.append(avg)
    
    return generations, best_fitness, second_best_fitness, avg_fitness

def plot_fitness(generations, best_fitness, second_best_fitness, avg_fitness, output_path):
    """適合度をプロット"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(generations, best_fitness, label='Best Fitness (1st)', 
             marker='o', markersize=4, linewidth=2, color='#2E86AB')
    plt.plot(generations, avg_fitness, label='Average Fitness', 
             marker='s', markersize=4, linewidth=2, color='#A23B72')
    plt.plot(generations, second_best_fitness, label='2nd Best Fitness', 
             marker='^', markersize=4, linewidth=2, color='#F18F01')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Fitness Evolution by Generation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='PopulationDump.jsonから適合度グラフを生成')
    parser.add_argument('-o', '--output', type=str,
                       help='出力PNGファイルのパス（デフォルト: スクリプトと同じディレクトリのfitness_plot.png）')
    
    args = parser.parse_args()
    
    # スクリプトと同じディレクトリのPopulationDump.jsonを読み込む
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'PopulationDump.json')
    
    # 出力ファイルパスの決定
    if args.output:
        # 相対パスの場合は絶対パスに変換
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            # 相対パスの場合、現在の作業ディレクトリを基準にする
            output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(script_dir, 'fitness_plot.png')
    
    if not os.path.exists(json_path):
        print(f"エラー: {json_path} が見つかりません")
        return
    
    print(f"JSONファイルを読み込み中: {json_path}")
    data = load_population_dump(json_path)
    
    print("適合度データを抽出中...")
    generations, best_fitness, second_best_fitness, avg_fitness = extract_fitness_data(data)
    
    print(f"世代数: {len(generations)}")
    print(f"最終世代の最良適合度: {best_fitness[-1] if best_fitness else 'N/A'}")
    print(f"最終世代の平均適合度: {avg_fitness[-1] if avg_fitness else 'N/A'}")
    
    print("グラフを作成中...")
    plot_fitness(generations, best_fitness, second_best_fitness, avg_fitness, output_path)

if __name__ == '__main__':
    main()

