import numpy as np
import matplotlib.pyplot as plt

from module.bootstrap import BOOTSTRAP




### def ############################
DISTRIBUTION_NAME = 'normal__loc5_scale5'
SAMPLE_SIZE = 50
SAMPLING_TRIAL = 10000

def make_true_distribution():
    '''
    Args: 
    Returns: 
        - true_values[np.ndarray]: 全体の分布
    '''
    # 今回は正規分布に従う乱数を100万個作成
    true_values = np.random.normal( 
        loc=5, # 平均
        scale=5, # 標準偏差
        size=1000000, # 出力サイズ（100万個）
    )
    true_values = true_values.reshape([-1, 1])

    return true_values


def score_function(hole_values: np.ndarray, sample_values: np.ndarray):
    '''
    Args: 
        hole_values[np.ndarray]: 全体(母集団)のデータ
        sample_values[np.ndarray]: 標本データ
    Returns:
        score[float]: スコア
    '''

    hole_mean = hole_values.mean()
    sample_mean = sample_values.mean()
    score = sample_mean - hole_mean

    return score

####################################




def main():

    values = make_true_distribution()
    bootstrap = BOOTSTRAP(values, score_function)

    # 全体分布のヒストグラムを作成
    values_figure = bootstrap.plot_hole_values(index=0, bins=100)
    values_figure.savefig(f'./result/fig/hole_values_hist/{DISTRIBUTION_NAME}__samplesize{SAMPLE_SIZE}__sampletrial{SAMPLING_TRIAL}.png')
    plt.close()

    # 全体分布の累積分布関数を作成
    cumulative_figure = bootstrap.plot_hole_cumulative_distribution(index=0)
    cumulative_figure.savefig(f'./result/fig/hole_cumulative_dist/{DISTRIBUTION_NAME}__samplesize{SAMPLE_SIZE}__sampletrial{SAMPLING_TRIAL}.png')
    plt.close()

    # ブートストラップ法によって、スコア分布を作成
    bootstrap.calc_sampling_scores(sample_size=SAMPLE_SIZE, sampling_trial=SAMPLING_TRIAL, is_there_duplicate=True, use_only_unsampled_to_sample_score=False)

    sampling_scores_figure = bootstrap.plot_sampling_scores(bins=100)
    sampling_scores_figure.savefig(f'./result/fig/sampling_scores_hist/{DISTRIBUTION_NAME}__samplesize{SAMPLE_SIZE}__sampletrial{SAMPLING_TRIAL}.png')
    plt.close()
    

if __name__ == '__main__':
    main()