import numpy as np

import matplotlib.pyplot as plt 
import japanize_matplotlib

class BOOTSTRAP:
    
    def __init__(self, hole_values: np.ndarray, score_function):
        '''
        Args: 
            hole_values[np.ndarray]: 母集団とみなす集合
        '''
        print('\n--- make BOOTSTRAP instance ---\n')
        self.hole_values: np.ndarray = hole_values 
        self.score_function: function = score_function
        self.sampling_score_array: np.ndarray = None



    def calc_sampling_scores(self, sample_size: int, sampling_trial: int, is_there_duplicate: bool=True, use_only_unsampled_to_sample_score=False) -> np.ndarray:
        '''標本サイズ[sample_size]のサンプリングを実施した際のスコア分布を計算する関数。
        [sample_trial]回の試行を行い、その結果をnumpy配列として sampling_scores に入れて返す。

        Args: 

            sample_size[int]: 一度のサンプリングで抽出する標本サイズ

            sampling_trial[int]: サンプリングの試行回数

            is_there_duplicate[bool]: サンプリング時に重複を許すか(True: 重複あり, False: 重複なし)

            use_only_unsampled_to_scoring[bool]: 全体スコア評価時に、サンプリングされなかったデータのみを使用するか(True: [values]からサンプリングされなかったデータのみ使用, False: [values]全体を使用)

        Returns: 

            sampling_scores[np.ndarray]: スコア分布
        '''

        sampling_scores = []


        # 全体データとして、サンプリングされなかったデータを渡す（非推奨）
        if use_only_unsampled_to_sample_score:

            # 試行回数：sampling_trial
            for trial in range(sampling_trial):
                chosen_indices = np.random.choice(self.hole_values.shape[0], size=sample_size, replace=True)
                chosen_sample = self.hole_values[chosen_indices, :]
                not_chosen_indices = np.setdiff1d(np.arange(self.hole_values.shape[0]), chosen_indices)
                not_chosen_sample = self.hole_values[not_chosen_indices, :]
                tmp_score = self.score_function(hole_values=not_chosen_sample, sample_values=chosen_sample)
                sampling_scores.append(tmp_score)

        # 全体データとして、[self.hole_values]内の全てを渡す（推奨・正式）
        else:
            for trial in range(sampling_trial):
                chosen_indices = np.random.choice(self.hole_values.shape[0], size=sample_size, replace=False)
                chosen_sample = self.hole_values[chosen_indices, :]
                tmp_score = self.score_function(hole_values=self.hole_values, sample_values=chosen_sample)
                sampling_scores.append(tmp_score)

        sampling_scores = np.array(sampling_scores)
        self.sampling_score_array = sampling_scores

        return sampling_scores







    def plot_hole_values(self, index:int=0, bins:int=100):
        '''
        母集団[self.hole_values]のヒストグラムを作成

        Args: 
            index[int]: ヒストグラムを作成する変数のindex (n次元データから1次元分だけ抜き出してヒストグラムを作成する)

            bins[int]: ヒストグラムのbin数
        '''
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)
        ax.hist(self.hole_values[:, index], bins=bins)
        ax.set_xlabel('x')
        ax.set_ylabel('データ数')
        ax.grid()
        return fig
    


    def plot_hole_cumulative_distribution(self, index:int=0):
        '''
        母集団[self.hole_values]の累積分布関数を作成

        Args: 
            index[int]: 累積分布関数を作成する変数のindex (n次元データから1次元分だけ抜き出して累積分布関数を作成)
        '''

        x = self.hole_values[:, index].reshape([1, -1])
        x = np.concatenate([x, x], axis=1)[0]
        x = np.sort(x)

        vec_size = self.hole_values.shape[0]
        y = []
        for i in range(vec_size):
            y += [(i)/vec_size, (i+1)/vec_size]
        
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('累積分布')
        ax.grid()
        return fig
    

    def plot_sampling_scores(self, bins:int=100):
        '''
        sampling_scores のヒストグラムを作成
        
        Args: 
            bins[int]: ヒストグラムのbin数
        '''
        bias = self.sampling_score_array.mean()
        variance = self.sampling_score_array.var()

        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)
        ax.hist(self.sampling_score_array, bins=bins)
        ax.set_xlabel('score')
        ax.set_ylabel('回数')
        ax.grid()

        
        ax.text(0.95, 0.95, f'Bias(平均): {bias:.5f}\nVariance(分散): {variance:.5f}', va='top', ha='right', transform=ax.transAxes)
        return fig