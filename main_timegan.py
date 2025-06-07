"""
Main script for TimeGAN experiments (TF2-compatible)
"""

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

def load_custom_npz_data(path):
    data = np.load(path)['data']
    return list(data)


def main(args):
    """Main function for timeGAN experiments.

    Args:
      - data_name: sine, stock, or energy
      - seq_len: sequence length
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation

    Returns:
      - ori_data: original data
      - generated_data: generated synthetic data
      - metric_results: discriminative and predictive scores
    """
    # Data loading
    if args.data_name == 'mimic_mtgan':
        ori_data = load_custom_npz_data('mimic_mtgan_for_timegan.npz')
    elif args.data_name == 'stock' or args.data_name == 'energy':
        ori_data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, args.seq_len, dim)

    print(f"{args.data_name} dataset is ready.")

    # Set network parameters
    parameters = {
        'module': args.module,
        'hidden_dim': args.hidden_dim,
        'num_layer': args.num_layer,
        'iterations': args.iteration,
        'batch_size': args.batch_size
    }

    generated_data = timegan(ori_data, parameters)
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = {}

    # 1. Discriminative Score
    discriminative_score = [
        discriminative_score_metrics(ori_data, generated_data)
        for _ in range(args.metric_iteration)
    ]
    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = [
        predictive_score_metrics(ori_data, generated_data)
        for _ in range(args.metric_iteration)
    ]
    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, generated_data, 'pca')
    visualization(ori_data, generated_data, 'tsne')

    print(metric_results)
    return ori_data, generated_data, metric_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['sine', 'stock', 'energy', 'mimic_mtgan'], default='stock', type=str)
    parser.add_argument('--seq_len', default=24, type=int)
    parser.add_argument('--module', choices=['gru', 'lstm', 'lstmLN'], default='gru', type=str)
    parser.add_argument('--hidden_dim', default=24, type=int)
    parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--iteration', default=50000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--metric_iteration', default=10, type=int)

    args = parser.parse_args()
    ori_data, generated_data, metrics = main(args)
