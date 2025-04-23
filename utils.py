import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest, t

def plot_tstudent_vs_normal(df):

    sigma = 1.00
    x = np.linspace(-5, 5, 500)
    
    pdf_normal = norm.pdf(x, loc=0, scale=sigma)
    pdf_t = t.pdf(x, df=df)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf_normal, label=f'Normal(0, {sigma}²)', linewidth=2)
    plt.plot(x, pdf_t, label=f't-distribution (df={df})', linestyle='--', linewidth=2)
    
    plt.title(f'Comparison: t-distribution (df={df}) vs Normal')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def gen_and_plot_population_distribution(size=500_000, scale1=2, scale2=1):

    np.random.seed(42)  # For reproducibility

    # "crazy" bimodal + skewed population
    population = np.concatenate([
        np.random.exponential(scale=scale1, size=size // 2),
        np.random.normal(loc=10, scale=scale2, size=size // 2)
    ])

    # Populaiton stats
    mu = np.mean(population)
    sigma = np.std(population)
    population_variance = np.var(population)

    plt.figure(figsize=(10, 5))
    sns.histplot(population, bins=100, kde=True, stat="density", color='gray')
    plt.title("Population Distribution (Crazy)")
    plt.axvline(mu, color='red', linestyle='--', label=f"μ = {mu:.2f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    plt.text(0.99, 0.88, f"Variance (σ²) = {population_variance:.2f}",
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

    return population

def plot_one_sample(population, size=1000):
 
    mu = np.mean(population)
    population_variance = np.var(population)
   
    one_sample = np.random.choice(population, size=size, replace=False)
    sample_mean = np.mean(one_sample)
    sample_variance = np.var(one_sample, ddof=1)

    
    plt.figure(figsize=(10, 5))
    sns.histplot(one_sample, bins=40, kde=True, stat="density")

    plt.axvline(mu, color='red', linestyle='--', label=f'Population Mean μ = {mu:.2f}')
    plt.axvline(sample_mean, color='blue', linestyle='--', label=rf'Sample Mean $\bar{{x}}$ = {sample_mean:.2f}')

    plt.title("Sample Histogram")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.xlim(0, 15)
    plt.legend()

    # Add text with variances
    plt.text(0.30, 0.95, f"Population Variance σ² = {population_variance:.2f}\nSample Variance s² = {sample_variance:.2f}", 
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()


def plot_hypothesis_test(obs_val, pop_std, mean_diff, alpha=0.05):
    mu_0 = 0
    mu_a = mu_0 + mean_diff
    sigma = pop_std
    cohen_effect_size = mean_diff / sigma

    x = np.linspace(-5, 8, 1000)

    pdf_H0 = norm.pdf(x, mu_0, sigma)
    pdf_Ha = norm.pdf(x, mu_a, sigma)

    crit_val = norm.ppf(1 - alpha, mu_0, sigma)
    p_value = 1 - norm.cdf(obs_val, mu_0, sigma)
    power = 1 - norm.cdf(crit_val, mu_a, sigma)

    return {
        'x': x, 'pdf_H0': pdf_H0, 'pdf_Ha': pdf_Ha,
        'mu_0': mu_0, 'mu_a': mu_a, 'sigma': sigma,
        'cohen_effect_size': cohen_effect_size,
        'crit_val': crit_val, 'p_value': p_value,
        'power': power, 'obs_val': obs_val
    }

def plot_4_panel_test(obs_val_pass, obs_val_fail, pop_std, mean_diff, alpha=0.05):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('Hypothesis Test: Reject H₀ vs. Fail to Reject H₀', fontsize=16)

    scenarios = [
        plot_hypothesis_test(obs_val_pass, pop_std, mean_diff, alpha),
        plot_hypothesis_test(obs_val_fail, pop_std, mean_diff, alpha)
    ]

    titles = ['Reject H₀', 'Fail to Reject H₀']

    for col, result in enumerate(scenarios):
        # Top row: H₀
        axs[0, col].plot(result['x'], result['pdf_H0'], label='H₀: μ = 0', color='black')
        axs[0, col].fill_between(result['x'], 0, result['pdf_H0'], where=(result['x'] >= result['obs_val']),
                                 color='red', alpha=0.3, label=f'p-value: {result["p_value"]:.3f}')
        axs[0, col].axvline(result['crit_val'], color='red', linestyle='--',
                            label=f'Critical Value: {result["crit_val"]:.2f}$')
        axs[0, col].axvline(result['obs_val'], color='orange', linestyle='--',
                            label=f'Observed: {result["obs_val"]:.2f}')
        axs[0, col].axvline(result['mu_a'], color='purple', linestyle='--',
                            label=f'μₐ = {result["mu_a"]:.2f}')
        axs[0, col].legend()
        axs[0, col].set_title(f'H₀ Distribution ({titles[col]})')
        axs[0, col].set_ylabel('Density')

        textstr = '\n'.join((
            r'$H_0: \mathcal{N}(\mu, \sigma^2)$',
            rf'Effect size: $d = \frac{{\mu_a - \mu_0}}{{\sigma}} = {result["cohen_effect_size"]:.2f}$'
        ))
        axs[0, col].text(0.02, 0.95, textstr, transform=axs[0, col].transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Bottom row: Hₐ
        axs[1, col].plot(result['x'], result['pdf_Ha'], label=f'Hₐ: μ = {result["mu_a"]:.2f}', color='blue')
        axs[1, col].fill_between(result['x'], 0, result['pdf_Ha'], where=(result['x'] >= result['crit_val']),
                                 color='green', alpha=0.3, label=f'Power: {result["power"]:.3f}')
        axs[1, col].fill_between(result['x'], 0, result['pdf_Ha'], where=(result['x'] < result['crit_val']),
                                 color='orange', alpha=0.3, label=f'β error: {(1 - result["power"]):.3f}')
        axs[1, col].axvline(result['crit_val'], color='red', linestyle='--',
                            label=f'Critical Value: {result["crit_val"]:.2f}')
        axs[1, col].axvline(result['obs_val'], color='orange', linestyle='--',
                            label=f'Observed: {result["obs_val"]:.2f}')
        axs[1, col].axvline(result['mu_a'], color='purple', linestyle='--',
                            label=f'μₐ = {result["mu_a"]:.2f}')
        axs[1, col].legend()
        axs[1, col].set_title(f'Hₐ Distribution ({titles[col]})')
        axs[1, col].set_xlabel('Test Statistic')
        axs[1, col].set_ylabel('Density')

        axs[1, col].text(0.02, 0.95, r'$H_a: \mathcal{N}(\mu + \delta, \sigma^2)$',
                         transform=axs[1, col].transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_power_comparison(obs_val=3.1, alpha=0.05):
    x = np.linspace(-5, 8, 1000)
    mu_0 = 0

    # Case 1: Reduced power by decreasing mean difference
    mean_diff_small = 2.75
    sigma_fixed = 1.5
    mu_a_small = mu_0 + mean_diff_small
    pdf_H0_1 = norm.pdf(x, mu_0, sigma_fixed)
    pdf_Ha_1 = norm.pdf(x, mu_a_small, sigma_fixed)
    crit_val_1 = norm.ppf(1 - alpha, mu_0, sigma_fixed)
    p_val_1 = 1 - norm.cdf(obs_val, mu_0, sigma_fixed)
    power_1 = 1 - norm.cdf(crit_val_1, mu_a_small, sigma_fixed)
    cohen_1 = mean_diff_small / sigma_fixed

    # Case 2: Reduced power by increasing variance
    mean_diff_fixed = 4.5
    sigma_large = 2.0
    mu_a_2 = mu_0 + mean_diff_fixed
    pdf_H0_2 = norm.pdf(x, mu_0, sigma_large)
    pdf_Ha_2 = norm.pdf(x, mu_a_2, sigma_large)
    crit_val_2 = norm.ppf(1 - alpha, mu_0, sigma_large)
    p_val_2 = 1 - norm.cdf(obs_val, mu_0, sigma_large)
    power_2 = 1 - norm.cdf(obs_val, mu_a_2, sigma_large)
    cohen_2 = mean_diff_fixed / sigma_large

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('Power Comparison: Decreasing Mean Difference vs Increasing Variance', fontsize=16)

    # -------- Left column (lower effect size) --------
    # H0
    axs[0, 0].plot(x, pdf_H0_1, color='black', label='H₀: μ=0')
    axs[0, 0].fill_between(x, 0, pdf_H0_1, where=(x >= obs_val), color='red', alpha=0.3, label=f'p-value: {p_val_1:.3f}')
    axs[0, 0].axvline(crit_val_1, color='red', linestyle='--', label=f'Critical Value: {crit_val_1:.2f}')
    axs[0, 0].axvline(obs_val, color='orange', linestyle='--', label=f'Observed: {obs_val:.2f}')
    axs[0, 0].axvline(mu_a_small, color='purple', linestyle='--', label=f'μₐ: {mu_a_small:.2f}')
    axs[0, 0].set_title('H₀: Reduced Mean Difference')
    axs[0, 0].legend()
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].text(0.02, 0.95,
        rf'$d = \frac{{\mu_a - \mu_0}}{{\sigma}} = {cohen_1:.2f}$',
        transform=axs[0, 0].transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ha
    axs[1, 0].plot(x, pdf_Ha_1, color='blue', label=f'Hₐ: μ={mu_a_small:.2f}')
    axs[1, 0].fill_between(x, 0, pdf_Ha_1, where=(x >= crit_val_1), color='green', alpha=0.3, label=f'Power: {power_1:.3f}')
    axs[1, 0].fill_between(x, 0, pdf_Ha_1, where=(x < crit_val_1), color='orange', alpha=0.3, label=f'β error: {(1 - power_1):.3f}')
    axs[1, 0].axvline(crit_val_1, color='red', linestyle='--')
    axs[1, 0].axvline(obs_val, color='orange', linestyle='--')
    axs[1, 0].axvline(mu_a_small, color='purple', linestyle='--')
    axs[1, 0].set_title('Hₐ: Reduced Mean Difference')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Test Statistic')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].text(0.02, 0.95, r'$H_a: \mathcal{N}(\mu + \delta, \sigma)$',
                  transform=axs[1, 0].transAxes, fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------- Right column (higher variance) --------
    # H0
    axs[0, 1].plot(x, pdf_H0_2, color='black', label='H₀: μ=0')
    axs[0, 1].fill_between(x, 0, pdf_H0_2, where=(x >= obs_val), color='red', alpha=0.3, label=f'p-value: {p_val_2:.3f}')
    axs[0, 1].axvline(crit_val_2, color='red', linestyle='--', label=f'Critical Value: {crit_val_2:.2f}')
    axs[0, 1].axvline(obs_val, color='orange', linestyle='--', label=f'Observed: {obs_val:.2f}')
    axs[0, 1].axvline(mu_a_2, color='purple', linestyle='--', label=f'μₐ: {mu_a_2:.2f}')
    axs[0, 1].set_title('H₀: Increased Variance')
    axs[0, 1].legend()
    axs[0, 1].text(0.02, 0.95,
        rf'$d = \frac{{\mu_a - \mu_0}}{{\sigma^2}} = {cohen_2:.2f}$',
        transform=axs[0, 1].transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ha
    axs[1, 1].plot(x, pdf_Ha_2, color='blue', label=f'Hₐ: μ={mu_a_2:.2f}')
    axs[1, 1].fill_between(x, 0, pdf_Ha_2, where=(x >= obs_val), color='green', alpha=0.3, label=f'Power: {power_2:.3f}')
    axs[1, 1].fill_between(x, 0, pdf_Ha_2, where=(x < obs_val), color='orange', alpha=0.3, label=f'β error: {(1 - power_2):.3f}')
    axs[1, 1].axvline(crit_val_2, color='red', linestyle='--')
    axs[1, 1].axvline(obs_val, color='orange', linestyle='--')
    axs[1, 1].axvline(mu_a_2, color='purple', linestyle='--')
    axs[1, 1].set_title('Hₐ: Increased Variance')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Test Statistic')
    axs[1, 1].text(0.02, 0.95, r'$H_a: \mathcal{N}(\mu + \delta, \sigma)$',
                  transform=axs[1, 1].transAxes, fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
import seaborn as sns

def plot_sampling_distributions(population, sample_sizes, num_samples):

    mu = np.mean(population)
    sigma = np.std(population)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle("Sampling Distributions of the Mean", fontsize=18)

    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(population, size=n, replace=False)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)
        sample_means_mu = np.mean(sample_means)
        sample_means_std = np.std(sample_means)
        sample_means_var = sample_means_std**2

        ks_stat, _ = kstest(sample_means, 'norm', args=(sample_means_mu, sample_means_std))
        expected_sample_means_var = sigma**2 / n

        ax = axes[i]
        sns.histplot(sample_means, kde=True, stat='density', bins=30, ax=ax)

        x = np.linspace(min(sample_means), max(sample_means), 200)
        ax.plot(x, norm.pdf(x, sample_means_mu, sample_means_std),
                color='green', linestyle='--', label='Normal PDF')

        ax.axvline(mu, color='red', linestyle='--', label='True μ')
        ax.set_title(f"n = {n}\nKS stat = {ks_stat:.3f}")
        ax.set_xlim(0, 12)
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        ax.legend()

        textstr = '\n'.join((
            f'Mean of sample means = {sample_means_mu:.2f}',
            f'Var of sample means = {sample_means_var:.4f}',
            f'Expected σ²/n = {expected_sample_means_var:.4f}'
        ))
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

