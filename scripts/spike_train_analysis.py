"""
Neural Spike Train Analysis
Author: Rashika
Date: January 2026
University of Sussex

This script analyzes simulated neural spike trains to demonstrate
basic computational neuroscience techniques including spike detection,
firing rate analysis, and statistical characterization of neural activity.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_spike_train(rate=10, duration=1000, seed=42):
    """
    Generate a Poisson spike train to simulate neural activity.
    
    The Poisson process is commonly used to model spontaneous neural
    firing because it captures the stochastic nature of neural activity.
    
    Parameters:
    -----------
    rate : float
        Mean firing rate in spikes per second (Hz)
    duration : float
        Duration of spike train in milliseconds
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    spike_times : numpy array
        Array of spike times in milliseconds, sorted chronologically
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Calculate expected number of spikes based on rate and duration
    expected_spikes = int(rate * duration / 1000)
    
    # Generate random number of spikes from Poisson distribution
    # This captures the variability in neural firing
    num_spikes = np.random.poisson(expected_spikes)
    
    # Generate spike times uniformly distributed across duration
    spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
    
    return spike_times


def calculate_firing_rate(spike_times, bin_size=50, duration=None):
    """
    Calculate time-varying firing rate using binned spike counts.
    
    This provides insight into how neural activity changes over time,
    which is crucial for understanding neural coding and dynamics.
    
    Parameters:
    -----------
    spike_times : numpy array
        Array of spike times in milliseconds
    bin_size : float
        Size of time bins in milliseconds (default: 50ms)
    duration : float or None
        Total duration. If None, uses maximum spike time
    
    Returns:
    --------
    bin_centers : numpy array
        Center time of each bin in milliseconds
    firing_rates : numpy array
        Firing rate in each bin (spikes per second)
    """
    if duration is None:
        duration = spike_times[-1] if len(spike_times) > 0 else 1000
    
    # Create time bins
    bins = np.arange(0, duration + bin_size, bin_size)
    
    # Count spikes in each bin
    counts, _ = np.histogram(spike_times, bins=bins)
    
    # Convert counts to firing rate (spikes per second)
    firing_rates = counts / (bin_size / 1000.0)
    
    # Calculate bin centers for plotting
    bin_centers = bins[:-1] + bin_size / 2
    
    return bin_centers, firing_rates


def calculate_isi_statistics(spike_times):
    """
    Calculate inter-spike interval (ISI) statistics.
    
    ISI analysis reveals important properties of neural firing patterns:
    - Mean ISI relates to average firing rate
    - ISI variability indicates regularity of firing
    - Coefficient of variation (CV) characterizes firing pattern:
      * CV ≈ 1: Poisson-like (irregular)
      * CV < 1: Regular firing
      * CV > 1: Bursty firing
    
    Parameters:
    -----------
    spike_times : numpy array
        Array of spike times in milliseconds
    
    Returns:
    --------
    dict : Dictionary containing ISI statistics
        - mean_isi: Mean inter-spike interval (ms)
        - std_isi: Standard deviation of ISI (ms)
        - cv: Coefficient of variation (dimensionless)
    """
    # Calculate intervals between consecutive spikes
    isis = np.diff(spike_times)
    
    if len(isis) == 0:
        return {
            'mean_isi': 0,
            'std_isi': 0,
            'cv': 0,
            'num_intervals': 0
        }
    
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    
    # Coefficient of variation: normalized measure of variability
    cv = std_isi / mean_isi if mean_isi > 0 else 0
    
    return {
        'mean_isi': mean_isi,
        'std_isi': std_isi,
        'cv': cv,
        'num_intervals': len(isis)
    }


def plot_spike_analysis(spike_times, save_path='figures/spike_analysis.png'):
    """
    Create comprehensive visualization of spike train analysis.
    
    Generates a three-panel figure showing:
    1. Raster plot - Visual representation of spike timing
    2. Firing rate - Temporal dynamics of neural activity
    3. ISI histogram - Statistical properties of spike intervals
    
    Parameters:
    -----------
    spike_times : numpy array
        Array of spike times in milliseconds
    save_path : str
        Path to save the output figure
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Neural Spike Train Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # PANEL 1: Raster Plot
    # Shows the precise timing of each spike
    axes[0].scatter(spike_times, np.ones_like(spike_times), 
                   marker='|', s=200, c='black', linewidths=2)
    axes[0].set_xlim(0, max(spike_times[-1], 1000))
    axes[0].set_ylim(0.5, 1.5)
    axes[0].set_ylabel('Neuron', fontsize=12, fontweight='bold')
    axes[0].set_title('A. Spike Raster Plot', fontsize=13, loc='left')
    axes[0].set_yticks([])
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_xlabel('')
    
    # PANEL 2: Firing Rate Over Time
    # Shows how neural activity varies across the recording
    times, rates = calculate_firing_rate(spike_times)
    axes[1].plot(times, rates, linewidth=2, color='steelblue')
    axes[1].fill_between(times, rates, alpha=0.3, color='steelblue')
    axes[1].axhline(y=np.mean(rates), color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(rates):.1f} Hz')
    axes[1].set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    axes[1].set_title('B. Instantaneous Firing Rate', fontsize=13, loc='left')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel('')
    
    # PANEL 3: Inter-Spike Interval Histogram
    # Reveals the distribution of time intervals between spikes
    isis = np.diff(spike_times)
    if len(isis) > 0:
        axes[2].hist(isis, bins=30, color='coral', alpha=0.7, 
                    edgecolor='black', linewidth=1.2)
        
        # Add vertical line for mean ISI
        mean_isi = np.mean(isis)
        axes[2].axvline(x=mean_isi, color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_isi:.1f} ms')
    
    axes[2].set_xlabel('Inter-Spike Interval (ms)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[2].set_title('C. Inter-Spike Interval Distribution', fontsize=13, loc='left')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    # Add statistics text box
    stats = calculate_isi_statistics(spike_times)
    stats_text = (
        f"Mean ISI: {stats['mean_isi']:.2f} ms\n"
        f"Std ISI: {stats['std_isi']:.2f} ms\n"
        f"CV: {stats['cv']:.2f}\n"
        f"Intervals: {stats['num_intervals']}"
    )
    axes[2].text(0.98, 0.97, stats_text, transform=axes[2].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8),
                fontsize=10, family='monospace')
    
    if len(isis) > 0:
        axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {save_path}")
    plt.close()


def print_summary_statistics(spike_times):
    """
    Print comprehensive summary statistics of spike train.
    
    Parameters:
    -----------
    spike_times : numpy array
        Array of spike times in milliseconds
    """
    duration = spike_times[-1] / 1000.0 if len(spike_times) > 0 else 1.0
    num_spikes = len(spike_times)
    mean_rate = num_spikes / duration
    
    stats = calculate_isi_statistics(spike_times)
    
    print("\n" + "="*60)
    print("           SPIKE TRAIN ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nRecording Properties:")
    print(f"  Duration:              {duration:.2f} seconds")
    print(f"  Total spikes:          {num_spikes}")
    print(f"  Mean firing rate:      {mean_rate:.2f} Hz")
    
    print(f"\nInter-Spike Interval Statistics:")
    print(f"  Mean ISI:              {stats['mean_isi']:.2f} ms")
    print(f"  ISI standard dev:      {stats['std_isi']:.2f} ms")
    print(f"  Coefficient of var:    {stats['cv']:.2f}")
    print(f"  Number of intervals:   {stats['num_intervals']}")
    
    # Interpret CV value
    print(f"\nFiring Pattern Interpretation:")
    if stats['cv'] < 0.5:
        pattern = "Regular/rhythmic firing"
    elif stats['cv'] < 1.5:
        pattern = "Poisson-like (irregular) firing"
    else:
        pattern = "Bursty/highly variable firing"
    print(f"  CV = {stats['cv']:.2f} suggests: {pattern}")
    
    print("="*60 + "\n")


def main():
    """
    Main execution function for spike train analysis.
    
    This function demonstrates a complete analysis pipeline:
    1. Data generation (simulated spikes)
    2. Statistical analysis
    3. Visualization
    """
    print("\n" + "="*60)
    print("      NEURAL SPIKE TRAIN ANALYSIS")
    print("      Author: Rashika | University of Sussex")
    print("="*60)
    
    print("\n[1/3] Generating simulated neural data...")
    
    # Simulation parameters
    # These can be modified to explore different firing patterns
    firing_rate = 15    # Hz (spikes per second)
    duration = 1000     # milliseconds (1 second)
    random_seed = 42    # For reproducibility
    
    # Generate spike train using Poisson process
    spike_times = generate_spike_train(
        rate=firing_rate,
        duration=duration,
        seed=random_seed
    )
    
    print(f"✓ Generated {len(spike_times)} spikes over {duration} ms")
    print(f"  (Expected ~{firing_rate} spikes for {firing_rate} Hz rate)")
    
    # Statistical analysis
    print("\n[2/3] Calculating statistics...")
    print_summary_statistics(spike_times)
    
    # Visualization
    print("[3/3] Creating visualizations...")
    plot_spike_analysis(spike_times)
    
    print("="*60)
    print("✓ Analysis complete!")
    print("="*60)
    print("\nOutput:")
    print("  • figures/spike_analysis.png - Three-panel visualization")
    print("\nNext Steps:")
    print("  1. Open figures/spike_analysis.png to view results")
    print("  2. Try changing firing_rate (lines 237-239) to explore")
    print("  3. Modify bin_size in firing rate calculation")
    print("  4. Experiment with different random seeds")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt

# 1. Spike train visualization
plt.figure(figsize=(12, 4))
# [your spike train plotting code]
plt.savefig('figures/spike_train_example.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Firing rate histogram
plt.figure(figsize=(8, 6))
# [your firing rate histogram code]
plt.savefig('figures/firing_rate_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Raster plot (if you have multiple trials)
plt.figure(figsize=(10, 6))
# [your raster plot code]
plt.savefig('figures/raster_plot.png', dpi=300, bbox_inches='tight')
plt.close()
