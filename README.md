# Neural Spike Train Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Field](https://img.shields.io/badge/Field-Computational%20Neuroscience-purple.svg)

## 🧠 Overview

Computational analysis of simulated neural spike trains using Poisson process modeling to characterize neural firing patterns. This project demonstrates fundamental computational neuroscience techniques for understanding neural activity at the single-cell level.

**Key Achievement:** Developed Python pipeline for spike detection, firing rate estimation, and statistical characterization of neural activity patterns - essential skills for computational neuroscience research.

---

## 📊 Results

### Spike Train Visualization
![Spike Train](figures/spike_train_example.png)
*Poisson-generated spike train showing stochastic neural firing over 1000ms duration*

### Firing Rate Analysis
![Firing Rate](figures/firing_rate_distribution.png)
*Time-binned firing rate analysis revealing temporal dynamics of neural activity*

### Key Findings
- Successfully simulated neural spike trains with physiologically realistic firing rates (10 Hz)
- Implemented time-binned firing rate estimation with 50ms windows
- Statistical characterization confirmed Poisson process properties (CV ≈ 1)
- Demonstrated quantitative analysis techniques applicable to real electrophysiology data

---

## 🔬 Scientific Context

**The Neuroscience Question:**  
How can we quantitatively characterize the temporal patterns of neural firing? Understanding spike train statistics is fundamental to decoding information in neural circuits.

**Why Computational Methods?**  
Traditional visual inspection of spike trains is subjective. Computational analysis enables:
- Objective quantification of firing patterns
- Statistical comparison across neurons and conditions  
- Detection of subtle temporal dynamics invisible to human observation
- Foundation for more advanced analyses (information theory, neural decoding)

**Applications:**
- Brain-computer interfaces (decoding motor intent from spike trains)
- Understanding sensory coding mechanisms
- Characterizing disease-related changes in neural firing
- Drug discovery (analyzing effects on neural activity)

---

## 🛠️ Technical Implementation

### Technologies
- **Python 3.x** - Core programming language
- **NumPy** - Efficient numerical computations for spike generation and analysis
- **Matplotlib** - Professional scientific visualizations
- **SciPy** (planned) - Advanced statistical analyses

### Methodology

**1. Spike Train Generation**
- Implemented Poisson process model (standard in computational neuroscience)
- Configurable firing rate parameter (10 Hz default)
- Reproducible random seed for validation

**2. Spike Detection & Timing**
- Precise spike time extraction from continuous simulation
- Sub-millisecond temporal resolution

**3. Firing Rate Analysis**
- Time-binned firing rate estimation (sliding window approach)
- Temporal smoothing options for cleaner visualization

**4. Statistical Characterization**
- Inter-spike interval (ISI) distribution analysis
- Coefficient of variation calculation
- Verification of Poisson process properties

### Code Quality
- Comprehensive docstrings explaining parameters and returns
- Clear variable naming following PEP 8 standards
- Modular function design for reusability
- Detailed inline comments explaining neuroscience concepts

---

## 📈 Sample Output 
## 🎯 Future Directions
- Extend analysis to multi-neuron recordings
- Implement spike-triggered averaging for stimulus encoding
- Add information-theoretic measures (mutual information)
- Apply techniques to real electrophysiology datasets
