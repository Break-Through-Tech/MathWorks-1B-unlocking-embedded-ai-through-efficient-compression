# Smarter Devices, Smaller Models: Embedded AI for Bearing Fault Diagnosis

### 👥 **Team Members**

Each member explored the data, trained their own model, and applied their own compression techniques. We then compared results as a team.

| Name | GitHub Handle | Contribution |
|------|---------------|--------------|
| Maayan Matsliah | @maayanmatsliah-tech | 1D CNN, INT8 quantization |
| Rajarshi Das | @rajarshistudy | Decision tree, standardization |
| Bharathi Kaliraj | @bharathiks06 | EDA, decision tree, 1D CNN |
| Landon Rodrin | @landonrodrin | 1D CNN, pruning + INT8 quantization |
| Chibuzor Abia | @chibuzor-source | N/A |
| Liezeil Jimenez | @liezeil | 2D spectrogram CNN, pruning + INT8 quantization |
| Alayna Veeck | @alaynav | EDA |

---

## 🎯 **Project Highlights**

- Built deep learning models to diagnose bearing faults from vibration data.
- Classified three conditions: Normal, Inner Race Fault, and Outer Race Fault.
- Reached 99% to 100% test accuracy with CNNs. Traditional ML reached about 90%.
- Compressed models with pruning and quantization. 

---

## 👩🏽‍💻 **Setup and Installation**

1. Install Miniforge from https://github.com/conda-forge/miniforge/releases/latest.
2. Clone the repo:
   ```bash
   git clone https://github.com/Break-Through-Tech/MathWorks-1B-unlocking-embedded-ai-through-efficient-compression.git
   cd MathWorks-1B-unlocking-embedded-ai-through-efficient-compression
   ```
3. Create the environment:
   ```bash
   mamba env create -f environment.yml
   ```
4. Activate it:
   ```bash
   mamba activate mathworks1b
   ```
5. Register the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name mathworks1b --display-name "MathWorks 1B"
   ```
6. Start Jupyter and pick the "MathWorks 1B" kernel:
   ```bash
   jupyter lab
   ```
7. The data is in the `data/` folder. See `data/README.md` for details.

To add a package, update `environment.yml`. Then run
`mamba env update -f environment.yml --prune`.

Each teammate's scripts live in `notebooks/<Name>/`.

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

- Dataset: Rolling Element Bearing Fault Diagnosis, from Eric Bechhoefer.
- Format: MATLAB (.mat) files.
- It has 522 samples. Each sample has 5,000 vibration readings.
- Split: 393 train, 27 validation, 102 test.
- Classes are perfectly balanced.
- The data came pre-segmented, balanced, and split.
- All data comes from one lab test rig. Results may not carry to other machines.

**EDA insights:**
- Normal signals are steady noise.
- Inner race faults show sharp, periodic spikes.
- Outer race faults show smaller periodic spikes.
- Standard deviation: 0.77 (Normal), 0.90 (Outer), 1.89 (Inner).

---

## 🧠 **Model Development**

**Traditional ML baselines:**
- Logistic regression: about 90% accuracy.
- SVM (RBF kernel): about 91% accuracy.
- Decision tree on feature inputs: 99% unstandardized, 100% standardized.

**Deep learning:**
- 1D CNN (6,923 parameters): 100% test accuracy on 5 seeds.
- Spectrogram 2D CNN (6,099 parameters): 100% test accuracy on 5 seeds.
- Small 1D CNN (1,027 parameters, 7.0 KB): 99.0% test accuracy.
- Large 1D CNN (47,683 parameters): 100% test accuracy.

**Compression:**
- Pruning: unstructured (global L1, 50% to 98% sparsity) and structured
  (filter removal, 75% down to 12.5% of filters kept), with fine-tuning.
- Quantization: INT8 post-training quantization with BatchNorm folded into the
  convolutions.

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

- Test on other bearing datasets and noisier signals.
- Deploy the compressed model to real embedded hardware and measure on-device latency and memory.
- Limitations: one test rig, a small test set (102 samples), and 100% accuracy leaves little room to tell models apart.

---

## 📝 **License**

- The dataset is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 📄 **References**

- Dataset: Eric Bechhoefer, http://data-acoustics.com/measurements/bearing-faults/bearing-2

---

## 🙏 **Acknowledgements** 

- Thank you to Nayara Gomes de Aguiar at MathWorks for guidance and feedback.
- Thank you to Eric Bechhoefer for the dataset.
- Thank you to Rashidah Carr, our AI Studio Coach.
- Thank you to Break Through Tech.
