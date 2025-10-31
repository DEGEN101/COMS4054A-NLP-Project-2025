# NLP In-Context Learning Project: WCST Task with a Transformer

This project explores in-context learning using a simplified Transformer model to solve the Wisconsin Card Sorting Test (WCST). The WCST is a cognitive test that assesses abstract reasoning and cognitive flexibility.

## Project Structure
```
├── requirements.txt #List of required the Python packages
├── README.md
├── .gitignore
├── NLP_In_Context_Project.pdf
│
├── datasets
│   ├── custom_dataset.py # Custom dataset class for WCST
│   ├── wcst.py # Standard WCST environment
│   └── wcst2.py # Extended or modified WCST environment
│
├── misc # Miscellaneous scripts
│
├── models # Saved model checkpoints (optional)
│
├── notebooks # Jupyter notebooks for each experiment
│   ├── model_architecture
│   ├── scaling_laws
│   └── training_methods
│
├── plots # Generated plots for analysis
│
├── results # Model outputs, metrics, and experiment results
│
├── src
│   ├── transformer.py # Transformer model definition
│   └── utilities.py # Helper functions for training/evaluation
│
└── utilities # Utility functions for plotting & displaying results
    ├── analyze_round_robin_vs_mixed.py
    ├── plot_performance_metrics.py
    └── plot_scaling_law_performance_metrics.py
```


## Installation
1. Clone the repository:
```bash
    git clone https://github.com/DEGEN101/COMS4054A-NLP-Project-2025.git
    cd COMS4054A-NLP-Project-2025
```

Install these packages using `pip`:

```bash
    pip install -r requirements.txt
```


## Usage
Open and run any Jupyter Notebook in the notebook directory to reproduce an experiments results. Then run the corresponding plotting script to plot/display the results, e.g. For the architecture experiments:
1. Run the notebooks `train_baseline_model.ipynb` and `train_ce_model.ipynb` in the `/notebooks/model_architecture/` directory (these will produce two json files in the `/results/architecture/` directory)
2. Run `plot_performace_metrics.py` to plot results as follows:
    ``` bash
        python plot_compare_models.py <baseline_json> <ce_json>
    ```
3. Plots of the results will be found in the `plots` directory.


## Contributions
Contributions to this project are welcome. Feel free to submit pull requests or open issues to suggest improvements or report bugs.

## References
- WCST: Wikipedia - [Wisconsin Card Sorting Test](https://en.wikipedia.org/wiki/Wisconsin_Card_Sorting_Test)
- Transformer architecture: Vaswani et al., Attention Is All You Need (2017)
