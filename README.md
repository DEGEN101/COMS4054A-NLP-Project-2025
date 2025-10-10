# NLP In-Context Learning Project: WCST Task with a Transformer

This project explores in-context learning using a simplified Transformer model to solve the Wisconsin Card Sorting Test (WCST). The WCST is a cognitive test that assesses abstract reasoning and cognitive flexibility.

## Project Structure
*   `datasets/`:
    *   `wcst.py`: Contains the `WCST` class for generating WCST trials and visualising batches.
*   `notebooks/`:
    *   `train_model.ipynb`: A Jupyter Notebook containing the code for Transformer model definition, training, and inference.
    *   `create_dataset.ipynb`: A Jupyter Notebook containing the code to create train, test, and validation datasets from `wcst.py`.
*   `src/`:
    *   `transformer.py`: Defines the Transformer model.
    *   `utilities.py`: Contains utility functions for the Transformer model.
*   `requirements.txt`: Lists the Python packages required to run the project.

## Requirements
You can install these packages using `pip`:

```bash
    pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
    git clone https://github.com/DEGEN101/COMS4054A-NLP-Project-2025.git
    cd COMS4054A-NLP-Project-2025
```

2. Install the required packages (as mentioned above).

3. Open and run the **train_model.ipynb** Jupyter Notebook. The notebook
contains detailed explanations and code for each step of the project, including:
    - Dataset creation and loading
    - Transformer model definition and initialisation
    - Training the Transformer model
    - Testing the Transformer model
    - Model inference and visualisation of results

## WCST Task
The WCST class in wcst.py simulates the Wisconsin Card Sorting Test. Key features:
- Generates a deck of cards with features like colour, shape and quantity.
- Creates batches of trials for training and testing.
- Implements context switching, where the relevant feature for classification changes.
- Provides visualisation of the card sorting process.

## Contributions
Contributions to this project are welcome. Feel free to submit pull requests or open issues to suggest improvements or report bugs.
