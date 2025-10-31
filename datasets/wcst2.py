import numpy as np
import itertools

class WCST:
    """
    Generalized Wisconsin Card Sorting Test (WCST)
    Scales to any number of features, *assuming all features have the
    same number of values* (e.g., N=4 for 4x4x4, or N=5 for 5x5x5).
    """

    def __init__(self, batch_size, features=None):
        """
        features: dict mapping feature name -> list of possible values
                  e.g. {'colour': ['red','blue'], 'shape': ['circle','square']}
                  
                  !! All lists *must* be the same length.
        """
        if features is None:
            features = {
                'colour': ['red', 'blue', 'green', 'yellow'],
                'shape': ['circle', 'square', 'star', 'cross'],
                'quantity': ['1', '2', '3', '4']
            }

        self.features = features
        self.feature_names = list(features.keys())
        self.feature_values = [features[f] for f in self.feature_names]
        self.feature_sizes = [len(v) for v in self.feature_values]

        # --- Corrected Logic ---
        # 1. Validate that all features have the same number of values (N)
        sizes = self.feature_sizes
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(
                "All features must have the same number of values. "
                f"Got sizes: {self.feature_sizes}"
            )
        
        # N = number of values per feature (e.g., 4)
        self.N = sizes[0] 
        # n_features = number of feature dimensions (e.g., 3)
        self.n_features = len(self.feature_values) 

        # 2. Categories (C1...CN) must match N
        self.categories = [f"C{i+1}" for i in range(self.N)]

        # Context category feature (which dimension determines the rule)
        self.category_feature = np.random.choice(range(self.n_features))
        self.batch_size = batch_size

        # 3. Corrected Tokens
        self.gen_deck()
        self.deck_size = len(self.cards)
        self.cat_token_start = self.deck_size
        self.sep_token = self.cat_token_start + self.N  # e.g., 64 + 4 = 68
        self.eos_token = self.sep_token + 1            # e.g., 68 + 1 = 69
        
        # 4. Pre-calculate strides for _extract_features
        # This is the correct way to get strides for itertools.product
        rev_sizes = self.feature_sizes[::-1]
        strides_rev = np.cumprod([1] + rev_sizes[:-1])
        self._strides = strides_rev[::-1]
        # --- End Corrected Logic ---

    def get_vocabulary_size(self):
        return self.deck_size + len(self.categories) + 2

    # === Deck generation ===
    def gen_deck(self):
        """Generates all possible combinations of features."""
        # This implementation was correct
        self.cards = np.array(list(itertools.product(*self.feature_values)))
        self.card_indices = np.arange(len(self.cards))

    # === Context switching ===
    def context_switch(self):
        """Randomly switches to a new rule (feature dimension)."""
        available = np.delete(np.arange(self.n_features), self.category_feature)
        self.category_feature = np.random.choice(available)

    def set_context(self, context):
        if context < 0 or context >= self.n_features:
            raise IndexError("[!] Context out of range")
        self.category_feature = context

    # === Feature Extraction (Corrected) ===
    def _extract_features(self, card_idx):
        """
        Correctly decomposes card indices into their feature indices.
        e.g., card 42 -> [feature_0_idx, feature_1_idx, feature_2_idx]
        
        Supports scalar, 1D, or 2D card_idx arrays.
        """
        # Ensure broadcasting works
        card_idx = np.asarray(card_idx)[..., np.newaxis]
        
        # Math: (index // stride) % feature_size
        features = (card_idx // self._strides) % self.feature_sizes
        
        # For scalar input, return a 1D array
        if features.ndim > 1 and features.shape[0] == 1 and features.shape[1] == 1:
             return features.squeeze(axis=(0,1))
        # For 1D input, return a 2D array [n_cards, n_features]
        elif features.ndim > 1 and features.shape[1] == 1:
            return features.squeeze(axis=1)
        # For 2D input, return a 3D array [batch, n_cards, n_features]
        return features

    # === Batch Generation (Corrected) ===
    def gen_batch(self):
        """Yields batches of card indices and their corresponding labels."""
        n_cards = self.deck_size
        batch_size = self.batch_size
        N = self.N # Number of categories (e.g., 4)

        # Get features for ALL cards once
        all_card_features = self._extract_features(self.card_indices)

        while True:
            cat_feature = self.category_feature # Current rule (e.g., 0 for 'color')

            # --- 1. Correct Partitioning ---
            # Get the feature value for the current rule for all cards
            rule_features = all_card_features[:, cat_feature]
            
            # Find all card indices that match each feature value
            card_partitions = [
                self.card_indices[rule_features == fv]
                for fv in range(N)
            ]
            # --- End Correct Partitioning ---

            # --- 2. Choose Category Cards ---
            category_cards = np.vstack([
                np.random.choice(partition, batch_size, replace=True)
                for partition in card_partitions
            ]).T

            # Shuffle category cards
            category_cards = category_cards[
                np.arange(batch_size)[:, np.newaxis],
                [np.random.permutation(N) for _ in range(batch_size)]
            ]

            # --- 3. Correct 2-Stage Sampling (No Replacement) ---
            
            # 3a. Find all cards *not* in category_cards
            all_flat = np.tile(self.card_indices, batch_size)
            used_flat_idx = (category_cards + np.arange(batch_size)[:, np.newaxis] * n_cards).reshape(-1)
            available_cards_1_flat = np.delete(all_flat, used_flat_idx)
            available_cards_1 = available_cards_1_flat.reshape(batch_size, n_cards - N)
            
            # 3b. Sample example_cards from this pool
            example_idx = np.random.randint(0, n_cards - N, batch_size)
            example_cards = available_cards_1[np.arange(batch_size), example_idx]

            # 3c. *Remove* example_cards to create a new, smaller pool
            avail_1_flat = available_cards_1.reshape(-1)
            avail_1_cols = n_cards - N
            # Get flat indices of the example cards *within* available_cards_1
            example_flat_idx = example_idx + np.arange(batch_size) * avail_1_cols
            available_cards_2_flat = np.delete(avail_1_flat, example_flat_idx)
            available_cards_2 = available_cards_2_flat.reshape(batch_size, n_cards - N - 1)
            
            # 3d. Sample question_cards from the final pool
            question_idx = np.random.randint(0, n_cards - N - 1, batch_size)
            question_cards = available_cards_2[np.arange(batch_size), question_idx]
            # --- End 2-Stage Sampling ---

            # --- 4. Get Features and Labels ---
            card_features = self._extract_features(category_cards)
            example_feature = self._extract_features(example_cards)
            question_feature = self._extract_features(question_cards)

            # Labels: match on active feature
            # card_features: [bs, N, n_feat]
            # example_feature: [bs, n_feat] -> [bs, 1, n_feat]
            example_labels = (card_features[:, :, cat_feature] == example_feature[:, np.newaxis, cat_feature]).astype(int).argmax(axis=1)
            question_labels = (card_features[:, :, cat_feature] == question_feature[:, np.newaxis, cat_feature]).astype(int).argmax(axis=1)

            # --- 5. Yield Tokenized Data ---
            context_batch = np.hstack([
                category_cards, 
                example_cards[:, None], 
                np.ones((batch_size, 1)) * self.sep_token,
                example_labels[:, None] + self.cat_token_start, # e.g., 2 + 64 = 66
                np.ones((batch_size, 1)) * self.eos_token
            ])
            
            target_batch = np.hstack([
                question_cards[:, None], 
                np.ones((batch_size, 1)) * self.sep_token,
                question_labels[:, None] + self.cat_token_start # e.g., 1 + 64 = 65
            ])
            
            yield context_batch.astype(int), target_batch.astype(int)

    # === Visualisation (Updated) ===
    def visualise_batch(self, batch):
        trials = []
        batch = np.hstack(batch)
        
        print(f"--- Feature for classification: {self.feature_names[self.category_feature]} ({self.category_feature}) ---\n")
        
        for trial_idx in range(batch.shape[0]):
            trial = batch[trial_idx].astype(int)
            trial_cards = []
            for token_idx in trial:
                # Use correct token boundaries
                if token_idx < self.deck_size:
                    trial_cards.append(tuple(self.cards[token_idx]))
                elif token_idx < self.sep_token:
                    trial_cards.append(self.categories[token_idx - self.cat_token_start])
                elif token_idx == self.sep_token:
                    trial_cards.append('SEP')
                elif token_idx == self.eos_token:
                    trial_cards.append('EOS')
            print(f"Trial {trial_idx}: {trial_cards}")
            trials.append(trial_cards)
        
        print("\n")
        return trials


if __name__ == "__main__":
    
    print("--- Standard WCST (4x4x4) ---")
    wcst_4 = WCST(2, features={
        'colour': ['red', 'blue', 'green', 'yellow'],
        'shape': ['circle', 'square', 'star', 'cross'],
        'quantity': ['1', '2', '3', '4']
    })
    
    print(f"Vocabulary size: {wcst_4.get_vocabulary_size()}")
    
    batch_4 = next(wcst_4.gen_batch())
    wcst_4.visualise_batch(batch_4)
    
    wcst_4.context_switch()
    batch_4_switched = next(wcst_4.gen_batch())
    wcst_4.visualise_batch(batch_4_switched)

    print("--- Scaled-Up WCST (3x3x3) ---")
    wcst_3 = WCST(1, features={
        'colour': ['red', 'blue', 'green'],
        'shape': ['circle', 'square', 'star'],
        'quantity': ['1', '2', '3']
    })

    print(f"Vocabulary size: {wcst_3.get_vocabulary_size()}")

    batch_3 = next(wcst_3.gen_batch())
    wcst_3.visualise_batch(batch_3)

    print("--- Testing Error Case (Unequal Features) ---")
    try:
        wcst_bad = WCST(1, features={
            'colour': ['red', 'blue'],
            'shape': ['circle', 'square', 'star']
        })
    except ValueError as e:
        print(f"Correctly caught error: {e}")