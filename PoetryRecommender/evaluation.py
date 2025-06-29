import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from poetry_recommender import PoetryRecommender
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
import re
import random
warnings.filterwarnings('ignore')

def _norm(s):
    """Normalize (just in case) whitespaces and lowercase"""
    if pd.isna(s):
        return ""
    return re.sub(r'\s+', ' ', str(s)).strip().lower()

class PoetryRecommenderEvaluator:
    """Evaluation with  for hold-out splits."""
    
    def __init__(self, model_path='poetry_recommender_model.pkl'):
        self.recommender = PoetryRecommender()
        try:
            self.recommender.load_model(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        self.full_df = self.recommender.poems_df.copy()
    
    def split_train_test_by_poet(self, test_ratio=0.2, random_state=42):
    
        poets = self.full_df['Poet'].dropna().unique()
        rng = np.random.RandomState(random_state)
        test_poets = set(rng.choice(poets, size=int(len(poets) * test_ratio), replace=False))
        train_poets = set(poets) - test_poets
        
        train_df = self.full_df[self.full_df['Poet'].isin(train_poets)].reset_index(drop=True)
        test_df = self.full_df[self.full_df['Poet'].isin(test_poets)].reset_index(drop=True)
        
        return train_df, test_df, {_norm(p) for p in train_poets}, {_norm(p) for p in test_poets}
    
    def create_ground_truth(self, query_df, candidate_df, sample_size=50, shape_th=0.30, sem_th=0.05):
        """
        Create ground truth relevance lists using criteria:
        - Shape-score >= shape_th (structural similarity)
        - OR semantic_overlap > sem_th (Jaccard similarity on words)
        """
        sample_poems = query_df.sample(min(sample_size, len(query_df)), random_state=0)
        ground_truth = {}
        
        for _, query in sample_poems.iterrows():
            print(f"Creating ground truth for query {len(ground_truth)+1}/{len(sample_poems)}: "
                  f"'{query['Title']}' by {query['Poet']}")
            
            # extract structural features
            qf = self.recommender.extract_structural_features(query['Poem_clean'])
            qvec = np.array([[qf['avg_line_length'], 
                            qf['num_stanzas'],
                            qf['avg_stanza_length'],
                            qf['total_syllables']]])
            qvec = self.recommender.scaler.transform(qvec)[0]
            
            # get poem words
            q_words = set(re.findall(r'\w+', query['Poem_clean'].lower()))
            
            relevant_poems = []
            
            # check all poems in candidate_df
            for _, cand in candidate_df.iterrows():
                # skip the query poem itself
                if (_norm(cand['Title']) == _norm(query['Title']) and 
                    _norm(cand['Poet']) == _norm(query['Poet'])):
                    continue
                
                # calculate structural similarity
                cf = self.recommender.extract_structural_features(cand['Poem_clean'])
                cvec = np.array([[cf['avg_line_length'], 
                                cf['num_stanzas'],
                                cf['avg_stanza_length'],
                                cf['total_syllables']]])
                cvec = self.recommender.scaler.transform(cvec)[0]
                shape_score = np.dot(qvec, cvec) / (np.linalg.norm(qvec) * np.linalg.norm(cvec))
                
                # calculate words overlap (Jaccard)
                c_words = set(re.findall(r'\w+', cand['Poem_clean'].lower()))
                if q_words or c_words:
                    sem_overlap = len(q_words & c_words) / len(q_words | c_words)
                else:
                    sem_overlap = 0.0
                
                # check criteria
                if shape_score >= shape_th or sem_overlap > sem_th:
                    key = f"{_norm(cand['Title'])}@@{_norm(cand['Poet'])}"
                    relevant_poems.append(key)
            
            # store in ground truth
            qkey = f"{_norm(query['Title'])}@@{_norm(query['Poet'])}"
            ground_truth[qkey] = relevant_poems[:20]  #top 20
        
        return ground_truth
    
    def calculate_precision_at_k(self, recommendations, ground_truth, k):
        """Calculate precision@k for a single query."""
        k = min(k, len(recommendations))
        hits = sum(f"{_norm(rec['title'])}@@{_norm(rec['poet'])}" in ground_truth for rec in recommendations[:k])
        return hits / k if k > 0 else 0
    
    def calculate_recall_at_k(self, recommendations, ground_truth, k):
        """Calculate recall@k."""
        if not ground_truth:
            return np.nan
        k = min(k, len(recommendations))
        hits = sum(f"{_norm(rec['title'])}@@{_norm(rec['poet'])}" in ground_truth for rec in recommendations[:k])
        return hits / len(ground_truth)
    
    def calculate_ndcg_at_k(self, recommendations, ground_truth, k):
        """Calculate nDCG@k."""
        k = min(k, len(recommendations))
        
        # Calculate DCG
        dcg = 0
        for i in range(k):
            rec_key = f"{_norm(recommendations[i]['title'])}@@{_norm(recommendations[i]['poet'])}"
            if rec_key in ground_truth:
                dcg += 1 / np.log2(i + 2) 
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_model(self, ground_truth, allowed_authors=None, k_values=[5, 10, 20]):
        """
        Evaluate model performance by precision, recall, and nDCG.
        Recommendations from authors not in the setallowed_authors
        will be filtered out before calculating metrics.
        """
        results = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'all_similarities': [],
            'all_relevance': []
        }
        
        for query_key, relevant_poems in ground_truth.items():
            # Extract title and poet from query key
            title, poet = query_key.split('@@')
            
            # Find the query poem in the dataset
            query_poem = self.full_df[
                (self.full_df['Title'].apply(_norm) == title) & 
                (self.full_df['Poet'].apply(_norm) == poet)
            ]
            
            if query_poem.empty:
                print(f"Warning: Query poem '{query_key}' not found in dataset")
                continue
            
            # Get recommendations
            recommendations = self.recommender.find_similar_poems(
                query_poem.iloc[0]['Poem_clean'], 
                max(k_values) * 5,  # Get extra recommendations to account for filtering
                max_per_poet=None
            )
            
            # Filter by allowed authors if specified (for hold-out evaluation)
            if allowed_authors is not None:
                recommendations = [rec for rec in recommendations if _norm(rec['poet']) in allowed_authors]
            
            # Keep only top max(k_values) recommendations
            recommendations = recommendations[:max(k_values)]
            
            # Calculate metrics for each k
            for k in k_values:
                precision = self.calculate_precision_at_k(recommendations, relevant_poems, k)
                recall = self.calculate_recall_at_k(recommendations, relevant_poems, k)
                ndcg = self.calculate_ndcg_at_k(recommendations, relevant_poems, k)
                
                results['precision'][k].append(precision)
                results['recall'][k].append(recall)
                results['ndcg'][k].append(ndcg)
            
            # Store similarities and relevance for PR curve
            for rec in recommendations:
                rec_key = f"{_norm(rec['title'])}@@{_norm(rec['poet'])}"
                results['all_similarities'].append(rec['similarity_score'])
                results['all_relevance'].append(1 if rec_key in relevant_poems else 0)
        
        return results
    
    def create_evaluation_plots(self, results, ground_truth, save_path='evaluation_plots.png'):
        """Create comprehensive evaluation plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Poetry Recommendation System Evaluation - Precision/Recall Metrics', fontsize=16, fontweight='bold')
        
        # Precision@k
        k_values = list(results['precision'].keys())
        precision_means = [np.mean(results['precision'][k]) for k in k_values]
        precision_stds = [np.std(results['precision'][k]) for k in k_values]
        
        axes[0, 0].bar(k_values, precision_means, yerr=precision_stds, capsize=5, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Precision@k')
        axes[0, 0].set_xlabel('k')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim(0, 1)
        
        # Recall@k
        recall_means = [np.mean(results['recall'][k]) for k in k_values]
        recall_stds = [np.std(results['recall'][k]) for k in k_values]
        
        axes[0, 1].bar(k_values, recall_means, yerr=recall_stds, capsize=5, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Recall@k')
        axes[0, 1].set_xlabel('k')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim(0, 1)
        
        # nDCG@k
        ndcg_means = [np.mean(results['ndcg'][k]) for k in k_values]
        ndcg_stds = [np.std(results['ndcg'][k]) for k in k_values]
        
        axes[0, 2].bar(k_values, ndcg_means, yerr=ndcg_stds, capsize=5, alpha=0.7, color='salmon')
        axes[0, 2].set_title('nDCG@k')
        axes[0, 2].set_xlabel('k')
        axes[0, 2].set_ylabel('nDCG')
        axes[0, 2].set_ylim(0, 1)
        
        # Precision-Recall Curve
        if results['all_similarities'] and results['all_relevance']:
            precision, recall, _ = precision_recall_curve(results['all_relevance'], results['all_similarities'])
            ap_score = average_precision_score(results['all_relevance'], results['all_similarities'])
            
            axes[1, 0].plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Similarity Score Distribution
        axes[1, 1].hist(results['all_similarities'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('Similarity Score Distribution')
        axes[1, 1].set_xlabel('Similarity Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(results['all_similarities']), color='red', linestyle='--',
                          label=f'Mean: {np.mean(results["all_similarities"]):.3f}')
        axes[1, 1].legend()
        
        # Summary Statistics
        summary_text = f"""
        Evaluation Summary:
        
        Ground Truth:
        - Queries: {len(ground_truth)}
        - Avg Relevant per Query: {np.mean([len(relevant) for relevant in ground_truth.values()]):.1f}
        
        Precision@5: {np.mean(results['precision'][5]):.3f} ± {np.std(results['precision'][5]):.3f}
        Recall@5: {np.mean(results['recall'][5]):.3f} ± {np.std(results['recall'][5]):.3f}
        nDCG@5: {np.mean(results['ndcg'][5]):.3f} ± {np.std(results['ndcg'][5]):.3f}
        
        Precision@10: {np.mean(results['precision'][10]):.3f} ± {np.std(results['precision'][10]):.3f}
        Recall@10: {np.mean(results['recall'][10]):.3f} ± {np.std(results['recall'][10]):.3f}
        nDCG@10: {np.mean(results['ndcg'][10]):.3f} ± {np.std(results['ndcg'][10]):.3f}
        
        Precision@20: {np.mean(results['precision'][20]):.3f} ± {np.std(results['precision'][20]):.3f}
        Recall@20: {np.mean(results['recall'][20]):.3f} ± {np.std(results['recall'][20]):.3f}
        nDCG@20: {np.mean(results['ndcg'][20]):.3f} ± {np.std(results['ndcg'][20]):.3f}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=9, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Summary Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_path}")
        plt.show()

def main():
    print("Poetry Recommendation System - Precision/Recall Evaluation")
    print("="*60)
    
    # Configuration
    use_holdout = True  # Set to False for evaluation on full dataset
    sample_size = 50
    k_values = [5, 10, 20]
    
    try:
        evaluator = PoetryRecommenderEvaluator()
        
        if use_holdout:
            # Hold-out evaluation: split by authors
            train_df, test_df, train_authors, test_authors = \
                evaluator.split_train_test_by_poet(test_ratio=0.2, random_state=42)
            
            print(f"Hold-out split: {len(train_authors)} train authors / {len(test_authors)} test authors")
            
            # Create ground truth (queries from test, candidates from train)
            ground_truth = evaluator.create_ground_truth(
                query_df=test_df,
                candidate_df=train_df,
                sample_size=sample_size
            )
            
            # Evaluate model (only train authors allowed in recommendations)
            results = evaluator.evaluate_model(
                ground_truth,
                allowed_authors=train_authors,
                k_values=k_values
            )
            
        else:
            # Traditional evaluation on full dataset
            ground_truth = evaluator.create_ground_truth(
                query_df=evaluator.full_df,
                candidate_df=evaluator.full_df,
                sample_size=sample_size
            )
            results = evaluator.evaluate_model(ground_truth, k_values=k_values)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 40)
        for k in k_values:
            precision = np.nanmean(results['precision'][k])
            recall = np.nanmean(results['recall'][k])
            ndcg = np.nanmean(results['ndcg'][k])
            print(f"k={k:2d}: Precision={precision:.3f}, Recall={recall:.3f}, nDCG={ndcg:.3f}")
        
        # Create plots
        evaluator.create_evaluation_plots(results, ground_truth)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 