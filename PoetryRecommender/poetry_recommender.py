import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
import pickle
import warnings


class PoetryRecommender:
    def __init__(self, model_name='all-mpnet-base-v2', struct_weight=0.3):
      
        self.model_name = model_name
        self.struct_weight = struct_weight  # relative weight of structural features
        self.model = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.index = None
        self.poems_df = None
        self.feature_names = None
        
    def extract_structural_features(self, poem_clean):
        """
        Extract four structural features from a cleaned poem:
        1. Average line length
        2. Number of stanzas
        3. Average stanza length
        4. Syllable count
        """
        # split into stanzas
        stanzas = poem_clean.split('<SB>')
        stanzas = [s.strip() for s in stanzas if s.strip()]
        
        # split stanzas into lines
        all_lines = []
        for stanza in stanzas:
            lines = stanza.split('<LB>')
            lines = [line.strip() for line in lines if line.strip()]
            all_lines.extend(lines)
        
        #1. average line length
        avg_line_length = np.mean([len(line) for line in all_lines]) if all_lines else 0
        
        #2. number of stanzas
        num_stanzas = len(stanzas)
        
        #3. average stanza length
        stanza_lengths = [len([line for line in stanza.split('<LB>') if line.strip()]) 
                         for stanza in stanzas]
        avg_stanza_length = np.mean(stanza_lengths) if stanza_lengths else 0
        
        #4. total syllables
        total_syllables = sum(self._count_syllables(line) for line in all_lines)
        
        return {
            'avg_line_length': avg_line_length,
            'num_stanzas': num_stanzas,
            'avg_stanza_length': avg_stanza_length,
            'total_syllables': total_syllables
        }
    
    def _count_syllables(self, text):
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        syllables = 0
        for word in text.split():
            # count vowels 
            vowels = len(re.findall(r'[aeiouy]', word))
            if vowels == 0:
                vowels = 1  # every word has at least one syllable
            syllables += vowels
        
        return syllables
    
    def extract_all_features(self, df):
        """Extract structural features for all poems and add them to the dataframe."""
        
        features_list = []
        for _, row in df.iterrows():
           
            
            features = self.extract_structural_features(row['Poem_clean'])
            features_list.append(features)
        
        # create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # combine with original data
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
       
        return result_df
    
    def generate_embeddings(self, poems):
        """Generate text embeddings using Sentence-BERT."""
      
        embeddings = self.model.encode(poems, show_progress_bar=True)
        
        # normalize embeddings to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        
        return embeddings
    
    def build_hybrid_features(self, df):
        """Combine text embeddings with structural features."""
        
        # generate text embeddings
        embeddings = self.generate_embeddings(df['Poem_clean'].tolist())
        
        # extract structural features
        structural_features = df[['avg_line_length', 'num_stanzas', 'avg_stanza_length', 
                                 'total_syllables']].values
        
        # normalize structural features (zero-mean, unit-var)
        structural_features_scaled = self.scaler.fit_transform(structural_features)

        # apply weighting factor to control influence
        structural_features_weighted = structural_features_scaled * self.struct_weight  # type: ignore

        # combine embeddings with weighted structural features
        combined_features = np.hstack([embeddings, structural_features_weighted])

        # final L2 normalisation of the full vector
        norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
        combined_features = combined_features / norms
        
        # create feature names
        embedding_names = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        structural_names = ['avg_line_length', 'num_stanzas', 'avg_stanza_length', 
                           'total_syllables']
        self.feature_names = embedding_names + structural_names
        
  
        return combined_features
    
    def build_faiss_index(self, features):
        """Build FAISS index."""
        
        # create FAISS index
        dimension = features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # add vectors to index
        index.add(features.astype('float32'))  # type: ignore
        
        return index
    
    def train(self, filepath):

        # load data
        self.poems_df = pd.read_csv(filepath)
        
        # extract structural features
        self.poems_df = self.extract_all_features(self.poems_df)
        
        # build hybrid features
        features = self.build_hybrid_features(self.poems_df)
        
        # build FAISS index
        self.index = self.build_faiss_index(features)
        
        print("Training completed successfully!")
    
    def find_similar_poems(self, query_poem, k=5, max_per_poet=2):
        if self.index is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # extract features for query poem
        query_features = self.extract_structural_features(query_poem)
        
        # generate embedding for query poem
        query_embedding = self.model.encode([query_poem])
        
        # normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # scale structural features
        query_structural = np.array([[query_features['avg_line_length'], 
                                    query_features['num_stanzas'],
                                    query_features['avg_stanza_length'],
                                    query_features['total_syllables']]])
        query_structural_scaled = self.scaler.transform(query_structural)

        # apply same weighting
        query_structural_weighted = query_structural_scaled * self.struct_weight  # type: ignore

        # combine features
        query_combined = np.hstack([query_embedding, query_structural_weighted])

        # final L2 normalisation
        query_norm = np.linalg.norm(query_combined, axis=1, keepdims=True)
        query_combined = query_combined / query_norm
        
        # search FAISS index with oversampling to allow filtering duplicates
        overfetch = max(5, k * 3)  # fetch more candidates than needed
        distances, indices = self.index.search(query_combined.astype('float32'), overfetch + 1)  
        
        # return similar poems with per-poet limit
        similar_poems = []
        poet_counts = {}
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # skip the query poem itself (distance very close to 0)
            if i == 0 and distance < 1e-6:
                continue
            
            if self.poems_df is not None:
                poet = self.poems_df.iloc[idx]['Poet'].strip() if isinstance(self.poems_df.iloc[idx]['Poet'], str) else ""
                if max_per_poet is not None and poet_counts.get(poet, 0) >= max_per_poet:
                    
                    continue
                
                poem_info = {
                    'rank': len(similar_poems) + 1,
                    'title': self.poems_df.iloc[idx]['Title'],
                    'poet': poet,
                    'tags': self.poems_df.iloc[idx]['Tags'],
                    'poem': self.poems_df.iloc[idx]['Poem'],
                    'similarity_score': 1.0 / (1.0 + distance),  # convert distance to similarity
                    'distance': distance
                }
                similar_poems.append(poem_info)
                poet_counts[poet] = poet_counts.get(poet, 0) + 1
                
                if len(similar_poems) >= k:
                    break
        
        return similar_poems
    
    def save_model(self, filepath):
    
        model_data = {
            'model_name': self.model_name,
            'scaler': self.scaler,
            'index': self.index,
            'poems_df': self.poems_df,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.model = SentenceTransformer(self.model_name)
        self.scaler = model_data['scaler']
        self.index = model_data['index']
        self.poems_df = model_data['poems_df']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")

def main():
    # initialize recommender
    recommender = PoetryRecommender()
    
    # train the model
    print("Training Poetry Recommendation System")
    recommender.train('data/processed/final_clean.csv')
    
    # save the trained model
    recommender.save_model('poetry_recommender_model.pkl')
    
    print("\nTraining completed! Model saved as 'poetry_recommender_model.pkl'")
    print("You can now use the interactive recommender: python interactive_recommender.py")

if __name__ == "__main__":
    main() 