import pandas as pd
import numpy as np
from poetry_recommender import PoetryRecommender

def search_poem_by_title(recommender, title):
    """Search for a poem by title in the dataset."""
    # case-insensitive search
    matches = recommender.poems_df[
        recommender.poems_df['Title'].str.contains(title, case=False, na=False)
    ]
    
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches.iloc[0]
    else:
        print(f"\nFound {len(matches)} poems with similar titles:")
        for i, (idx, row) in enumerate(matches.iterrows(), 1):
            print(f"{i}. '{row['Title']}' by {row['Poet']}")
        
        choice = input(f"\nSelect a poem (1-{len(matches)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matches):
                return matches.iloc[choice_idx]
        except ValueError:
            pass
        return None

def display_recommendations(recommendations):
    print(f"\n{'='*60}")
    print(f"TOP {len(recommendations)} SIMILAR POEMS")
    print(f"{'='*60}")
    
    for i, poem in enumerate(recommendations, 1):
        print(f"\n{i}. '{poem['title']}' by {poem['poet']}")
        print(f"   Similarity Score: {poem['similarity_score']:.3f}")
        print(f"   Tags: {poem['tags']}")
        print(f"   Preview: {poem['poem'][:150]}...")
        print("-" * 40)

def main():
    print("Loading trained poetry recommendation model...")
    
    try:
        # load the trained model
        recommender = PoetryRecommender()
        recommender.load_model('poetry_recommender_model.pkl')
        print("Model loaded successfully!")
        if recommender.poems_df is not None:
            print(f"Dataset contains {len(recommender.poems_df)} poems")
        else:
            print("Dataset loaded but poems_df is None")
        
    except FileNotFoundError:
        print("Error: Model file 'poetry_recommender_model.pkl' not found!")
        print("Please run the main training script first: python poetry_recommender.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    while True:
        print("\n" + "="*60)
        print("Choose an option:")
        print("1. Find similar poems (enter poem title)")
        print("2. Try a random poem from the dataset")
        print("3. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # get poem title from user
            poem_title = input("\nEnter poem title: ").strip()
            if not poem_title:
                print("No title entered. Please try again.")
                continue
            
            # search for the poem
            poem = search_poem_by_title(recommender, poem_title)
            if poem is None:
                print(f"Poem '{poem_title}' not found in the dataset.")
                continue
            
            print(f"\nFound: '{poem['Title']}' by {poem['Poet']}")
            print(f"Tags: {poem['Tags']}")
            print(f"Poem preview: {poem['Poem_clean'][:200]}...")
            
            try:
                # find similar poems
                recommendations = recommender.find_similar_poems(poem['Poem_clean'], k=5)
                display_recommendations(recommendations)
            except Exception as e:
                print(f"Error finding similar poems: {e}")
        
        elif choice == "2":
            # try a sample poem from the dataset
            print("\nSelecting a random poem from the dataset...")
            if recommender.poems_df is not None:
                sample_poem = recommender.poems_df.sample(1).iloc[0]
                
                print(f"\nSample poem: '{sample_poem['Title']}' by {sample_poem['Poet']}")
                print(f"Tags: {sample_poem['Tags']}")
                print(f"Poem preview: {sample_poem['Poem_clean'][:200]}...")
                
                try:
                    recommendations = recommender.find_similar_poems(sample_poem['Poem_clean'], k=5)
                    display_recommendations(recommendations)
                except Exception as e:
                    print(f"Error finding similar poems: {e}")
            else:
                print("Error: Dataset not loaded properly")
        
        elif choice == "3":
            print("\nThank you for using the Poetry Recommender!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 