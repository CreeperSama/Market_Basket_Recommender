import pandas as pd
import os

RAW_DIR = "C:/Users/natha/Downloads/Product Recommendation System/data/raw/instacart"
OUTPUT_DIR = "C:/Users/natha/Downloads/Product Recommendation System/data/processed/instacart_feature_matrix.csv"

def main():
    print("Loading the Instacart Datasets.....")

    orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    prior = pd.read_csv(os.path.join(RAW_DIR, "order_products__prior.csv"))
    train = pd.read_csv(os.path.join(RAW_DIR, "order_products__train.csv"))

    print("Merging the Values: ")

    prior_merged = pd.merge(prior, orders, on='order_id', how='inner')


    # Product Features
    prd_features = prior_merged.groupby('product_id').agg(
        prod_total_purchases = ('order_id', 'count'),
        prod_reorder_ratio = ('reordered', 'mean')
    ).reset_index()

    # User Features
    usr_features = prior_merged.groupby('user_id').agg(
        user_total_orders = ('order_number', 'max'),
        in_between_days_orders = ('days_since_prior_order', 'mean')
    ).reset_index()

    #Loyalty to a specific item

    uxp_features = prior_merged.groupby(['user_id', 'product_id']).agg(
        uxp_times_bought=('order_id', 'count')
    ).reset_index()

    print("4. Calculating Market Basket Affinity (Frequently Bought Together)...")
    # Sample 100,000 orders to prevent RAM explosion during self-join
    sample_orders = prior['order_id'].drop_duplicates().sample(n=100000, random_state=42)
    basket_subset = prior[prior['order_id'].isin(sample_orders)][['order_id', 'product_id']]
    
    # Self-join to find products in the same cart
    pairs = pd.merge(basket_subset, basket_subset, on='order_id')
    pairs = pairs[pairs['product_id_x'] != pairs['product_id_y']]
    
    # Count frequencies
    # Count frequencies
    item_affinity = pairs.groupby(['product_id_x', 'product_id_y']).size().reset_index(name='basket_affinity_score')
    
    # MEMORY FIX: Keep only the top 5 strongest product pairings for each item
    item_affinity = item_affinity.sort_values(['product_id_x', 'basket_affinity_score'], ascending=[True, False])
    item_affinity = item_affinity.groupby('product_id_x').head(5)
    
    item_affinity.rename(columns={'product_id_x': 'user_past_product', 'product_id_y': 'product_id'}, inplace=True)
    item_affinity.to_csv("C:/Users/natha/Downloads/Product Recommendation System/data/processed/product_affinity.csv", index=False)
    
    # Map back to the user's historical purchases
    user_history = prior_merged[['user_id', 'product_id']].drop_duplicates()
    user_history.rename(columns={'product_id': 'user_past_product'}, inplace=True)
    
    user_affinity_feature = pd.merge(user_history, item_affinity, on='user_past_product', how='inner')
    user_affinity_feature = user_affinity_feature.groupby(['user_id', 'product_id'])['basket_affinity_score'].sum().reset_index()

    print("5. Assembling the Final Feature Matrix...")
    feature_matrix = uxp_features.copy()
    feature_matrix = pd.merge(feature_matrix, usr_features, on='user_id', how='left')
    feature_matrix = pd.merge(feature_matrix, prd_features, on='product_id', how='left')
    feature_matrix = pd.merge(feature_matrix, user_affinity_feature, on=['user_id', 'product_id'], how='left')
    
    # Fill missing affinity scores with 0 (meaning no co-occurrence found)
    feature_matrix['basket_affinity_score'] = feature_matrix['basket_affinity_score'].fillna(0)

    print("6. Attaching Target Variable (Did they reorder it in the train set?)...")
    train_merged = pd.merge(train, orders[['order_id', 'user_id']], on='order_id', how='inner')
    target_df = train_merged[['user_id', 'product_id', 'reordered']].copy()
    target_df.rename(columns={'reordered': 'target'}, inplace=True)

    final_df = pd.merge(feature_matrix, target_df, on=['user_id', 'product_id'], how='left')
    final_df['target'] = final_df['target'].fillna(0).astype(int)
    final_df.fillna(0, inplace=True) # Fill any remaining NaNs

    print("7. Saving preprocessed matrix...")
    final_df.to_csv(OUTPUT_DIR, index=False)
    print(f"Success! Matrix saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


