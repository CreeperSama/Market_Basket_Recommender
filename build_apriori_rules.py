import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import time
import warnings
import gc

warnings.filterwarnings('ignore')

# 1. Load Data (Reduced to 250k rows)
print("Loading 250,000 rows of transaction data...")
orders_df = pd.read_csv('data/raw/instacart/order_products__prior.csv', nrows=15000000)
products_df = pd.read_csv('data/raw/instacart/products.csv')

print("Merging product names...")
df = orders_df.merge(products_df[['product_id', 'product_name']], on='product_id')
del orders_df, products_df
gc.collect()

# 2. Pre-prune rare items
print("Pre-pruning items below threshold...")
min_support_threshold = 0.0001
item_freq = df['product_name'].value_counts()
valid_items = item_freq[item_freq >= (df['order_id'].nunique() * min_support_threshold)].index
df = df[df['product_name'].isin(valid_items)]

# 3. Build Sparse Matrix
print("Building sparse boolean matrix...")
transactions = df.groupby('order_id')['product_name'].apply(list).tolist()
del df
gc.collect()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)
basket_sets = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

# 4. Execute FP-Growth ML Algorithm
print(f"\nStarting FP-Growth tree traversal at {time.strftime('%X')}...")
start_time = time.time()

frequent_itemsets = fpgrowth(basket_sets, min_support=min_support_threshold, use_colnames=True, max_len=3)

end_time = time.time()
print(f"FP-Growth completed in {round((end_time - start_time), 2)} seconds.")

# 5. Generate and Save Rules
print("Calculating Confidence and Lift...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)
rules['antecedents'] = rules['antecedents'].apply(lambda a: ', '.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ', '.join(list(a)))

output_path = 'data/processed/apriori_rules.csv'
rules.to_csv(output_path, index=False)
print(f"Successfully saved {len(rules)} rules to {output_path}")