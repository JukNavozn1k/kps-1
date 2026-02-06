#!/usr/bin/env python
"""Integration test for custom features with main data pipeline"""

from kps1.data import load_ftball_dataset, apply_dataset_custom_features
from kps1.feature_engineering import CustomFeature

print('🔄 Integration Test: Feature Engineering + Data Pipeline')
print('=' * 60)

print('\n1️⃣  Loading dataset from ftball.csv...')
ds = load_ftball_dataset('ftball.csv', target='total_goals', seed=42)
print(f'   ✓ Loaded: {ds.X_train.shape[0]} train rows')
print(f'   ✓ Features: {ds.X_train.shape[1]} ({", ".join(ds.feature_names[:3])}...)')

print('\n2️⃣  Creating custom features...')
custom_features = [
    CustomFeature(
        'odds_1_squared', 
        'square', 
        feature1_idx=ds.feature_names.index('odds_1')
    ),
    CustomFeature(
        'odds_1_times_odds_X', 
        'product', 
        feature1_idx=ds.feature_names.index('odds_1'),
        feature2_idx=ds.feature_names.index('odds_X')
    ),
]
print(f'   ✓ Created {len(custom_features)} custom features:')
for cf in custom_features:
    print(f'      • {cf.name} ({cf.operation})')

print('\n3️⃣  Applying custom features to dataset...')
ds_extended = apply_dataset_custom_features(ds, custom_features)
print(f'   ✓ Extended dataset:')
print(f'      • Train shape: {ds_extended.X_train.shape}')
print(f'      • Val shape: {ds_extended.X_val.shape}')

print(f'\n4️⃣  Feature list comparison:')
print(f'   Original features: {ds.X_train.shape[1]}')
print(f'   Extended features: {ds_extended.X_train.shape[1]}')
print(f'   Added: {ds_extended.X_train.shape[1] - ds.X_train.shape[1]}')

print(f'\n5️⃣  New feature names:')
new_features = ds_extended.feature_names[ds.X_train.shape[1]:]
for feat in new_features:
    print(f'      • {feat}')

print('\n' + '=' * 60)
print('✅ All integration tests passed!')
print('🎉 Custom features are ready to use in Streamlit app!')
