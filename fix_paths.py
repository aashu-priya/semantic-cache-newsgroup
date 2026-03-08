import json, os

PERSIST_DIR = '/app/newsgroups_chromadb'

with open(f'{PERSIST_DIR}/manifest.json') as f:
    m = json.load(f)

m['persist_dir'] = PERSIST_DIR
m['embeddings_backup'] = f'{PERSIST_DIR}/embeddings_backup.npy'

for key in ['fuzzy_memberships', 'kmeans_centroids', 'cluster_metadata']:
    if key in m.get('part2', {}):
        m['part2'][key] = f'{PERSIST_DIR}/{os.path.basename(m["part2"][key])}'

for key in ['cache_config', 'test_results', 'threshold_plot']:
    if key in m.get('part3', {}):
        m['part3'][key] = f'{PERSIST_DIR}/{os.path.basename(m["part3"][key])}'

with open(f'{PERSIST_DIR}/manifest.json', 'w') as f:
    json.dump(m, f, indent=2)

print('Paths fixed for container')