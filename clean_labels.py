import csv
import re

# Read the messy labels file
with open('messy_list_of_router_labels.csv', 'r') as f:
    content = f.read()

labels = {}

# Parse direct format: "address, label"
direct_pattern = r'^([A-Za-z0-9]{32,44}),\s*(.+)$'
for line in content.split('\n'):
    line = line.strip()
    match = re.match(direct_pattern, line)
    if match:
        addr, label = match.groups()
        labels[addr.strip()] = label.strip()

# Parse SQL CASE format: when trade_source='address' then 'label'
sql_pattern = r"when trade_source='([A-Za-z0-9]{32,44})' then '([^']+)'"
for match in re.finditer(sql_pattern, content):
    addr, label = match.groups()
    labels[addr] = label

# Parse Rust const format: ("address", "label")
rust_pattern = r'\("([A-Za-z0-9]{32,44})",\s*"([^"]+)"\)'
for match in re.finditer(rust_pattern, content):
    addr, label = match.groups()
    # Capitalize and clean up rust-style labels
    clean_label = label.replace('_', ' ').title()
    # Only add if not already present (prefer more descriptive labels)
    if addr not in labels:
        labels[addr] = clean_label

print(f"Found {len(labels)} label mappings")

# Read the router list and add labels
with open('top_router_list.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Add labels to each row
for row in rows:
    addr = row['OUTER_PROGRAM_ID']
    row['LABEL'] = labels.get(addr, '')

# Write the updated file
with open('top_router_list.csv', 'w', newline='') as f:
    fieldnames = ['OUTER_PROGRAM_ID', 'FILLS', 'VOLUME_USD', 'LABEL']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print summary
labeled = sum(1 for row in rows if row['LABEL'])
print(f"Labeled {labeled} out of {len(rows)} routers")
print("\nSample mappings:")
for row in rows[:15]:
    if row['LABEL']:
        print(f"  {row['OUTER_PROGRAM_ID'][:12]}... -> {row['LABEL']}")
