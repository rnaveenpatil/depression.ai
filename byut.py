import sys, json
d = json.load(sys.stdin)
rows=[]
for t in d['ResultsByTime']:
    for g in t['Groups']:
        a=float(g['Metrics']['UnblendedCost']['Amount'])
        if a>0.0001:
            rows.append((a, g['Keys'][0]))
for a,k in sorted(rows, reverse=True):
    print(f"  ${a:0.5f}  {k}")
