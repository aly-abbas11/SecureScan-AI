
import json
import os

# Load CWE database
with open('cwe_database.json', 'r') as f:
    CWE_DATABASE = json.load(f)

def get_fix_suggestion(cwe_id):
    cwe_id = cwe_id.strip().upper()
    if not cwe_id.startswith("CWE-"):
        cwe_id = "CWE-" + cwe_id
    if cwe_id not in CWE_DATABASE:
        return {
            "found": False,
            "cwe_id": cwe_id,
            "message": f"CWE {cwe_id} not in database."
        }
    data = CWE_DATABASE[cwe_id]
    return {
        "found": True,
        "cwe_id": cwe_id,
        "name": data["name"],
        "severity": data["severity"],
        "description": data["description"],
        "why_dangerous": data["why_dangerous"],
        "bad_code": data["bad_code"],
        "good_code": data["good_code"],
        "fix_steps": data["fix_steps"]
    }

def print_fix_report(cwe_id):
    result = get_fix_suggestion(cwe_id)
    if not result["found"]:
        print(f"Not found: {result['message']}")
        return
    severity_emoji = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢"
    }
    emoji = severity_emoji.get(result["severity"], "⚪")
    print("=" * 60)
    print(f"🔍 VULNERABILITY REPORT")
    print("=" * 60)
    print(f"CWE ID:    {result['cwe_id']}")
    print(f"Name:      {result['name']}")
    print(f"Severity:  {emoji} {result['severity']}")
    print("-" * 60)
    print(f"\n📖 WHAT IS IT?")
    print(f"{result['description']}")
    print(f"\n⚠️  WHY IS IT DANGEROUS?")
    print(f"{result['why_dangerous']}")
    print(f"\n❌ VULNERABLE CODE:")
    print(result['bad_code'])
    print(f"\n✅ FIXED CODE:")
    print(result['good_code'])
    print(f"\n🔧 HOW TO FIX IT:")
    for i, step in enumerate(result['fix_steps'], 1):
        print(f"  {i}. {step}")
    print("=" * 60)
