import json

input_file = "/home/jovyan/solKB/agent_kb_database.json"
output_file = "/home/jovyan/solKB/agent_kb_database_filtered.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []
for item in data:
    val = item.get("search_agent_planning", None)

    # 포함 조건:
    # - 키가 없으면(val is None) 포함
    # - 문자열이고 strip() 후 비면(공백 포함) 포함
    if val is None or (isinstance(val, str) and val.strip() == ""):
        filtered.append(item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"총 {len(filtered)}개 저장 완료 → {output_file}")
