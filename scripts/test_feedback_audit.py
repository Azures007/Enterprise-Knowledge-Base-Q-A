"""
端到端验证脚本：登录 → 反馈接口 → 审计接口 → 问答(LLM) → 审计落库
用 FastAPI TestClient 单进程内验证，不依赖常驻服务。
"""
import sys
sys.path.insert(0, r"C:\Users\lx\Desktop\Enterprise-Knowledge-Base-Q-A")

import os
os.chdir(r"C:\Users\lx\Desktop\Enterprise-Knowledge-Base-Q-A")
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))

import time
_last_req = 0
def pace(seconds=0.25):
    """限流中间件有 IP 10QPS 限制，请求间加间隔"""
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < seconds:
        time.sleep(seconds - elapsed)
    _last_req = time.time()

with TestClient(__import__("app").app) as client:
    print("== 1. 登录 ==")
    pace()
    r = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    check("login 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    token = data["data"]["token"] if data.get("data") else None
    check("login has token", bool(token))
    headers = {"Authorization": f"Bearer {token}"}

    print("== 2. 创建对话 + 消息 ==")
    pace()
    r = client.post("/api/conversations", headers=headers)
    conv_id = r.json()["data"]["id"]
    check("create conversation", r.status_code == 200, f"conv_id={conv_id}")

    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages", headers=headers,
                    json={"role": "user", "content": "测试问题"})
    check("add user msg", r.status_code == 200)
    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages", headers=headers,
                    json={"role": "ai", "content": "测试回答", "answer_type": "kb",
                          "sources": [{"filename": "test.pdf", "score": 0.9}]})
    msg_id = r.json()["data"]["id"]
    check("add ai msg", r.status_code == 200, f"msg_id={msg_id}")

    print("== 3. 反馈接口 ==")
    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages/{msg_id}/feedback",
                    headers=headers, json={"feedback": 1})
    check("feedback like=1", r.status_code == 200, f"resp={r.json()}")
    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages/{msg_id}/feedback",
                    headers=headers, json={"feedback": -1, "comment": "答案不准确"})
    check("feedback dislike=-1 with comment", r.status_code == 200)
    pace()
    r = client.get(f"/api/conversations/{conv_id}/messages", headers=headers)
    ai_msg = [m for m in r.json()["data"] if m["id"] == msg_id][0]
    check("message has feedback field", "feedback" in ai_msg and "feedback_comment" in ai_msg,
          f"feedback={ai_msg.get('feedback')} comment={ai_msg.get('feedback_comment')}")
    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages/{msg_id}/feedback",
                    headers=headers, json={"feedback": 0})
    check("feedback clear=0", r.status_code == 200)
    # 非法值
    pace()
    r = client.post(f"/api/conversations/{conv_id}/messages/{msg_id}/feedback",
                    headers=headers, json={"feedback": 5})
    check("feedback invalid value rejected", r.status_code == 400)

    print("== 4. 审计接口（管理员）==")
    pace()
    r = client.get("/api/audit/summary", headers=headers)
    check("audit summary 200", r.status_code == 200, f"data={r.json().get('data', {})}")
    pace()
    r = client.get("/api/audit/queries", headers=headers)
    check("audit queries 200", r.status_code == 200,
          f"count={len(r.json().get('data', []))}")
    pace()
    r = client.get("/api/audit/queries?limit=3", headers=headers)
    check("audit queries limit works", r.status_code == 200 and len(r.json().get("data", [])) <= 3)

    print("== 5. 非管理员无权限 ==")
    pace()
    r = client.post("/api/auth/login", json={"username": "nobody", "password": "wrong"})
    check("wrong login rejected", r.status_code in (401, 200) and (r.status_code != 200 or not r.json().get("data")))

    print("== 6. 问答(真实 LLM) + 审计落库 ==")
    pace()
    r = client.post("/api/query", headers=headers, json={"question": "用一句话介绍你自己"})
    if r.status_code == 200:
        qd = r.json()["data"]
        check("query ok", True, f"answer_type={qd.get('answer_type')}")
        check("query returns related_questions field", "related_questions" in qd)
        # 等审计写入完成
        import time
        time.sleep(0.5)
        r2 = client.get("/api/audit/queries?limit=1", headers=headers)
        rows = r2.json().get("data", [])
        latest = rows[0] if rows else None
        check("audit has query record", bool(latest) and latest.get("question"),
              f"q='{(latest or {}).get('question','')[:20]}' type={(latest or {}).get('answer_type')} "
              f"tokens={(latest or {}).get('total_tokens')} latency={(latest or {}).get('latency_ms')}ms "
              f"user={(latest or {}).get('username')}")
    else:
        check("query ok", False, f"status={r.status_code} detail={r.json()}")

print("\n===== 结果汇总 =====")
failed = [n for n, ok, _ in results if not ok]
print(f"通过 {len(results)-len(failed)}/{len(results)}" + ("  |  失败: " + ", ".join(failed) if failed else "  ✅ 全部通过"))
sys.exit(1 if failed else 0)
