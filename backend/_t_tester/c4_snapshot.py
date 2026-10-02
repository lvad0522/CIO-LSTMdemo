# -*- coding: utf-8 -*-
"""C-4 独立证据：记录两个真实任务库的「目录名全集 + 各自 mtime(mtime_ns) + 文件数」。

总量相等掩盖不了「删一个建一个」——所以必须存**全集**而不是计数。
用法:  python c4_snapshot.py <输出.json>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # backend/_t_tester
BACKEND = HERE.parent
ROOTS = {
    "prediction_jobs": BACKEND / "uploads" / "prediction_jobs",
    "chain_jobs": BACKEND / "uploads" / "chain_jobs",
}


def snap():
    out = {}
    for name, root in ROOTS.items():
        entry = {"path": str(root), "exists": root.is_dir(), "dirs": {}}
        # 根目录自身的 mtime：堵住"建一个又删一个"剩下的最后一个洞——
        # 目录名全集不变，但创建/删除会让**父目录**的 mtime 变。
        for label, d in (("self", root), ("parent", root.parent)):
            try:
                entry["%s_mtime_ns" % label] = d.stat().st_mtime_ns
            except OSError as exc:
                entry["%s_mtime_ns" % label] = str(exc)
        if root.is_dir():
            for d in sorted(root.iterdir(), key=lambda p: p.name):
                try:
                    st = d.stat()
                except OSError as exc:
                    entry["dirs"][d.name] = {"error": str(exc)}
                    continue
                entry["dirs"][d.name] = {
                    "mtime_ns": st.st_mtime_ns,
                    "is_dir": d.is_dir(),
                    # 目录内文件名全集 + 大小，进一步钉住「内容被动过」
                    "files": {
                        f.name: f.stat().st_size
                        for f in sorted(d.iterdir(), key=lambda p: p.name)
                    } if d.is_dir() else None,
                }
        out[name] = entry
    return out


def main():
    data = snap()
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True)
    if dest:
        dest.write_text(text, encoding="utf-8")
        for name, entry in data.items():
            print("%-18s exists=%s  n_dirs=%d  -> %s"
                  % (name, entry["exists"], len(entry["dirs"]), entry["path"]))
        print("快照写入 %s" % dest)
    else:
        print(text)


if __name__ == "__main__":
    main()
