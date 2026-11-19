#!/usr/bin/env python3
"""Find up to 500 real Hugging Face dataset repos relevant to spoken dialogue.

The results are candidates, NOT 500 certified full-duplex audio corpora.
HF metadata and listed files are validated, but archives are NOT opened.
No dataset downloads take place.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

TERMS = (
    "full-duplex duplex spoken-dialogue dialogue dialog conversation conversational "
    "meeting meetings telephone telephony callhome callfriend switchboard fisher "
    "podcast interview interviews multi-speaker multispeaker multi-talker multitalker "
    "speaker-diarization diarization overlap backchannel turn-taking turntaking "
    "spontaneous call-center callcenter speech-to-speech s2s voice-conversation "
    "speech-dialogue chat-audio speech-chat speech-conversation spoken-qa "
    "speech-dataset speech-corpus dialogue-speech chime aishell alimeeting "
    "multilingual-speech natural-speech code-switching bilingual-speech multi-turn "
    "doctor-patient counseling discussion debate call-recording"
).split()
STRONG = (
    "duplex", "dialog", "convers", "meeting", "telephone", "telephony",
    "callhome", "callfriend", "switchboard", "fisher", "podcast",
    "interview", "multi-speaker", "multispeaker", "multi-talker",
    "multitalker", "diariz", "backchannel", "turntaking", "turn-taking",
    "overlap", "spontaneous", "callcenter", "call-center", "speech-to-speech",
    "s2s", "spoken-qa", "code-switch", "multi-turn", "discussion", "debate",
    "counseling", "patient", "chime", "alimeeting"
)
DIRECT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma")
PACKED = (".tar", ".tar.gz", ".tgz", ".zip", ".parquet", ".arrow", ".h5", ".hdf5")
LANG = re.compile(r"^language:([a-z]{2,3}(?:-[a-z0-9]+)?)$", re.I)
FIELDNAMES = (
    "id name lang hours kind access notes url group train dup verified source "
    "license audio_evidence quality_tier file_count direct_audio_files container_files "
    "size_gib_known size_unknown_files gated candidate_score needs_human_review"
).split()

def repo_id(info):
    return getattr(info, "id", None) or getattr(info, "repo_id", None)

def tags(info):
    return list(getattr(info, "tags", None) or [])

def score_name(repo, tg):
    name = repo.lower().replace("_", "-")
    score = sum(4 for word in STRONG if word in name)
    score += min(3, sum(int(x in name) for x in ("speech", "audio", "asr", "voice")))
    score += min(5, sum(int(x in " ".join(tg).lower()) for x in STRONG))
    if any(x in name for x in ("text-only", "no-audio", "transcripts-only")):
        score -= 10
    return score

def license_of(info):
    card = getattr(info, "card_data", None)
    lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if isinstance(lic, list):
        lic = ",".join(map(str, lic))
    if not lic:
        lic = next((t.split(":", 1)[1] for t in tags(info) if t.startswith("license:")), None)
    return str(lic) if lic else None

def languages(info):
    card = getattr(info, "card_data", None)
    value = card.get("language") if isinstance(card, dict) else getattr(card, "language", None)
    ls = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    ls = ls + [m.group(1) for t in tags(info) if (m := LANG.match(t))]
    distinct = list(dict.fromkeys(str(x).strip().lower() for x in ls
                                  if isinstance(x, str) and 1 <= len(x) <= 20))
    return ",".join(distinct[:24]) if distinct else "und"

def enrich(api, rid, relevance):
    try:
        info = api.dataset_info(rid, files_metadata=True)
        children = list(getattr(info, "siblings", None) or [])
        files = [str(getattr(x, "rfilename", "")).lower() for x in children]
        direct = sum(x.endswith(DIRECT) for x in files)
        packed = sum(x.endswith(PACKED) for x in files)
        if direct:
            evidence = "direct_audio"
        elif packed:
            evidence = "container_unverified"
        elif any("audio" in t.lower() or "speech" in t.lower() for t in tags(info)):
            evidence = "tag_unverified"
        else:
            return None, "no_audio_evidence"
        if relevance < 3 and not direct:
            return None, "weak_relevance"
        sized = [x.size for x in children if isinstance(getattr(x, "size", None), int)]
        gated = getattr(info, "gated", False)
        private = bool(getattr(info, "private", False))
        tier = "A" if relevance >= 4 and direct else "B" if relevance >= 4 and packed else "C"
        return {
            "name": repo_id(info) or rid,
            "lang": languages(info), "hours": None,
            "kind": "spoken-dialogue candidate" if relevance >= 4 else "speech auxiliary candidate",
            "access": "제한" if private else "신청" if gated else "공개",
            "notes": {"direct_audio": "Direct files; validate actual speech/dialogue/channels",
                      "container_unverified": "TAR/Parquet present, audio content not verified",
                      "tag_unverified": "Audio tag only; playable audio NOT verified"}[evidence],
            "url": f"https://huggingface.co/datasets/{rid}",
            "group": "A" if tier == "A" else "B",
            "train": False, "dup": "",
            "verified": "HF repository API metadata", "source": "huggingface",
            "license": license_of(info), "audio_evidence": evidence,
            "quality_tier": tier, "file_count": len(children),
            "direct_audio_files": direct, "container_files": packed,
            "size_gib_known": round(sum(sized) / 1073741824, 4) if sized else None,
            "size_unknown_files": len(children) - len(sized),
            "gated": str(gated), "candidate_score": relevance,
            "needs_human_review": True
        }, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:140]}"

def discover(api, args):
    candidates = {}
    queries = TERMS[:args.max_queries]
    for index, word in enumerate(queries, 1):
        try:
            count = 0
            for info in api.list_datasets(search=word, limit=args.per_query):
                rid = repo_id(info)
                if not rid:
                    continue
                relevance = score_name(rid, tags(info))
                if relevance < 3:
                    continue
                candidates[rid] = max(candidates.get(rid, 0), relevance)
                count += 1
            print(f"[search {index}/{len(queries)}] {word}: {count} matches, {len(candidates)} unique", flush=True)
        except Exception as exc:
            print(f"[search {index}] {word}: {exc}", file=sys.stderr)
    if args.catalog.is_file():
        legacy = json.loads(args.catalog.read_text(encoding="utf-8"))
        for row in legacy.get("datasets", []):
            m = re.match(r"https://huggingface\.co/datasets/([^/?#]+/[^/?#]+)", row.get("url", ""))
            if m:
                candidates[m.group(1)] = max(7, candidates.get(m.group(1), 0))
    return sorted(candidates.items(), key=lambda x: (-x[1], x[0].lower()))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--per-query", type=int, default=120)
    p.add_argument("--max-queries", type=int, default=len(TERMS))
    p.add_argument("--max-verify", type=int, default=1400)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    p.add_argument("--output", type=Path, default=Path("data/researched_datasets_500"))
    args = p.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    candidates = discover(api, args)
    print(f"API-verifying up to {min(args.max_verify, len(candidates))} candidate repos", flush=True)
    found, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs = {pool.submit(enrich, api, rid, score): rid
              for rid, score in candidates[:args.max_verify]}
        for i, future in enumerate(as_completed(fs), 1):
            row, err = future.result()
            if row:
                found.append(row)
            elif err:
                errors.append({"repo": fs[future], "reason": err})
            if i % 50 == 0:
                print(f"[verify {i}/{len(fs)}] eligible={len(found)}", flush=True)
    priority = {"A": 0, "B": 1, "C": 2}
    found.sort(key=lambda r: (priority[r["quality_tier"]], -r["candidate_score"], r["name"].lower()))
    selected = found[:args.target]
    for n, r in enumerate(selected, 1):
        r["id"] = n
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "requested": args.target, "found": len(selected),
                     "searched": len(candidates), "verified": len(fs),
                     "warning": "CANDIDATES only. No claim of 500 actual dialogue audio datasets. "
                                "Container audio, license and stereo channels need human review."},
        "datasets": selected}
    output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    (output.parent / (output.name + "_errors.json")).write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Result:", len(selected), "/", args.target,
          "verified HF repository candidates; tiers=", dict(Counter(r["quality_tier"] for r in selected)))
    if len(selected) < args.target:
        print("Insufficient verified candidates; did NOT fabricate missing entries.")
    print("CSV:", output.with_suffix(".csv"), "JSON:", output.with_suffix(".json"))

if __name__ == "__main__":
    main()
