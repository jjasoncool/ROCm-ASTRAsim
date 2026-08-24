#!/usr/bin/env python3
"""Trace RDMA flows entering (send_flow) and leaving (qp_finish) the ns-3 backend.

Multi-dimensional ring AllReduce on a Twisted Torus deadlocks after roughly 5,337
flows (astra-sim issue #137, see deadlock-reproduction/); comparing the two
counters shows which flows were issued but never completed. Sampling is sparse so
it does not drown a multi-day 128-node run. Diagnostic only.

NOTE: entry.h ships CRLF, so lines are inserted with the file's own terminator.

Usage: python3 entry_flow_diagnostics.py <astra-sim-root>
"""
import sys

REL = "astra-sim/network_frontend/ns3/entry.h"
MARKER = "[DIAG] send_flow"

SEND_FLOW = (
    '  static uint64_t send_flow_count = 0; send_flow_count++; '
    'if (send_flow_count <= 20 || send_flow_count % 1000 == 0 || '
    '(send_flow_count >= 5330 && send_flow_count <= 5350)) '
    'fprintf(stderr, "[DIAG] send_flow #%lu src=%d dst=%d size=%d tag=%d\\n", '
    'send_flow_count, src_id, dst, maxPacketCount, tag);'
)
QP_FINISH = (
    '  static uint64_t qp_finish_count = 0; qp_finish_count++; '
    'if (qp_finish_count <= 20 || qp_finish_count % 1000 == 0 || '
    '(qp_finish_count >= 5330 && qp_finish_count <= 5350)) '
    'fprintf(stderr, "[DIAG] qp_finish #%lu src=%u dst=%u size=%lu tag=%u sport=%u\\n", '
    'qp_finish_count, ip_to_node_id(q->sip), ip_to_node_id(q->dip), q->m_size, '
    'q->sport, q->dport);'
)


def insert_after(src, anchors, new_line):
    """Insert new_line after a unique run of consecutive lines matching anchors."""
    nl = "\r\n" if "\r\n" in src else "\n"
    lines = src.split(nl)
    hits = [i for i in range(len(lines) - len(anchors) + 1)
            if all(a in lines[i + k] for k, a in enumerate(anchors))]
    if len(hits) != 1:
        sys.exit(f"[entry_flow_diagnostics] ERROR: anchor matched {len(hits)} places, "
                 f"expected 1 (upstream changed?): {anchors}")
    lines.insert(hits[0] + len(anchors), new_line)
    return nl.join(lines)


path = f"{sys.argv[1] if len(sys.argv) > 1 else '/workspace/astra-sim'}/{REL}"
try:
    with open(path, newline="") as f:
        src = f.read()
except OSError as e:
    sys.exit(f"[entry_flow_diagnostics] ERROR: {e}")

if MARKER in src:
    print(f"[entry_flow_diagnostics] already patched: {REL}")
    sys.exit(0)

# send_flow's signature wraps onto a second line; that continuation is unique.
src = insert_after(src, ["void *fun_arg, int tag) {"], SEND_FLOW)
# ip_to_node_id(q->sip) appears twice, so pin it to qp_finish's signature.
src = insert_after(src, ["void qp_finish(FILE *fout",
                         "uint32_t sid = ip_to_node_id(q->sip)"], QP_FINISH)

with open(path, "w", newline="") as f:
    f.write(src)
print(f"[entry_flow_diagnostics] patched: {REL}")
