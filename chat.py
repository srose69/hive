#!/usr/bin/env python3
"""
chat.py — interactive PT-style completion + routing inspector for Hive v5

Usage:
  python chat.py --config config_fss1str.yaml [options]

Options:
  --config PATH         yaml config (required)
  --checkpoint PATH     hive.pt path (default: out_dir/hive.pt)
  --temp FLOAT          sampling temperature (default: 0.8)
  --top-p FLOAT         nucleus sampling cutoff (default: 0.92)
  --max-new INT         max new tokens per completion (default: 128)
  --router-temp FLOAT   router logit temperature, <1 sharpens, >1 diffuses (default: 1.0)
  --top-k-route INT     override top-k active cubes (default: cfg.top_x)
  --show-route          always print routing (default: only on /route command)
  --no-color            disable ANSI color output

Commands (type in prompt):
  /route                show last routing decision
  /temp T               set sampling temperature
  /router-temp T        set router temperature
  /top-k K              set top-k active cubes
  /top-p P              set top-p nucleus
  /max N                set max new tokens
  /reset                clear conversation history
  /cubes                show per-cube info from cluster_eval
  /help                 show this help
  /quit or /exit        exit
"""

import os, sys, json, math, argparse, re
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hive import (
    load_config, load_tokenizer, tok_vocab, tok_eos,
    HiveModel, load_pt, sparsemax, Config
)

# ── ANSI colours ──────────────────────────────────────────────────────────────

def ansi(code): return f'\033[{code}m'
RESET = ansi(0); BOLD = ansi(1); DIM = ansi(2)
CYAN = ansi(36); GREEN = ansi(32); YELLOW = ansi(33); MAGENTA = ansi(35); RED = ansi(31)

def colored(text, *codes, use_color=True):
    if not use_color: return text
    return ''.join(codes) + text + RESET

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(config_path, ckpt_path=None):
    cfg = load_config(config_path)
    ckpt = ckpt_path or os.path.join(cfg.out_dir, 'hive.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')
    sd = load_pt(ckpt, map_location='cpu')
    vocab_size = sd.get('vocab_size') or cfg.raw.get('tokenizer_vocab_size', 16384)
    raw = sd.get('cfg', cfg.raw)
    model_cfg = Config(raw)
    model_cfg.derive(vocab_size)
    model_cfg.config_path = config_path
    tok_path = os.path.join(model_cfg.out_dir, 'tokenizer.json')
    tok = load_tokenizer(tok_path)
    m = HiveModel(model_cfg)
    own_sd = m.state_dict()
    filtered = {k: v for k, v in sd['model'].items()
                if k in own_sd and v.shape == own_sd[k].shape}
    m.load_state_dict(filtered, strict=False)
    m.eval(); m.move_rope()
    return m, tok, model_cfg

# ── Routing ───────────────────────────────────────────────────────────────────

def compute_route(model, tokens, router_temp=1.0, top_k=None):
    cfg = model.cfg
    if top_k is None:
        top_k = cfg.top_x
    with torch.no_grad():
        emb, h0 = model.encode_context(tokens)
        logits = model.planner(emb, h0)[0]   # (total_cubes,)
    route = []
    raw_weights = []
    for l in range(cfg.layers):
        a, b = cfg.layer_slice(l)
        zl = logits[a:b] / max(router_temp, 1e-6)
        w = sparsemax(zl, dim=-1)
        nz = (w > 0).nonzero(as_tuple=True)[0]
        order = nz[torch.argsort(w[nz], descending=True)][:top_k]
        pairs = [(int(i), float(w[i])) for i in order]
        route.append(pairs)
        raw_weights.append(w.tolist())
    return route, raw_weights

def format_route(route, raw_weights=None, use_color=True):
    lines = []
    for l, layer in enumerate(route):
        parts = []
        for c, w in layer:
            cube_str = f'C{c}:{w:.3f}'
            if w > 0.7:
                cube_str = colored(cube_str, BOLD, GREEN, use_color=use_color)
            elif w > 0.3:
                cube_str = colored(cube_str, YELLOW, use_color=use_color)
            else:
                cube_str = colored(cube_str, DIM, use_color=use_color)
            parts.append(cube_str)
        active = colored(f'L{l}', CYAN, BOLD, use_color=use_color)
        lines.append(f'  {active}: [{", ".join(parts)}]')
    return '\n'.join(lines)

# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tok, tokens, route, max_new=128, temp=0.8, top_p=0.92,
             stream=True, use_color=True):
    eos = tok_eos(tok)
    generated = []
    with torch.no_grad():
        logits_out, cache = model.prefill(tokens, route)
        pos = tokens.shape[1]
        for _ in range(max_new):
            lgt = logits_out[0, -1].float()
            if temp > 0:
                lgt = lgt / temp
                probs = torch.softmax(lgt, dim=-1)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=0)
                    mask = (cum - sp) > top_p
                    sp[mask] = 0.0
                    sp /= sp.sum().clamp(min=1e-9)
                    nxt = si[torch.multinomial(sp, 1)].item()
                else:
                    nxt = torch.multinomial(probs, 1).item()
            else:
                nxt = torch.argmax(lgt).item()
            if nxt == eos:
                break
            generated.append(nxt)
            if stream:
                try:
                    piece = tok.decode([nxt])
                    print(colored(piece, GREEN, use_color=use_color), end='', flush=True)
                except Exception:
                    pass
            ntok = torch.tensor([[nxt]], dtype=torch.long)
            logits_out, cache = model.decode_step(ntok, route, cache, pos)
            pos += 1
    if stream:
        print()
    return tok.decode(generated)

# ── Cube info ─────────────────────────────────────────────────────────────────

def load_cluster_eval(out_dir):
    p = os.path.join(out_dir, 'cluster_eval.json')
    if os.path.exists(p):
        return json.load(open(p))
    return None

def show_cubes(model, cluster_eval, use_color=True):
    cfg = model.cfg
    ce = cluster_eval
    print(colored('=== Cube specialization ===', BOLD, CYAN, use_color=use_color))
    for l in range(cfg.layers):
        print(colored(f'Layer {l}:', BOLD, use_color=use_color))
        for ci in range(cfg.cubes[l]):
            alpha = float(model.layers[l][ci].alpha.detach())
            ppl_key = f'layer{l}_cube{ci}'
            ppl = ce['single'].get(ppl_key, '?') if ce else '?'
            ppl_str = f'{ppl:.2f}' if isinstance(ppl, float) else str(ppl)
            if isinstance(ppl, float) and ppl < 3:
                ppl_str = colored(ppl_str, GREEN, BOLD, use_color=use_color)
            elif isinstance(ppl, float) and ppl < 10:
                ppl_str = colored(ppl_str, YELLOW, use_color=use_color)
            print(f'  C{ci}: alpha={alpha:.4f}  single_ppl={ppl_str}')

# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Hive v5 interactive chat')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--top-p', type=float, default=0.92)
    ap.add_argument('--max-new', type=int, default=128)
    ap.add_argument('--router-temp', type=float, default=1.0)
    ap.add_argument('--top-k-route', type=int, default=None)
    ap.add_argument('--show-route', action='store_true')
    ap.add_argument('--no-color', action='store_true')
    args = ap.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    print(colored('Loading model...', DIM, use_color=use_color), flush=True)
    model, tok, cfg = load_model(args.config, args.checkpoint)
    cluster_eval = load_cluster_eval(cfg.out_dir)
    eos = tok_eos(tok)

    # state
    state = {
        'temp': args.temp,
        'top_p': args.top_p,
        'max_new': args.max_new,
        'router_temp': args.router_temp,
        'top_k_route': args.top_k_route or cfg.top_x,
        'show_route': args.show_route,
        'history': [],      # list of token ids (flat)
        'last_route': None,
        'last_raw_weights': None,
    }

    print(colored(f'Hive v5 — {cfg.summary()}', BOLD, use_color=use_color))
    print(colored(f'temp={state["temp"]} top_p={state["top_p"]} max_new={state["max_new"]} '
                  f'router_temp={state["router_temp"]} top_k_route={state["top_k_route"]}',
                  DIM, use_color=use_color))
    print(colored('Type /help for commands. Ctrl-C or /quit to exit.', DIM, use_color=use_color))
    print()

    def parse_float(s, name):
        try: return float(s)
        except: print(colored(f'Invalid {name}: {s}', RED, use_color=use_color)); return None

    def parse_int(s, name):
        try: return int(s)
        except: print(colored(f'Invalid {name}: {s}', RED, use_color=use_color)); return None

    while True:
        try:
            prompt = input(colored('>>> ', BOLD, CYAN, use_color=use_color))
        except (EOFError, KeyboardInterrupt):
            print(); break

        prompt = prompt.strip()
        if not prompt:
            continue

        # ── commands ──────────────────────────────────────────────────────────
        if prompt.startswith('/'):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd in ('/quit', '/exit'):
                break

            elif cmd == '/help':
                print(__doc__)

            elif cmd == '/route':
                if state['last_route']:
                    print(colored('Last route:', BOLD, use_color=use_color))
                    print(format_route(state['last_route'], state['last_raw_weights'], use_color))
                else:
                    print(colored('No route yet.', DIM, use_color=use_color))

            elif cmd == '/cubes':
                show_cubes(model, cluster_eval, use_color)

            elif cmd == '/reset':
                state['history'] = []
                print(colored('History cleared.', DIM, use_color=use_color))

            elif cmd == '/temp' and len(parts) == 2:
                v = parse_float(parts[1], 'temp')
                if v is not None: state['temp'] = v; print(f'temp={v}')

            elif cmd == '/router-temp' and len(parts) == 2:
                v = parse_float(parts[1], 'router-temp')
                if v is not None: state['router_temp'] = v; print(f'router_temp={v}')

            elif cmd == '/top-k' and len(parts) == 2:
                v = parse_int(parts[1], 'top-k')
                if v is not None: state['top_k_route'] = v; print(f'top_k_route={v}')

            elif cmd == '/top-p' and len(parts) == 2:
                v = parse_float(parts[1], 'top-p')
                if v is not None: state['top_p'] = v; print(f'top_p={v}')

            elif cmd == '/max' and len(parts) == 2:
                v = parse_int(parts[1], 'max')
                if v is not None: state['max_new'] = v; print(f'max_new={v}')

            elif cmd == '/show-route':
                state['show_route'] = not state['show_route']
                print(f'show_route={state["show_route"]}')

            else:
                print(colored(f'Unknown command: {cmd}. Type /help.', RED, use_color=use_color))
            continue

        # ── completion ────────────────────────────────────────────────────────
        new_ids = tok.encode(prompt)
        history_ids = state['history'] + new_ids
        # trim to seq_len
        max_ctx = cfg.seq_len - state['max_new'] - 1
        if len(history_ids) > max_ctx:
            history_ids = history_ids[-max_ctx:]

        tokens = torch.tensor([history_ids], dtype=torch.long)

        route, raw_w = compute_route(model, tokens,
                                     router_temp=state['router_temp'],
                                     top_k=state['top_k_route'])
        state['last_route'] = route
        state['last_raw_weights'] = raw_w

        if state['show_route']:
            print(colored('Route:', BOLD, use_color=use_color))
            print(format_route(route, raw_w, use_color))

        print(colored('', GREEN, use_color=use_color), end='')
        out = generate(model, tok, tokens, route,
                       max_new=state['max_new'],
                       temp=state['temp'],
                       top_p=state['top_p'],
                       stream=True,
                       use_color=use_color)

        # append to history (prompt + completion)
        state['history'] = history_ids + tok.encode(out)
        # cap history
        if len(state['history']) > cfg.seq_len * 2:
            state['history'] = state['history'][-cfg.seq_len:]

    print(colored('Bye.', DIM, use_color=use_color))


if __name__ == '__main__':
    main()
